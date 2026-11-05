"""asr.py —— 语音转写（course-video-notes 的「识别层」之二）。

职责（对应 references/DATA-CONTRACT.md 第 4 节）：
    把视频/音频转成带时间戳的逐字稿，写出两份产物：
        out_dir/transcript.json   机器可读（align 阶段吃它）
        out_dir/transcript.srt    人能直接看的字幕

    {
      "engine": "faster-whisper|srt-import",
      "model": "medium",
      "language": "zh",
      "duration": 3725.4,
      "segments": [
        {"i": 0, "start": 12.4, "end": 18.9, "text": "看这个图，考试必考。", "slide": 0}
      ]
    }

三条硬约束（都是踩过的坑，改代码时别踩回去）：
1. **模型下载必须走镜像**。默认 HF 官方站连不上（curl 返回 000），必须
   HF_ENDPOINT=https://hf-mirror.com；这里在 import faster_whisper 之前 setdefault，
   用户自己设了别的镜像也不覆盖。
2. **缓存目录必须可写**。~/.cache 在本机不可写（PEP 668 沙箱 + 受管 Mac），
   所以 HF_HOME / XDG_CACHE_HOME 都指到 $TMPDIR 下的目录。
3. **faster-whisper 惰性导入**。它依赖 ctranslate2 等一堆二进制轮子，不是所有
   环境都有；缺了就抛中文 CvnError 告诉用户怎么装，而不是 ImportError traceback。

另外提供一个 import_srt()：用户常有现成字幕文件，直接导入比重新转写强得多。

对外接口：
    run_asr(out_dir, video, *, model="medium", language="zh", device="auto",
            compute_type="auto", force=False) -> dict
    import_srt(out_dir, srt_path) -> dict
    probe_asr() -> dict
    write_srt(path, segments) -> Path
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

try:  # 允许 `python3 scripts/cvn/asr.py ...` 直跑，也允许 `from cvn import asr`
    from . import util
except ImportError:  # pragma: no cover - 仅在脚本直跑时触发
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import util  # type: ignore[no-redef]


# --------------------------------------------------------------------------
# 环境准备：镜像 + 可写缓存（必须在 import faster_whisper 之前调用）
# --------------------------------------------------------------------------

# 官方站不通的机器上这是唯一能下的入口；用户显式设置过就不覆盖。
DEFAULT_HF_ENDPOINT = "https://hf-mirror.com"

# 采样率固定 16k 单声道：whisper 的前端就是这么吃的，先转好省得它自己再解
ASR_SAMPLE_RATE = 16000

# 合并过短片段的阈值（秒）：比这还短的片段并到相邻片段里，
# 否则 SRT 会出现一堆「嗯」「对」的碎片，阅读体验很差。
MIN_SEGMENT_SECONDS = 1.2

# 单个片段的最长时长：whisper 偶尔会吐 30 秒长句，字幕读不完
MAX_SEGMENT_SECONDS = 15.0

# 中文之间做句子拼接时用的断句符
_SENT_END = "。！？!?；;…"


def _prepare_env() -> dict[str, Any]:
    """把 HuggingFace 相关环境变量指到「能用 + 可写」的地方，返回生效值的快照。

    幂等：多次调用不会互相打架，也不会覆盖用户已有的设置（除了缓存目录，
    那个必须落到可写位置，否则下载一定失败）。

    返回快照会写进 meta / 日志，方便用户复现问题。
    """
    # 1) 镜像：默认官方站在本机不通，必须给镜像兜底
    endpoint = os.environ.get("HF_ENDPOINT", "").strip()
    if not endpoint:
        os.environ["HF_ENDPOINT"] = DEFAULT_HF_ENDPOINT
        endpoint = DEFAULT_HF_ENDPOINT
    else:
        os.environ.setdefault("HF_ENDPOINT", endpoint)

    # 2) 缓存根目录：优先用户指定的 HF_HOME，其次 ~/.cache/huggingface（能写就复用
    #    已下载模型），最后才落到临时目录。有些受管 Mac 上 ~/.cache 不可写，
    #    所以必须保留临时目录兜底。
    cache_root: Path | None = None
    configured = os.environ.get("HF_HOME", "").strip()
    if configured:
        candidate = Path(configured)
        if util.is_writable_dir(candidate):
            cache_root = candidate
    if cache_root is None:
        default_cache = Path.home() / ".cache" / "huggingface"
        try:
            default_cache.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        if util.is_writable_dir(default_cache):
            cache_root = default_cache
    if cache_root is None:
        # 不能直接信 ~/.cache（可能不可写），用 util 挑一个真能写的临时根
        cache_root = util.pick_temp_root() / "hf"
        util.ensure_dir(cache_root)

    os.environ["HF_HOME"] = str(cache_root)
    hub_cache = cache_root / "hub"
    util.ensure_dir(hub_cache)
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hub_cache))
    # 部分旧版本读这个
    os.environ.setdefault("XDG_CACHE_HOME", str(cache_root / "xdg"))
    util.ensure_dir(os.environ["XDG_CACHE_HOME"])
    # 下载体积大，xet 传输在本机不稳定，禁掉更稳（可用性优先）
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    # 关掉 symlink 警告噪声
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    return {
        "HF_ENDPOINT": os.environ["HF_ENDPOINT"],
        "HF_HOME": os.environ["HF_HOME"],
        "HUGGINGFACE_HUB_CACHE": os.environ.get("HUGGINGFACE_HUB_CACHE", ""),
    }


def _import_faster_whisper():
    """惰性导入 faster_whisper；缺依赖时给中文可读错误。

    这个函数必须在 _prepare_env() 之后调用：faster_whisper 在 import 期就会
    读环境变量决定 hub 缓存位置，先 import 后设环境就晚了。
    """
    _prepare_env()
    try:
        from faster_whisper import WhisperModel  # type: ignore
        return WhisperModel
    except ImportError:
        pass
    except Exception as exc:  # ctranslate2 之类的二进制装坏了
        raise util.CvnError(
            "faster-whisper 已安装但加载失败：" + str(exc) + "\n"
            "它依赖 ctranslate2 / av 的原生库，二进制不匹配时会出现这种错误。\n"
            "建议重装：\n"
            "  UV_CACHE_DIR=/tmp/uvcache UV_PYTHON_INSTALL_DIR=/tmp/uvpy \\\n"
            "    uv pip install faster-whisper"
        ) from exc

    raise util.CvnError(
        "未安装 faster-whisper，无法做语音转写。\n"
        "安装方法（本机 ~/.cache 不可写，必须把 uv 缓存指到可写目录）：\n"
        "  UV_CACHE_DIR=/tmp/uvcache UV_PYTHON_INSTALL_DIR=/tmp/uvpy \\\n"
        "    uv pip install faster-whisper\n"
        "装好后如果模型下载失败，设置镜像：\n"
        f"  export HF_ENDPOINT={DEFAULT_HF_ENDPOINT}\n"
        "如果你已经有字幕文件，也可以跳过转写直接导入：cvn asr --srt <文件>"
    )


# --------------------------------------------------------------------------
# 探测
# --------------------------------------------------------------------------


def probe_asr() -> dict:
    """探测转写能力，供 CLI 的 doctor 打印。

    返回：
        {
          "faster_whisper": {"available": bool, "version": str|None, "error": str},
          "ffmpeg": str|None,
          "ffprobe": str|None,
          "env": {...HF 相关环境变量快照...},
        }
    不会真正加载模型（那要几秒 + 可能触发下载），只检查能不能 import。
    """
    info: dict[str, Any] = {
        "ffmpeg": util.which("ffmpeg"),
        "ffprobe": util.which("ffprobe"),
        "faster_whisper": {"available": False, "version": None, "error": ""},
        "env": {},
    }
    try:
        info["env"] = _prepare_env()
    except util.CvnError as exc:
        info["env"] = {"error": str(exc)}

    try:
        import faster_whisper  # type: ignore

        info["faster_whisper"] = {
            "available": True,
            "version": getattr(faster_whisper, "__version__", "unknown"),
            "error": "",
        }
    except ImportError as exc:
        info["faster_whisper"] = {"available": False, "version": None, "error": str(exc)}
    except Exception as exc:  # 二进制装坏
        info["faster_whisper"] = {"available": False, "version": None, "error": str(exc)}
    return info


# --------------------------------------------------------------------------
# 媒体：时长探测 / 抽音轨
# --------------------------------------------------------------------------


def _media_duration(path: Path) -> float:
    """用 ffprobe 读时长（秒）；拿不到返回 0.0（调用方要能容忍）。"""
    ffprobe = util.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=60, errors="replace",
        )
        return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except (ValueError, subprocess.TimeoutExpired, OSError):
        return 0.0


def _has_audio_stream(path: Path) -> bool:
    """检查媒体里有没有音轨（没音轨的录屏很常见，要给出人能看懂的提示）。"""
    ffprobe = util.which("ffprobe")
    if not ffprobe:
        return True  # 探测不了就当有，交给后面 ffmpeg 报错
    try:
        proc = subprocess.run(
            [ffprobe, "-v", "error", "-select_streams", "a",
             "-show_entries", "stream=index", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=60, errors="replace",
        )
        return bool((proc.stdout or "").strip())
    except (subprocess.TimeoutExpired, OSError):
        return True


def extract_audio(video: str | os.PathLike[str], workdir: str | os.PathLike[str] | None = None) -> tuple[Path, float]:
    """从任意媒体里抽出 16kHz 单声道 wav，返回 (wav 路径, 媒体时长秒)。

    为什么自己抽而不是让 whisper 直接读视频：
      - faster-whisper 内部用 av 解码，遇到某些容器（录屏的 flv/ts）会失败；
      - 统一转成 16k 单声道 wav 后，转写更快也更稳，还能顺便拿到时长。

    参数：
        video:   视频/音频文件路径
        workdir: 放临时 wav 的目录；None 则用 util.pick_temp_root() 下的新建目录
    异常：
        util.CvnError —— 文件不存在 / 没有音轨 / ffmpeg 失败（都是中文消息）
    """
    # 支持 http(s) 直链：URL 不能走 Path()，否则 https:// 会被规范化成 https:/
    raw = str(video)
    if re.match(r"^(https?|rtmp|rtsp|ftp)://", raw, re.I):
        src = raw                      # 交给 ffmpeg 流式读取
    else:
        src = Path(video)
        if not src.exists():
            raise util.CvnError(f"找不到媒体文件：{src}")
        if not src.is_file():
            raise util.CvnError(f"不是文件：{src}")

    ffmpeg = util.which("ffmpeg")
    if not ffmpeg:
        raise util.CvnError(
            "找不到 ffmpeg，无法抽取音轨。\n"
            "安装方法：brew install ffmpeg"
        )

    duration = _media_duration(src)
    if not _has_audio_stream(src):
        raise util.CvnError(
            f"这个文件没有音轨，无法转写：{src.name}\n"
            "常见原因：录屏时静音、只录了画面，或视频本身就是无声的。\n"
            "如果是这种情况，可以用 --srt 参数导入现成字幕，或只做课件 OCR。"
        )

    if workdir is not None:
        out_dir = util.ensure_dir(workdir)
    else:
        out_dir = util.ensure_dir(util.pick_temp_root() / f"asr-{int(time.time())}")
    wav = out_dir / "audio16k.wav"

    args = [
        ffmpeg, "-nostdin", "-y", "-v", "error",
        "-i", str(src),
        "-vn", "-ac", "1", "-ar", str(ASR_SAMPLE_RATE),
        "-c:a", "pcm_s16le",
        str(wav),
    ]
    proc = subprocess.run(args, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not wav.exists() or wav.stat().st_size == 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        raise util.CvnError(
            f"ffmpeg 抽取音轨失败（exit {proc.returncode}）：{src.name}\n{tail}\n"
            "提示：文件可能已损坏，或者是不支持的编码格式。可以先用播放器确认能播。"
        )
    return wav, duration


# --------------------------------------------------------------------------
# 片段整理
# --------------------------------------------------------------------------



# --------------------------------------------------------------------------
# 繁体 -> 简体
# --------------------------------------------------------------------------
#
# whisper 对大陆普通话的输出偶尔整段是繁体（模型训练数据混杂）。
# 引入 opencc 太重，这里内置一张高频单字映射表，覆盖绝大多数常见繁体字。
# 只做单字映射、不做词级转换，所以不会改动任何用词（「头发/发展」这类
# 简繁一对多的情况极少出现在课件口播里，可接受）。

_T2S_TRAD = "愛礙襖奧幣辨辯標錶別賓補才財採參慘燦蒼艙層嘗長償廠場車徹塵陳稱懲遲馳沖醜礎處觸傳瘡闖創錘純詞辭賜從叢湊蹙錯帶貸單擔膽誕彈當擋黨盪檔島禱燈鄧敵滌遞締點澱電墊釣調諜疊釘訂東動棟鬥獨讀賭鍍鍛隊對噸鈍奪墮額惡兒爾餌發罰閥礬釩飯範販訪紡飛廢費紛墳奮憤糞豐楓鋒風瘋馮縫諷鳳膚扶撫復負訃賦傅富縛該鈣蓋幹桿趕綱崗鋼槓鎬個給鞏貢構購夠蠱顧剛關觀館慣貫廣歸龜規軌貴劊滾棍鍋國過駭韓漢號閡鶴賀橫轟紅後壺護滬戶花華話劃畫懷壞歡環還緩換喚煥黃謊揮輝毀賄穢會匯諱燴鬱暈渾渾誆誨貨禍獲擊機積譏嘰磯雞績緝極輯級擠幾薊劑濟計記際繼紀夾莢頰賈價駕殲監堅箋艱撿簡儉減薦檻鑑踐賤見鍵艦劍餞漸濺澗將漿蔣槳獎講醬膠澆驕嬌攪絞餃腳繳轎較階節潔結誡屆緊謹錦僅進晉燼盡勁荊莖鯨驚經靜境鏡競敬痙舊劇據懼俱颶鋸軍鈞駿開凱顆殼課墾懇摳褲誇塊儈寬礦曠況虧巋窺饋潰闊蠟臘蘭瀾攬欖懶爛濫撈勞澇樂類淚籬離裡鯉禮麗厲勵礫歷瀝隸倆聯蓮連鐮憐簾斂臉鏈戀煉練糧涼兩輛諒遼療獵臨鄰鱗凜賃齡鈴靈嶺領餾劉龍聾嚨籠壟攏隴樓婁簍蘆盧爐魯鹵虜錄慮濾巒攣亂掄輪倫淪綸論羅絡駱麼馬罵瑪碼螞買賣邁麥脈瞞饅滿謾貓門悶們萌蒙夢彌謎覓冪綿緬廟滅憫閩鳴銘謬謀畝鈉納難撓腦惱鬧餒內擬膩釀鳥聶孽檸擰鈕紐膿農濃諾歐毆嘔藕盤龐拋賠噴鵬騙飄頻貧蘋憑評潑鋪僕譜齊騎豈啟棄氣訖遷僉簽謙錢鉗潛淺譴槍嗆腔強牆搶橋喬僑翹竅竊親欽侵寢慶輕傾請磬窮趨區軀驅齲顴勸詮卻鵲讓饒擾熱認韌榮絨軟銳閏潤灑薩賽傘喪騷澀殺剎紗篩曬刪閃陝贍傷賞燒賒設紳審腎滲聲繩勝師獅濕詩時蝕實識駛勢適釋飾視試壽獸樞輸屬術樹豎帥雙誰稅順說碩絲飼鬆聳訟頌誦蘇訴肅雖隨綏歲孫筍損縮瑣鎖臺態攤貪癱灘壇譚談嘆湯燙濤討騰謄題體屜條鐵廳聽統頭圖塗團頹蛻脫鴕馱橢窪襪彎灣萬網韋違圍為濰維偉偽緯衛溫聞紋穩問甕撾蝸渦窩嗚鎢無吳蕪霧務誤錫犧襲習銑戲細蝦轄峽俠狹廈嚇纖鮮賢閒嫻顯險現獻縣餡羨憲線陷鄉詳響項蕭銷曉嘯協脅挾攜諧寫瀉謝鋅薪興洶胸繡鏽虛噓須許敘緒續軒懸選癬學勳詢尋馴訓訊遜壓鴉鴨啞亞煙閹鹽嚴顏閻豔厭硯彥諺驗鴦楊揚癢樣謠搖堯遙藥爺業葉醫銥儀頤遺顧彝蟻義億憶議詣譯異陰蔭銀隱應嬰鷹櫻贏穎憂優尤猶油誘魷輿魚娛漁渝瑜榆逾愉預馭鴛淵轅園員圓緣遠願約躍鑰粵悅閱雲勻隕運蘊醞暈韻雜災載贊暫贓髒鑿棗責擇則澤賊贈軋鍘閘詐齋債氈盞斬輾嶄戰張漲帳賬脹趙蟄轍鍺這貞針偵診鎮陣掙睜猙幀鄭證織職執紙摯擲幟質滯鐘終種腫眾謅軸皺晝燭囑矚鑄築專轉賺樁庄裝壯狀錐贅諄濁茲資漬蹤綜總縱鄒詛組鑽醉鱒尊遵"
_T2S_EXTRA_T = "來繞麴龜龍鐵門問題開關時間間長們東車馬鳥語語氣風雲電機學學生員圓園遠運動靜聽見貝見長門"
_T2S_EXTRA_S = "来绕曲龟龙铁门问题开关时间间长们东车马鸟语语气风云电机学学生员圆园远运动静听见贝见长门"
_T2S_SIMP = "爱碍袄奥币辨辩标表别宾补才财采参惨灿苍舱层尝长偿厂场车彻尘陈称惩迟驰冲丑础处触传疮闯创锤纯词辞赐从丛凑蹙错带贷单担胆诞弹当挡党荡档岛祷灯邓敌涤递缔点淀电垫钓调谍叠钉订东动栋斗独读赌镀锻队对吨钝夺堕额恶儿尔饵发罚阀矾钒饭范贩访纺飞废费纷坟奋愤粪丰枫锋风疯冯缝讽凤肤扶抚复负讣赋傅富缚该钙盖干杆赶纲岗钢杠镐个给巩贡构购够蛊顾刚关观馆惯贯广归龟规轨贵刽滚棍锅国过骇韩汉号阂鹤贺横轰红后壶护沪户花华话划画怀坏欢环还缓换唤焕黄谎挥辉毁贿秽会汇讳烩郁晕浑浑诓诲货祸获击机积讥叽矶鸡绩缉极辑级挤几蓟剂济计记际继纪夹荚颊贾价驾歼监坚笺艰捡简俭减荐槛鉴践贱见键舰剑饯渐溅涧将浆蒋桨奖讲酱胶浇骄娇搅绞饺脚缴轿较阶节洁结诫届紧谨锦仅进晋烬尽劲荆茎鲸惊经静境镜竞敬痉旧剧据惧俱飓锯军钧骏开凯颗壳课垦恳抠裤夸块侩宽矿旷况亏岿窥馈溃阔蜡腊兰澜揽榄懒烂滥捞劳涝乐类泪篱离里鲤礼丽厉励砾历沥隶俩联莲连镰怜帘敛脸链恋炼练粮凉两辆谅辽疗猎临邻鳞凛赁龄铃灵岭领馏刘龙聋咙笼垄拢陇楼娄篓芦卢炉鲁卤虏录虑滤峦挛乱抡轮伦沦纶论罗络骆么马骂玛码蚂买卖迈麦脉瞒馒满谩猫门闷们萌蒙梦弥谜觅幂绵缅庙灭悯闽鸣铭谬谋亩钠纳难挠脑恼闹馁内拟腻酿鸟聂孽柠拧钮纽脓农浓诺欧殴呕藕盘庞抛赔喷鹏骗飘频贫苹凭评泼铺仆谱齐骑岂启弃气讫迁佥签谦钱钳潜浅谴枪呛腔强墙抢桥乔侨翘窍窃亲钦侵寝庆轻倾请磬穷趋区躯驱龋颧劝诠却鹊让饶扰热认韧荣绒软锐闰润洒萨赛伞丧骚涩杀刹纱筛晒删闪陕赡伤赏烧赊设绅审肾渗声绳胜师狮湿诗时蚀实识驶势适释饰视试寿兽枢输属术树竖帅双谁税顺说硕丝饲松耸讼颂诵苏诉肃虽随绥岁孙笋损缩琐锁台态摊贪瘫滩坛谭谈叹汤烫涛讨腾誊题体屉条铁厅听统头图涂团颓蜕脱鸵驮椭洼袜弯湾万网韦违围为潍维伟伪纬卫温闻纹稳问瓮挝蜗涡窝呜钨无吴芜雾务误锡牺袭习铣戏细虾辖峡侠狭厦吓纤鲜贤闲娴显险现献县馅羡宪线陷乡详响项萧销晓啸协胁挟携谐写泻谢锌薪兴汹胸绣锈虚嘘须许叙绪续轩悬选癣学勋询寻驯训讯逊压鸦鸭哑亚烟阉盐严颜阎艳厌砚彦谚验鸯杨扬痒样谣摇尧遥药爷业叶医铱仪颐遗顾彝蚁义亿忆议诣译异阴荫银隐应婴鹰樱赢颖忧优尤犹油诱鱿舆鱼娱渔渝瑜榆逾愉预驭鸳渊辕园员圆缘远愿约跃钥粤悦阅云匀陨运蕴酝晕韵杂灾载赞暂赃脏凿枣责择则泽贼赠轧铡闸诈斋债毡盏斩辗崭战张涨帐账胀赵蛰辙锗这贞针侦诊镇阵挣睁狰帧郑证织职执纸挚掷帜质滞钟终种肿众诌轴皱昼烛嘱瞩铸筑专转赚桩庄装壮状锥赘谆浊兹资渍踪综总纵邹诅组钻醉鳟尊遵"
assert len(_T2S_TRAD) == len(_T2S_SIMP), "繁简映射表长度不一致"
assert len(_T2S_EXTRA_T) == len(_T2S_EXTRA_S), "补充映射表长度不一致"
_T2S_TABLE = str.maketrans(_T2S_TRAD + _T2S_EXTRA_T, _T2S_SIMP + _T2S_EXTRA_S)


def _t2s(text: str) -> str:
    """繁体 -> 简体（单字映射，不改词）。对纯简体/英文文本是恒等变换。"""
    if not text:
        return text
    return text.translate(_T2S_TABLE)


def _clean_segment_text(text: str) -> str:
    """清洗一段转写文本：去多余空格、把半角标点留在原样、去掉纯噪声。"""
    s = util.clean_line(text or "")
    # whisper 的中文输出偶尔带前后空白和重复标点
    s = re.sub(r"\s+", " ", s).strip()
    s = re.sub(r"([，。！？；：])\1+", r"\1", s)
    if not os.environ.get("CVN_NO_T2S"):
        # whisper 对大陆普通话有时整段输出繁体，统一转回简体
        s = _t2s(s)
    return s


def _is_noise(text: str) -> bool:
    """判断是不是纯噪声片段（语气词、掌声标记、空字幕）。

    只丢「明显没用」的：单个语气词、括号标记、纯标点。
    注意：中文的两个字很可能是真词（"磁场"），不能按长度一刀切。
    """
    s = util.squash(text)
    if not s:
        return True
    if re.fullmatch(r"[\[\(【（][^\]\)】）]*[\]\)】）]", text.strip()):
        return True  # [掌声] / (音乐)
    return s in {"嗯", "啊", "呃", "哦", "额", "唉", "诶", "哈", "嗯嗯", "啊啊", "哦哦"}


def _merge_segments(
    raw: Sequence[dict[str, Any]],
    *,
    min_duration: float = MIN_SEGMENT_SECONDS,
    max_duration: float = MAX_SEGMENT_SECONDS,
) -> list[dict[str, Any]]:
    """把 whisper 的碎片片段整理成朗读友好的字幕段。

    规则（按顺序）：
      1. 丢掉噪声片段（语气词 / [掌声] / 空文本）。
      2. 过短的片段（< min_duration）并到前一或后一段，避免字幕闪一下就没。
      3. 过长的片段（> max_duration）按标点切成几段，时间按字数比例分配。
      4. 重叠/倒序的时间戳夹正（end <= start 时给个最小值），保证契约的
         「按 start 升序」永远成立。

    返回的每段只有 start/end/text（不含 i/slide），由调用方补。
    """
    cleaned: list[dict[str, Any]] = []
    for item in raw:
        text = _clean_segment_text(str(item.get("text") or ""))
        if _is_noise(text):
            continue
        try:
            start = max(0.0, float(item.get("start") or 0.0))
            end = max(0.0, float(item.get("end") or 0.0))
        except (TypeError, ValueError):
            continue
        if end <= start:
            end = start + 0.6          # 没有时长信息的片段给个最小可见时长
        cleaned.append({"start": start, "end": end, "text": text})

    cleaned.sort(key=lambda s: (s["start"], s["end"]))

    # --- 合并过短片段 ---
    merged: list[dict[str, Any]] = []
    for seg in cleaned:
        if merged:
            prev = merged[-1]
            prev_dur = prev["end"] - prev["start"]
            same_run = seg["start"] - prev["end"] <= 1.0   # 中间几乎没停顿
            if prev_dur < min_duration and same_run:
                prev["end"] = max(prev["end"], seg["end"])
                prev["text"] = _join_texts(prev["text"], seg["text"])
                continue
        merged.append(dict(seg))

    # 第一段如果太短，并入第二段
    if len(merged) >= 2:
        head = merged[0]
        if head["end"] - head["start"] < min_duration and merged[1]["start"] - head["end"] <= 1.0:
            merged[1]["start"] = head["start"]
            merged[1]["text"] = _join_texts(head["text"], merged[1]["text"])
            merged.pop(0)

    # --- 切过长片段 ---
    out: list[dict[str, Any]] = []
    for seg in merged:
        out.extend(_split_long(seg, max_duration))
    return out


def _join_texts(a: str, b: str) -> str:
    """拼两段文本：中文直接连（前一句有句末标点就不用加），拉丁词加空格。"""
    if not a:
        return b
    if not b:
        return a
    if a[-1] in _SENT_END or a[-1] in "，,、":
        return a + b
    if util.has_cjk(a[-1:]) and util.has_cjk(b[:1]):
        return a + b
    return a + " " + b


def _split_long(seg: dict[str, Any], max_duration: float) -> list[dict[str, Any]]:
    """把超过 max_duration 的片段按标点/长度切开，时间按字数比例分配。"""
    start, end = seg["start"], seg["end"]
    text = seg["text"]
    if end - start <= max_duration or len(text) <= 4:
        return [seg]

    pieces = [p for p in re.split(r"(?<=[。！？!?；;，,、])", text) if p.strip()]
    if len(pieces) <= 1:
        # 没有标点可切，硬按字数均分
        n = max(2, int((end - start) / max_duration) + 1)
        size = max(1, len(text) // n)
        pieces = [text[i : i + size] for i in range(0, len(text), size)]

    # 把碎片攒成不超长的组
    groups: list[str] = []
    for piece in pieces:
        if groups and len(groups[-1]) + len(piece) <= max(1, len(text) // 2):
            groups[-1] += piece
        else:
            groups.append(piece)

    total_chars = sum(len(g) for g in groups) or 1
    out: list[dict[str, Any]] = []
    cursor = start
    span = end - start
    for i, group in enumerate(groups):
        share = span * len(group) / total_chars
        seg_end = end if i == len(groups) - 1 else min(end, cursor + share)
        out.append({"start": round(cursor, 3), "end": round(max(seg_end, cursor + 0.3), 3), "text": group.strip()})
        cursor = seg_end
    # 夹正：保证不越界、不倒序
    for item in out:
        item["start"] = max(start, min(item["start"], end))
        item["end"] = max(item["start"] + 0.2, min(item["end"], end))
    return out


def _finalise_segments(segments: Sequence[dict[str, Any]], duration: float) -> list[dict[str, Any]]:
    """补齐契约要求的字段（i / slide），并保证 start 升序、时间不越界。"""
    out: list[dict[str, Any]] = []
    cursor = 0.0
    for i, seg in enumerate(segments):
        start = max(0.0, float(seg.get("start") or 0.0))
        end = max(start + 0.2, float(seg.get("end") or 0.0))
        if duration > 0:
            end = min(end, duration)
            start = min(start, max(0.0, end - 0.2))
        # 契约：segments 按 start 升序 —— 这里顺手修掉轻微倒序
        start = max(start, cursor)
        cursor = start
        out.append(
            {
                "i": i,
                "start": round(start, 3),
                "end": round(end, 3),
                "text": str(seg.get("text") or "").strip(),
                "slide": int(seg.get("slide") or 0),  # align 阶段回填，0 = 未归属
            }
        )
    out.sort(key=lambda s: s["start"])
    for i, seg in enumerate(out):  # 排序后重编号
        seg["i"] = i
    return out


# --------------------------------------------------------------------------
# SRT 读写
# --------------------------------------------------------------------------


def write_srt(path: str | os.PathLike[str], segments: Sequence[dict[str, Any]]) -> Path:
    """按契约写标准 SRT 字幕文件（UTF-8，用 util.fmt_srt_ts 保证格式一致）。

    空文本段会被跳过；序号从 1 连续编号（重新编号，不沿用 segments 里的 i）。
    """
    lines: list[str] = []
    number = 1
    for seg in segments:
        text = str(seg.get("text") or "").strip()
        if not text:
            continue
        start = float(seg.get("start") or 0.0)
        end = float(seg.get("end") or 0.0)
        if end <= start:
            end = start + 0.5
        lines.append(str(number))
        lines.append(f"{util.fmt_srt_ts(start)} --> {util.fmt_srt_ts(end)}")
        lines.append(text)
        lines.append("")
        number += 1
    return util.write_text(path, "\n".join(lines))


_SRT_TIME_RE = re.compile(
    r"(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})\s*-->\s*(\d{1,2}):(\d{2}):(\d{2})[.,](\d{1,3})"
)


def parse_srt(text: str) -> list[dict[str, Any]]:
    """解析 SRT 字幕文本，返回 [{"start","end","text"}]，按 start 升序。

    容忍：CRLF / BOM / 多余空行 / 序号缺失 / 时间轴带样式标记（{\\an8} 之类）。
    遇到完全无法解析的块就跳过，不抛异常（用户给的字幕质量参差是常态）。
    """
    content = (text or "").replace("\r\n", "\n").replace("\r", "\n").lstrip("\ufeff")
    blocks = re.split(r"\n\s*\n", content)
    segments: list[dict[str, Any]] = []
    for block in blocks:
        rows = [r for r in block.split("\n") if r.strip()]
        if not rows:
            continue
        time_row = -1
        match: re.Match[str] | None = None
        for idx, row in enumerate(rows[:3]):
            match = _SRT_TIME_RE.search(row)
            if match:
                time_row = idx
                break
        if match is None or time_row < 0:
            continue
        start = (
            int(match.group(1)) * 3600
            + int(match.group(2)) * 60
            + int(match.group(3))
            + int(match.group(4).ljust(3, "0")) / 1000.0
        )
        end = (
            int(match.group(5)) * 3600
            + int(match.group(6)) * 60
            + int(match.group(7))
            + int(match.group(8).ljust(3, "0")) / 1000.0
        )
        body = " ".join(rows[time_row + 1 :]).strip()
        body = re.sub(r"<[^>]+>", "", body)          # <i> 等样式标签
        body = re.sub(r"\{\\[^}]*\}", "", body)      # {\an8} 等位置标记
        body = _clean_segment_text(body)
        if not body:
            continue
        segments.append({"start": start, "end": end, "text": body})
    segments.sort(key=lambda s: (s["start"], s["end"]))
    return segments


# --------------------------------------------------------------------------
# 主入口：转写
# --------------------------------------------------------------------------


def auto_initial_prompt(out_dir: str | os.PathLike[str], *, limit: int = 220) -> str | None:
    """从 slides.json 自动拼一个 initial_prompt（课程名 + 课件标题）。

    Whisper 的 initial_prompt 只有 ~224 token，塞整页正文会挤掉真正有用的词。
    这里只取「课件页标题」，去重后按出现顺序拼接。标题通常就是老师课上会说的
    核心术语（课件标题里的专有名词），比手工猜词准。
    """
    root = Path(out_dir)
    data = util.read_json(root / "slides.json", default=None)
    if not isinstance(data, dict):
        return None
    parts: list[str] = []
    for s in data.get("slides") or []:
        if s.get("kind") not in ("slide", "blank", "unknown"):
            continue
        title = re.sub(r"\s+", "", util.clean_line(str(s.get("title") or "")))
        if not (2 <= len(title) <= 30):
            continue
        if re.search(r"\d{1,2}:\d{2}", title):
            continue
        if title not in parts:
            parts.append(title)
    if not parts:
        return None
    return "，".join(parts[:16])[:limit]


def _drop_stretched_hallucinations(
    segments: list[dict[str, Any]],
    *,
    ratio: float = 8.0,
    min_span: float = 30.0,
) -> tuple[list[dict[str, Any]], int]:
    """剔除「一句话横跨一两分钟」的静音幻觉片段。

    背景：录屏开头/间隙常有长静音，whisper 会在这段静音上"编"出一句话，
    并把时间戳拉得很长。实测一段 60 分钟录屏：

        正常讲解片段时长  中位 3.0s / p90 7.0s / p99 21s
        幻觉片段          161s / 177s / 64s / 180s

    两者差一个量级，非常好分。判据：片段时长 > max(min_span, 中位数 × ratio)。
    """
    if not segments:
        return segments, 0
    spans = sorted(float(s.get("end") or 0) - float(s.get("start") or 0) for s in segments)
    med = spans[len(spans) // 2] if spans else 0.0
    limit = max(min_span, med * ratio)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for seg in segments:
        span = float(seg.get("end") or 0) - float(seg.get("start") or 0)
        if span > limit:
            dropped += 1
            util.vlog(
                f"  · 丢弃超长幻觉片段（{util.fmt_ts(float(seg.get('start') or 0))}, "
                f"{span:.0f}s）：{util.squash(str(seg.get('text') or ''))[:24]}"
            )
            continue
        kept.append(seg)
    return kept, dropped


def _drop_prompt_leakage(
    segments: list[dict[str, Any]],
    initial_prompt: str | None,
    *,
    head_seconds: float = 30.0,
) -> tuple[list[dict[str, Any]], int]:
    """剔除「whisper 把提示词当语音输出」的幻觉片段。

    背景：initial_prompt 是双刃剑。音频开头若是静音/环境噪声（录屏前调设备
    很常见），whisper 会把提示词当成听到的内容直接吐出来：

        initial_prompt = "本课程总学分为2学分,总学时为32课时，核心问题是什么…"
        segments[0]    = "本课程总学分为3学分,总学时为32课时,"   ← 凭空捏造

    这些片段会被当成老师讲课写进笔记，比不识别更糟。

    判据：片段的字词有很高比例来自提示词，**且**它出现在开头
    （whisper 的提示词幻觉几乎都发生在音频起始处）。

    返回 (清洗后的片段, 删除条数)。
    """
    if not initial_prompt or not segments:
        return segments, 0

    prompt_norm = util.squash(initial_prompt)
    if len(prompt_norm) < 8:
        return segments, 0
    prompt_chars = set(prompt_norm)

    kept: list[dict[str, Any]] = []
    dropped = 0
    for seg in segments:
        start = float(seg.get("start") or 0.0)
        text_norm = util.squash(str(seg.get("text") or ""))
        if len(text_norm) < 4:
            kept.append(seg)
            continue

        covered = sum(1 for ch in text_norm if ch in prompt_chars) / len(text_norm)
        in_prompt = text_norm in prompt_norm
        span = float(seg.get("end") or 0.0) - start

        # 判据 1：整段文字都在提示词里出现过 → 一定是幻觉
        # （真实讲课不可能每个字都恰好来自提示词）
        if in_prompt or covered >= 0.95:
            dropped += 1
            util.vlog(f"  · 丢弃提示词幻觉（{util.fmt_ts(start)}）：{text_norm[:24]}")
            continue

        # 判据 2：开头 + 覆盖率很高 + 时长明显异常（真人不会一口气说 2 分钟）
        # 注意不要用固定时间窗：幻觉段可能延续到第 3 分钟（实测 1~339 秒都有）。
        if covered >= 0.85 and (start <= head_seconds or span >= 45.0):
            dropped += 1
            util.vlog(f"  · 丢弃提示词幻觉（{util.fmt_ts(start)}, {span:.0f}s）：{text_norm[:24]}")
            continue

        kept.append(seg)
    return kept, dropped


def run_asr(
    out_dir: str | os.PathLike[str],
    video: str | os.PathLike[str],
    *,
    model: str = "medium",
    language: str = "zh",
    device: str = "auto",
    compute_type: str = "auto",
    force: bool = False,
    min_duration: float = MIN_SEGMENT_SECONDS,
    initial_prompt: str | None = None,
    dry_run: bool = False,
) -> dict:
    """转写视频音轨，写出 transcript.json + transcript.srt。

    参数：
        out_dir:       课程输出目录
        video:         视频/音频文件（ffmpeg 能解的都行）
        model:         whisper 模型规格（tiny/base/small/medium/large-v3-turbo 或 HF 仓库 id）
                       compute_type=int8 时 medium 约 1.5GB 内存；实测中文准确率
                       明显高于 small，62 分钟录屏约 28 分钟转完
        language:      语言代码，默认 zh；传 "auto" 让 whisper 自己判断
        device:        auto | cpu | cuda
        compute_type:  auto | int8 | int8_float16 | float16 | float32
        force:         True 才重做；False 且已有 transcript.json 时直接复用
        min_duration:  短于这个秒数的片段会被合并（见 _merge_segments）
        initial_prompt: 给模型的提示词，写上课程名/术语能显著改善专有名词识别；
                       不传则自动从 slides.json 的课件标题拼（CVN_ASR_PROMPT 可覆盖）
        dry_run:       只做检查并报告，不真正转写

    返回：写出的 transcript.json 内容（dict）。
    异常：util.CvnError（中文可读）。
    """
    root = Path(out_dir)
    util.ensure_dir(root)

    json_path = root / "transcript.json"
    srt_path = root / "transcript.srt"

    # ---- 增量：已有转写且不是 force 就直接复用 ----
    if not force and json_path.exists():
        existing = util.read_json(json_path, default=None)
        if isinstance(existing, dict) and isinstance(existing.get("segments"), list):
            util.banner("语音转写")
            util.log(
                f"  · 已有 transcript.json（{len(existing['segments'])} 段，"
                f"engine={existing.get('engine')}），跳过转写（--force 可重做）"
            )
            if not srt_path.exists():
                write_srt(srt_path, existing["segments"])
                util.log("  · 补写了缺失的 transcript.srt")
            return existing

    src = Path(video)
    if not src.exists():
        raise util.CvnError(f"找不到媒体文件：{src}")

    util.banner("语音转写")
    util.log(f"  · 媒体：{src.name}")
    env = _prepare_env()
    util.log(f"  · 模型：{model}（缓存：{env['HF_HOME']}，镜像：{env['HF_ENDPOINT']}）")

    # ---- 提示词：用户显式给了就用；否则从课件标题自动拼 ----
    if initial_prompt is None:
        initial_prompt = (
            os.environ.get("CVN_ASR_PROMPT", "").strip()
            or auto_initial_prompt(root)
        )
    if initial_prompt:
        util.log(f"  · 提示词：{initial_prompt[:100]}{'…' if len(initial_prompt) > 100 else ''}")

    if dry_run:
        util.log("  · --dry-run：不执行转写")
        return {"engine": "faster-whisper", "segments": [], "dry_run": True}

    # ---- 抽音轨（顺便校验有没有音轨）----
    wav, duration = extract_audio(src)
    util.log(f"  · 音轨已抽取（{util.human_duration(duration)}）")

    # ---- 加载模型 ----
    WhisperModel = _import_faster_whisper()
    lang = (language or "").strip() or "auto"
    device_arg = (device or "auto").strip()
    compute_arg = (compute_type or "auto").strip()

    model_obj = _load_model(
        WhisperModel, model, device=device_arg, compute_type=compute_arg
    )

    # ---- 转写 ----
    util.log("  · 开始转写（首次运行会先下载模型，可能需要几分钟）…")
    started = time.time()
    try:
        seg_iter, info = model_obj.transcribe(
            str(wav),
            language=None if lang in ("auto", "") else lang,
            task="transcribe",
            beam_size=5,
            vad_filter=True,           # 去掉长静音，避免 whisper 在静音处幻觉
            condition_on_previous_text=False,  # 长视频里能明显减少「复读」幻觉
            initial_prompt=initial_prompt,
            word_timestamps=False,
        )
        raw: list[dict[str, Any]] = []
        for seg in seg_iter:          # 这是个生成器，迭代过程中才真正开始算
            raw.append({"start": seg.start, "end": seg.end, "text": seg.text})
            if len(raw) % 50 == 0:
                util.log(f"    · 已转写 {len(raw)} 段（{util.fmt_ts(seg.end)}）")
    except Exception as exc:
        raise util.CvnError(_explain_transcribe_failure(exc, model)) from exc

    # 去掉「把提示词当语音」的幻觉片段（录屏开头静音时几乎必现）
    raw, dropped = _drop_prompt_leakage(raw, initial_prompt)
    if dropped:
        util.log(f"  · 已剔除 {dropped} 条提示词幻觉片段（音频开头静音导致）")
    raw, dropped2 = _drop_stretched_hallucinations(raw)
    if dropped2:
        util.log(f"  · 已剔除 {dropped2} 条超长幻觉片段（长静音处 whisper 会编话）")

    detected_lang = getattr(info, "language", None) or lang
    total_dur = float(getattr(info, "duration", 0.0) or duration or 0.0)
    util.log(
        f"  · 转写完成：{len(raw)} 个原始片段，"
        f"用时 {util.human_duration(time.time() - started)}（语言：{detected_lang}）"
    )

    # ---- 整理片段 ----
    merged = _merge_segments(raw, min_duration=min_duration)
    segments = _finalise_segments(merged, total_dur)

    payload = {
        "engine": "faster-whisper",
        "model": str(model),
        "language": str(detected_lang),
        "duration": round(total_dur, 3),
        "segments": segments,
        "initial_prompt": initial_prompt or "",
    }
    _write_transcripts(root, payload)
    # 转写完就把临时 wav 删掉（可能几百 MB）
    try:
        wav.unlink()
    except OSError:
        pass
    util.log(f"  · 已写入 {json_path} 与 {srt_path.name}（{len(segments)} 段）")
    _mark_manifest(root, payload)
    return payload


def _load_model(WhisperModel, model: str, *, device: str, compute_type: str):
    """加载 whisper 模型，处理「下载失败 / 设备不支持」两类常见错误。"""
    attempts: list[dict[str, str]] = []
    # compute_type=auto 时按设备给默认值：CPU 用 int8 最快，GPU 用 float16
    def default_compute(dev: str) -> str:
        if dev in ("cuda", "mps"):
            return "float16"
        return "int8"

    if device == "auto" or compute_type == "auto":
        for dev in ("cpu",):  # 本工具默认跑 CPU；用户显式给了 device 就用用户的
            attempts.append({"device": device if device != "auto" else dev,
                             "compute_type": compute_type if compute_type != "auto" else default_compute(dev)})
        if device != "auto":
            attempts = [{"device": device,
                         "compute_type": compute_type if compute_type != "auto" else default_compute(device)}]
    else:
        attempts = [{"device": device, "compute_type": compute_type}]

    last_exc: Exception | None = None
    for attempt in attempts:
        try:
            util.log(f"  · 加载模型 {model}（device={attempt['device']}, compute_type={attempt['compute_type']}）…")
            return WhisperModel(model, device=attempt["device"], compute_type=attempt["compute_type"])
        except Exception as exc:  # 下载失败 / 设备不可用 / 量化不支持
            last_exc = exc
            msg = str(exc)
            if re.search(r"(Connection|Timeout|Couldn't connect|Name or service|SSL|Max retries)", msg, re.I):
                raise util.CvnError(
                    f"下载 whisper 模型失败（{model}）。\n"
                    "国内网络通常连不上 HuggingFace 官方站，请设置镜像后重试：\n"
                    f"  export HF_ENDPOINT={DEFAULT_HF_ENDPOINT}\n"
                    f"（本次已尝试自动设置，实际值：{os.environ.get('HF_ENDPOINT')}）\n"
                    "也可以手动下载模型后把路径传给 --model：\n"
                    "  cvn asr <out_dir> --model /path/to/faster-whisper-small"
                ) from exc
            util.warn(f"  该配置不可用（{attempt['device']}/{attempt['compute_type']}）：{msg[:120]}")

    raise util.CvnError(
        f"加载 whisper 模型失败（{model}）：{last_exc}\n"
        "可能是设备/量化类型不支持，试试 --device cpu --compute-type int8。"
    ) from last_exc


def _explain_transcribe_failure(exc: Exception, model: str) -> str:
    """把转写期异常翻译成人话。"""
    msg = str(exc)
    if re.search(r"(Connection|Timeout|Max retries|SSL|Name or service)", msg, re.I):
        return (
            f"转写过程中网络中断（模型：{model}）。\n"
            f"请设置镜像后重试：export HF_ENDPOINT={DEFAULT_HF_ENDPOINT}\n"
            f"原始错误：{msg[:300]}"
        )
    if "ffmpeg" in msg.lower():
        return f"解码音频失败：{msg[:300]}\n提示：确认 ffmpeg 可用（brew install ffmpeg）。"
    return f"转写失败：{msg[:400]}\n如果是显存/内存不足，试试更小的模型（--model base）。"


def _write_transcripts(root: Path, payload: dict) -> None:
    """同时写 transcript.json 和 transcript.srt（契约要求两阶段都产出）。"""
    util.write_json(root / "transcript.json", payload)
    write_srt(root / "transcript.srt", payload.get("segments") or [])


def _mark_manifest(root: Path, payload: dict) -> None:
    """把 asr 阶段标记写进 manifest.json（存在才写；失败不影响主流程）。"""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = util.read_json(manifest_path, default=None)
        if not isinstance(manifest, dict):
            return
        stages = manifest.setdefault("stages", {})
        stages["asr"] = {
            "done": True,
            "at": util.stamp(),
            "params": {"engine": payload.get("engine"), "model": payload.get("model"),
                       "language": payload.get("language"), "segments": len(payload.get("segments") or [])},
        }
        util.write_json(manifest_path, manifest)
    except Exception as exc:
        util.vlog(f"  · 写 manifest.json 失败（忽略）：{exc}")


# --------------------------------------------------------------------------
# 主入口：导入现成字幕
# --------------------------------------------------------------------------


def import_srt(
    out_dir: str | os.PathLike[str],
    srt_path: str | os.PathLike[str],
    *,
    language: str = "zh",
    duration: float | None = None,
) -> dict:
    """导入现成的 SRT 字幕，转换成契约格式的 transcript.json + transcript.srt。

    用户手里常常已经有字幕（课程网站/剪映/人工），比重新跑 ASR 又准又快，
    所以这是 ASR 的合法替代路径，engine 记为 "srt-import"。

    参数：
        out_dir:  课程输出目录
        srt_path: SRT 文件路径（UTF-8 / UTF-8 BOM；GBK 会自动尝试）
        language: 字幕语言代码，只做记录
        duration: 覆盖时长（秒）；None 则取最后一段的 end

    返回：写出的 transcript.json 内容。
    异常：util.CvnError（文件不存在 / 解析不出任何字幕后段）。
    """
    root = Path(out_dir)
    util.ensure_dir(root)
    src = Path(srt_path)
    if not src.exists():
        raise util.CvnError(f"找不到字幕文件：{src}")
    if not src.is_file():
        raise util.CvnError(f"不是文件：{src}")

    util.banner("导入字幕")
    text = _read_text_with_fallback(src)
    raw = parse_srt(text)
    if not raw:
        raise util.CvnError(
            f"没能从 {src.name} 里解析出任何字幕段落。\n"
            "请确认它是标准 SRT 格式（每段：序号 / 00:00:01,000 --> 00:00:03,000 / 文本）。"
        )
    util.log(f"  · 解析出 {len(raw)} 段")

    total = float(duration) if duration else max(seg["end"] for seg in raw)
    merged = _merge_segments(raw)
    segments = _finalise_segments(merged, total)

    payload = {
        "engine": "srt-import",
        "model": "",
        "language": str(language or "zh"),
        "duration": round(total, 3),
        "segments": segments,
    }
    _write_transcripts(root, payload)
    util.log(f"  · 已写入 transcript.json 与 transcript.srt（{len(segments)} 段）")
    _mark_manifest(root, payload)
    return payload


def _read_text_with_fallback(path: Path) -> str:
    """读字幕文本：UTF-8 -> UTF-8-sig -> GB18030 -> big5，最后带替换读。"""
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "big5"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise util.CvnError(f"读取字幕文件失败：{path}：{exc}") from exc
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:  # pragma: no cover - 极端情况
        raise util.CvnError(f"读取字幕文件失败：{path}：{exc}") from exc


# --------------------------------------------------------------------------
# 命令行入口（方便单独调试；正式入口在 course_kit.py）
# --------------------------------------------------------------------------


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="cvn asr", description="语音转写 / 导入字幕")
    ap.add_argument("out_dir", help="课程输出目录")
    ap.add_argument("video", nargs="?", help="视频或音频文件（用 --srt 时可省略）")
    ap.add_argument("--srt", default=None, help="导入现成 SRT 字幕，而不是跑 ASR")
    ap.add_argument("--model", default="medium")
    ap.add_argument("--language", default="zh")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--compute-type", default="auto")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--probe", action="store_true", help="只打印环境探测结果")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    util.set_verbose(args.verbose)
    try:
        if args.probe:
            util.log(json.dumps(probe_asr(), ensure_ascii=False, indent=2))
            return 0
        if args.srt:
            payload = import_srt(args.out_dir, args.srt, language=args.language)
        else:
            if not args.video:
                ap.error("需要提供视频文件，或用 --srt 导入字幕")
            payload = run_asr(
                args.out_dir,
                args.video,
                model=args.model,
                language=args.language,
                device=args.device,
                compute_type=args.compute_type,
                force=args.force,
                dry_run=args.dry_run,
            )
        util.log(f"✔ 转写完成，engine={payload.get('engine')}，段落={len(payload.get('segments', []))}")
        return 0
    except util.CvnError as exc:
        util.log(f"✘ {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
