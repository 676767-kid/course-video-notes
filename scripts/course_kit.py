#!/usr/bin/env python3
"""course_kit.py —— 课程录像 → 可用的笔记 + 知识点清单

一条命令跑完：

    python course_kit.py all --video 第3讲.mp4 --out out/第3讲

也可以分步跑（适合调试 / 只重做某一环）：

    python course_kit.py probe
    python course_kit.py frames --video 第3讲.mp4 --out out/第3讲
    python course_kit.py ocr    --out out/第3讲
    python course_kit.py asr    --video 第3讲.mp4 --out out/第3讲
    python course_kit.py merge  --out out/第3讲

设计原则：
- 每一步都能单独重跑，默认复用已有结果（增量），--force 才重做
- 出错只打印中文原因 + 下一步建议，不甩 traceback（--debug 除外）
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

# 允许直接 `python course_kit.py` 运行，也允许 `python -m cvn.cli`
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from cvn import env as env_mod          # noqa: E402
from cvn import media                   # noqa: E402
from cvn.util import (                  # noqa: E402
    CvnError,
    banner,
    fmt_ts,
    human_duration,
    log,
    read_json,
    set_verbose,
    slugify,
    write_json,
    stamp,
)

PROG = "course_kit"

# 这几个模块按需惰性导入：`probe` 必须在任何依赖都没装、
# 甚至模块文件缺失时也能跑起来，否则用户第一眼看到的就是 traceback。


def _lazy(name: str):
    import importlib

    return importlib.import_module(f"cvn.{name}")


# --------------------------------------------------------------------------
# 参数
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog=PROG,
        description="把课程/讲座录像拆成「课件页 + 逐字稿 + 重点」，并生成知识点清单。",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "例子：\n"
            "  %(prog)s probe\n"
            "  %(prog)s all --video 第3讲.mp4 --out out/第3讲\n"
            "  %(prog)s all --video 录屏.mkv --out out/课 --interval 1 --pick last\n"
            "  %(prog)s all --video 课.mp4 --out out/课 --from-srt 已有字幕.srt\n"
            "  %(prog)s merge --out out/课 --min-score 5\n"
        ),
    )
    p.add_argument("-v", "--verbose", action="store_true", help="打印更详细的执行过程")
    p.add_argument("--debug", action="store_true", help="出错时打印完整 traceback")
    p.add_argument("--quiet", action="store_true", help="只输出结果路径")

    sub = p.add_subparsers(dest="cmd", required=True, metavar="命令")

    # ---- probe ----
    sp = sub.add_parser("probe", help="检查环境：能不能抽帧、识字、转写")
    sp.add_argument("--json", action="store_true", help="以 JSON 输出，便于脚本消费")
    sp.add_argument("--check-network", action="store_true", help="顺便测试模型下载站点连通性")

    # ---- setup ----
    sp = sub.add_parser("setup", help="一键装好语音转写（自动建独立 venv，不动系统环境）")
    sp.add_argument("--venv", help="venv 位置，默认 <本目录>/.cvn-venv")
    sp.add_argument("--model", default="", help="顺便预下载的 whisper 模型（如 small），不填则跳过")

    # ---- frames ----
    def add_video_args(sp):
        sp.add_argument("--video", required=True,
                        help="课程录像路径，或 http(s) 直链（会流式读取，不必先下载）")
        sp.add_argument("--out", help="输出目录，默认 out/<视频名>")

    sp = sub.add_parser("frames", help="抽帧 + 去重，得到每一页课件原图")
    add_video_args(sp)
    _add_frame_opts(sp)
    sp.add_argument("--force", action="store_true", help="忽略已有结果重做")

    # ---- ocr ----
    sp = sub.add_parser("ocr", help="对课件图做文字识别")
    sp.add_argument("--out", required=True, help="输出目录（与 frames 相同）")
    sp.add_argument("--engine", default="auto",
                    choices=["auto", "rapidocr", "hybrid", "vision", "pyobjc", "tesseract", "none"],
                    help="OCR 引擎，默认 auto（有 RapidOCR 时走 hybrid）")
    sp.add_argument("--vlm", action="store_true",
                    help="用本地视觉模型读标题/补图表说明（需 Ollama + qwen3-vl:2b-instruct）")
    sp.add_argument("--force", action="store_true", help="忽略已有结果重做")

    # ---- asr ----
    sp = sub.add_parser("asr", help="语音转写成带时间戳的逐字稿")
    sp.add_argument("--video", help="课程录像路径（用 --from-srt 时可不填）")
    sp.add_argument("--out", required=True, help="输出目录（与 frames 相同）")
    sp.add_argument("--from-srt", dest="from_srt", metavar="字幕.srt",
                    help="直接导入现成字幕，不做语音识别")
    sp.add_argument("--model", default="medium",
                    help="whisper 模型：tiny/base/small/medium/large-v3-turbo（默认 medium，"
                         "中文准确率明显优于 small）")
    sp.add_argument("--language", default="zh", help="语言代码，默认 zh")
    sp.add_argument("--device", default="auto", help="cpu / cuda / auto")
    sp.add_argument("--compute-type", dest="compute_type", default="auto",
                    help="int8 / float16 / auto")
    sp.add_argument("--initial-prompt", dest="initial_prompt", default=None,
                    help="给 whisper 的提示词（课程名/术语），能明显改善专有名词识别，"
                         "如 --initial-prompt '电机学，电枢反应，换向，等效电路'")
    sp.add_argument("--force", action="store_true", help="忽略已有结果重做")

    # ---- textbook / review ----
    sp = sub.add_parser("textbook", help="把多讲的课件合编成系统复习讲义（PDF）")
    sp.add_argument("--out", required=True, nargs="+", metavar="OUT",
                    help="一个或多个已跑完的课程目录，按讲课顺序给出")
    sp.add_argument("--content", required=True, metavar="JSON",
                    help="讲义内容 JSON（章节结构 + 正文），见 references/content-schema.md")
    sp.add_argument("--pdf", dest="pdf_path", help="输出 PDF 路径")
    sp.add_argument("--chrome", help="指定 Chrome/Chromium")

    # ---- pdf ----
    sp = sub.add_parser("pdf", help="把笔记导出成图文并茂的 PDF 讲义")
    sp.add_argument("--out", required=True, help="输出目录（与 frames 相同）")
    sp.add_argument("--pdf", dest="pdf_path", help="PDF 路径，默认 <out>/课程笔记.pdf")
    sp.add_argument("--title", help="封面标题，默认自动识别")
    sp.add_argument("--min-score", dest="min_score", type=int, default=3,
                    help="重点阈值，默认 3")
    sp.add_argument("--keep-html", dest="keep_html", action="store_true",
                    help="保留中间 HTML（便于自己改样式）")
    sp.add_argument("--chrome", help="指定 Chrome/Chromium 可执行文件")

    # ---- lms ----
    sp = sub.add_parser("lms", help="从在线教学平台抓课程录像直链（需先登录一次）")
    lsub = sp.add_subparsers(dest="lms_cmd", required=True, metavar="动作")
    for _name, _help in (
        ("login", "弹浏览器登录一次，保存会话"),
        ("list", "列出课程里的所有课时"),
        ("urls", "抓每节课的录像直链（默认只要屏幕画面）"),
    ):
        lp = lsub.add_parser(_name, help=_help)
        lp.add_argument("--site", required=True,
                        help="站点适配器名（在 scripts/cvn/lms.py 的 ADAPTERS 里注册）")
        lp.add_argument("--course", required=True,
                        help="课程 id（课程页 URL 里 /course/<id>/ 那一段）")
        if _name == "urls":
            lp.add_argument("--only", nargs="*", help="只处理标题含这些关键字的课")
            lp.add_argument("-o", "--out", help="保存为 JSON")

    # ---- merge ----
    sp = sub.add_parser("merge", help="对齐 + 划重点 + 生成 course.md 与知识点清单")
    sp.add_argument("--out", required=True, help="输出目录（与 frames 相同）")
    sp.add_argument("--min-score", dest="min_score", type=int, default=3,
                    help="划重点阈值：默认 3；想少而准调到 5，想多调到 2")
    sp.add_argument("--title", help="笔记标题，默认取视频文件名")
    sp.add_argument("--inject-by-p", dest="inject_by_p", metavar="目录",
                    help="把课件内容注入 dsh-video-to-knowledge 的 by_p 素材目录")
    sp.add_argument("--force", action="store_true", help="忽略已有结果重做")

    # ---- all ----
    sp = sub.add_parser("all", help="一条命令跑完全流程")
    add_video_args(sp)
    _add_frame_opts(sp)
    sp.add_argument("--engine", default="auto",
                    choices=["auto", "rapidocr", "hybrid", "vision", "pyobjc", "tesseract", "none"],
                    help="OCR 引擎，默认 auto（有 RapidOCR 时走 hybrid）")
    sp.add_argument("--vlm", action="store_true", help="用本地视觉模型读标题/补图表说明（需 Ollama）")
    sp.add_argument("--from-srt", dest="from_srt", metavar="字幕.srt",
                    help="跳过语音识别，直接导入现成字幕")
    sp.add_argument("--model", default="medium", help="whisper 模型（默认 medium）")
    sp.add_argument("--language", default="zh", help="语言代码，默认 zh")
    sp.add_argument("--device", default="auto")
    sp.add_argument("--compute-type", dest="compute_type", default="auto")
    sp.add_argument("--initial-prompt", dest="initial_prompt", default=None,
                    help="给 whisper 的提示词（课程名/术语），改善专有名词识别")
    sp.add_argument("--min-score", dest="min_score", type=int, default=3)
    sp.add_argument("--title", help="笔记标题")
    sp.add_argument("--inject-by-p", dest="inject_by_p", metavar="目录")
    sp.add_argument("--force", action="store_true", help="忽略已有结果，全部重做")
    sp.add_argument("--skip-asr", dest="skip_asr", action="store_true",
                    help="只做课件部分，不做语音转写")

    return p


def _add_frame_opts(sp: argparse.ArgumentParser) -> None:
    g = sp.add_argument_group("抽帧调参（一般不用改）")
    g.add_argument("--interval", type=float, default=media.DEFAULT_INTERVAL,
                   help="采样间隔秒数：翻页快用 1，两小时大课用 3（默认 2）")
    g.add_argument("--strategy", default="interval", choices=["interval", "scene", "both"],
                   help="抽帧策略：interval 定时采样（默认）/ scene 按场景切换 / both 两者结合")
    g.add_argument("--pick", default="first", choices=["first", "last", "mid"],
                   help="每页取哪一刻：first 首次出现（默认）/ last 讲完时（PPT 有逐条动画时用它）")
    g.add_argument("--change-ratio", dest="change_ratio", type=float, default=None,
                   help="翻页判定：变化面积占比阈值。默认不填 = 按素材自适应"
                        "（纯 PPT 视频和真人录屏的翻页幅度差一个量级，自适应更稳）；"
                        "填数值则强制使用该阈值")
    g.add_argument("--top-ratio", dest="top_ratio", type=float,
                   default=media.TOP_CHANGE_RATIO,
                   help="翻页判定：顶部标题带变化阈值（默认 %.2f）" % media.TOP_CHANGE_RATIO)
    g.add_argument("--min-hold", dest="min_hold", type=float, default=media.DEFAULT_MIN_HOLD,
                   help="一页最少停留秒数，更短的当转场残影并入上一页（默认 1.5）")
    g.add_argument("--min-page", dest="min_page", type=float, default=media.DEFAULT_MIN_PAGE,
                   help="停留不足这么多秒的页并入相邻页（默认 12，翻过去又翻回来时用）")
    g.add_argument("--crop", help="只识别画面某块区域：x,y,w,h（像素或 0~1 比例）")
    g.add_argument("--mask-top", dest="mask_top", type=float, default=0.0,
                   help="遮掉顶部比例，如 0.08 去掉录制软件标题栏")
    g.add_argument("--mask-bottom", dest="mask_bottom", type=float, default=0.0,
                   help="遮掉底部比例，如 0.12 去掉烧进画面的字幕条")
    g.add_argument("--max-width", dest="max_width", type=int, default=0,
                   help="课件图最长边上限（像素），0 = 原尺寸")


# --------------------------------------------------------------------------
# 输出目录
# --------------------------------------------------------------------------


def resolve_out_dir(video: str | None, out: str | None) -> Path:
    if out:
        return Path(out).expanduser().resolve()
    if not video:
        raise CvnError("需要 --out 指定输出目录")
    if media.is_url(video):
        # URL 里取文件名（去掉 query，那是签名参数）
        raw = str(video).split("?")[0].rstrip("/").split("/")[-1] or "remote"
        name = slugify(raw)
    else:
        name = slugify(Path(video).name)
    return (Path.cwd() / "out" / name).resolve()


def _run_meta(out: Path, video: str | None, args: argparse.Namespace) -> None:
    meta_path = out / "meta.json"
    meta = read_json(meta_path, default={}) or {}
    meta.setdefault("video", video)
    meta.setdefault("created", stamp())
    meta["updated"] = stamp()
    meta["argv"] = sys.argv
    meta["tool"] = "course-video-notes"
    meta["version"] = 1
    skip = {"video", "out", "force", "verbose", "debug", "quiet", "cmd", "title"}
    meta["params"] = {k: v for k, v in vars(args).items() if k not in skip}
    write_json(meta_path, meta)


def _stage_done(out: Path, stage: str, params: dict | None = None) -> None:
    mpath = out / "manifest.json"
    man = read_json(mpath, default={}) or {}
    man.setdefault("version", 1)
    man.setdefault("stages", {})
    meta = read_json(out / "meta.json", default={}) or {}
    man["video"] = meta.get("video") or man.get("video")
    man["stages"][stage] = {"done": True, "at": stamp(), "params": params or {}}
    write_json(mpath, man)


def _frames_path(out: Path) -> Path:
    return out / "frames.json"


def _no_frames_error(out: Path) -> CvnError:
    return CvnError(
        f"{out} 里还没有抽帧结果（frames.json）。\n"
        f"请先跑：python course_kit.py frames --video <视频> --out {out}"
    )


# --------------------------------------------------------------------------
# 各命令
# --------------------------------------------------------------------------


def cmd_probe(args: argparse.Namespace) -> int:
    report = env_mod.collect_report(check_network=args.check_network)
    venv_info = _probe_venv()
    if venv_info:
        report["venv"] = venv_info
        if venv_info.get("faster_whisper"):
            report["asr"]["ok"] = True
            report["asr"]["venv"] = venv_info["path"]
            report["hints"] = [h for h in report["hints"] if "faster-whisper" not in h]
            # ready 是 collect_report 一次性算好的，这里手动同步
            report["ready"]["asr"] = True
            report["ready"]["full"] = (
                report["ready"]["frames"] and report["ready"]["ocr"]
            )
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        print(env_mod.format_report(report))
        if venv_info and venv_info.get("faster_whisper"):
            print(f"    独立环境     ✔ {venv_info['path']}")
    return 0


def _default_venv_dir() -> Path:
    """venv 默认放用户缓存目录（~/.cache/course-video-notes/venv）。

    不放 skill 目录里：skill 可能随工程同步（如 iCloud），几百 MB 的 venv
    没必要到处跟着走；~/.cache 不可写时退回 skill 目录旁边。
    """
    candidates = [
        Path.home() / ".cache" / "course-video-notes" / "venv",
        Path(os.environ.get("TMPDIR") or "/tmp") / "course-video-notes" / "venv",
        _HERE / ".cvn-venv",
        Path("/tmp/course-video-notes/venv"),
    ]
    # 先看哪里**已经装了** venv——已存在的优先复用。
    # 否则同一台机器上 TMPDIR 变化（每次重启都会变）就会"找不到"装好的环境，
    # 白白重装几百 MB。
    for cand in candidates:
        if _venv_python(cand) and _venv_has_faster_whisper(cand):
            return cand
    # 没有现成的，就挑第一个可写位置新建
    for cand in candidates:
        parent = cand.parent
        try:
            parent.mkdir(parents=True, exist_ok=True)
            probe = parent / ".probe"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return cand
        except OSError:
            continue
    return _HERE / ".cvn-venv"


def _importable(module: str) -> bool:
    try:
        __import__(module)
        return True
    except Exception:  # noqa: BLE001 - 探测用途
        return False


def _venv_python(venv: Path) -> Path | None:
    """macOS/Linux venv 的 python；不存在返回 None。"""
    for cand in (venv / "bin" / "python3", venv / "bin" / "python"):
        if cand.exists():
            return cand
    return None


def _venv_has_faster_whisper(venv: Path) -> bool:
    py = _venv_python(venv)
    if not py:
        return False
    try:
        proc = subprocess.run(
            [str(py), "-c", "import faster_whisper"],
            capture_output=True, timeout=60,
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def _probe_venv() -> dict | None:
    """探测 skill 自带的 venv（setup 创建的）。

    会遍历所有候选位置，避免 TMPDIR 变化导致找不到已装好的环境。
    """
    venv = _default_venv_dir()
    if not venv.exists():
        return None
    return {
        "path": str(venv),
        "python": str(_venv_python(venv) or ""),
        "faster_whisper": _venv_has_faster_whisper(venv),
    }


def _asr_env(venv: Path) -> dict:
    """跑 venv 里的转写时需要的最小环境：HF 缓存指到可写目录。"""
    env = dict(os.environ)
    env.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    hf_home = venv / ".hf-home"
    hf_home.mkdir(parents=True, exist_ok=True)
    env["HF_HOME"] = str(hf_home)
    env["HUGGINGFACE_HUB_CACHE"] = str(hf_home / "hub")
    return env


def cmd_setup(args: argparse.Namespace) -> int:
    """一键装好转写依赖。全程在独立 venv 里，绝不碰系统 Python。"""
    import shutil
    import venv as venv_mod

    venv_dir = Path(args.venv).expanduser().resolve() if args.venv else _default_venv_dir()
    py = _venv_python(venv_dir)

    if py and _venv_has_faster_whisper(venv_dir):
        print(f"✔ 已就绪：{venv_dir}")
        print("  无需重复安装。")
    else:
        if not py:
            print(f"① 创建独立环境 {venv_dir} …")
            created = False
            uv = shutil.which("uv")
            if uv:
                # faster-whisper 依赖的 ctranslate2 只有 cp38~cp312 的轮子，
                # 系统默认 Python 可能太新（如 3.15），所以这里显式锁 3.12。
                # uv 会在需要时自动下载一个受管的 CPython，不碰系统环境。
                env = dict(os.environ)
                cache = venv_dir.parent / (venv_dir.name + "-cache")
                # 注意顺序：先删残留再建缓存，否则 mkdir 会把 venv_dir 复活，
                # uv 会报 "A directory already exists"。
                if venv_dir.exists():
                    shutil.rmtree(venv_dir, ignore_errors=True)
                cache.mkdir(parents=True, exist_ok=True)
                env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
                env["UV_PYTHON_INSTALL_DIR"] = str(cache / "uvpy")
                proc = subprocess.run(
                    [uv, "venv", "--python", "3.12", str(venv_dir)],
                    env=env, capture_output=True, text=True,
                )
                if proc.returncode != 0 and venv_dir.exists():
                    # 残留目录 uv 坚决不覆盖时，显式 --force 兜底
                    proc = subprocess.run(
                        [uv, "venv", "--force", "--python", "3.12", str(venv_dir)],
                        env=env, capture_output=True, text=True,
                    )
                created = proc.returncode == 0
                if not created:
                    tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-3:]
                    print("  uv 建 3.12 环境失败：" + " | ".join(tail))
                    print("  改用当前解释器…")
            if not created:
                import sys as _sys

                if _sys.version_info[:2] > (3, 12):
                    raise CvnError(
                        f"当前 Python {_sys.version.split()[0]} 太新，faster-whisper 依赖的 "
                        "ctranslate2 最高只支持 3.12。\n"
                        "  装 uv 后重试（会自动下载 3.12）：brew install uv\n"
                        "  或自备 3.11/3.12 后：python3.12 course_kit.py setup"
                    )
                venv_mod.create(str(venv_dir), with_pip=True)
            py = _venv_python(venv_dir)
            if not py:
                raise CvnError(f"venv 创建失败：{venv_dir}")
        print("② 安装 faster-whisper（首次要下载，约 1~2 分钟）…")
        uv = shutil.which("uv")
        pip_args: list[str]
        env = dict(os.environ)
        # 缓存目录必须可写：默认的 ~/.cache 在部分环境（如沙箱）被禁
        cache = venv_dir.parent / (venv_dir.name + "-cache")
        cache.mkdir(parents=True, exist_ok=True)
        if uv:
            pip_args = [uv, "pip", "install", "--python", str(py),
                        "faster-whisper", "av<19"]   # PyAV 19 删了 metadata_errors，faster-whisper 1.2.x 会崩
            env.setdefault("UV_CACHE_DIR", str(cache / "uv"))
            env["UV_PYTHON_INSTALL_DIR"] = str(cache / "uvpy")
        else:
            pip_args = [str(py), "-m", "pip", "install", "--upgrade", "pip",
                        "faster-whisper", "av<19"]
            env.setdefault("PIP_CACHE_DIR", str(cache / "pip"))
        proc = subprocess.run(pip_args, env=env)
        if proc.returncode != 0:
            raise CvnError("安装失败。可手动执行：" + " ".join(pip_args))
        if not _venv_has_faster_whisper(venv_dir):
            raise CvnError("装完了但导入失败，可能是 ctranslate2 与 Python 版本不匹配。\n"
                           "建议换 Python 3.11/3.12 重建 venv 再装。")
        print(f"✔ 转写依赖已装到独立环境：{venv_dir}")

    # 预下载模型（可选）
    if args.model:
        print(f"③ 预下载 whisper 模型 {args.model} …")
        code = (
            "from faster_whisper import WhisperModel\n"
            f"WhisperModel({args.model!r}, device='cpu', compute_type='int8')\n"
            "print('model ready')"
        )
        proc = subprocess.run([str(_venv_python(venv_dir)), "-c", code], env=_asr_env(venv_dir))
        if proc.returncode != 0:
            print("  模型预下载失败（不影响已装好的依赖）。首次转写时会重试，"
                  "如连不上官方站点请先：export HF_ENDPOINT=https://hf-mirror.com")

    print("")
    print("完成。之后正常使用即可：")
    print("  python3 course_kit.py all --video <视频> --out out/<课程名>")
    print("（course_kit 会自动发现这个 venv，无需额外参数）")
    return 0


def cmd_frames(args: argparse.Namespace) -> int:
    out = resolve_out_dir(args.video, args.out)
    out.mkdir(parents=True, exist_ok=True)
    _run_meta(out, args.video, args)
    result = media.extract_slides(
        out, args.video,
        interval=args.interval,
        strategy=args.strategy,
        pick=args.pick,
        change_ratio=args.change_ratio,
        top_ratio=args.top_ratio,
        min_hold=args.min_hold,
        min_page=args.min_page,
        crop=args.crop,
        mask_top=args.mask_top,
        mask_bottom=args.mask_bottom,
        max_width=args.max_width,
        force=args.force,
    )
    _stage_done(out, "frames", {
        "interval": args.interval, "strategy": args.strategy, "pick": args.pick,
    })
    if args.quiet:
        print(out)
    return 0


def cmd_ocr(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    if not _frames_path(out).exists():
        raise _no_frames_error(out)
    _run_meta(out, None, args)
    ocr_mod = _lazy("ocr")
    result = ocr_mod.run_ocr(
        out, engine=args.engine, vlm=args.vlm, force=args.force
    )
    _stage_done(out, "ocr", {"engine": args.engine, "vlm": args.vlm})
    if args.quiet:
        print(out / "slides.json")
    return 0


def cmd_asr(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    _run_meta(out, args.video, args)
    asr_mod = _lazy("asr")
    if args.from_srt:
        result = asr_mod.import_srt(out, args.from_srt)
        _stage_done(out, "asr", {"via": "srt"})
    else:
        if not args.video:
            raise CvnError("需要 --video，或用 --from-srt 导入现成字幕")
        result = asr_mod.run_asr(
            out, args.video, model=args.model, language=args.language,
            device=args.device, compute_type=args.compute_type, force=args.force,
            initial_prompt=args.initial_prompt,
        )
        _stage_done(out, "asr", {"model": args.model, "language": args.language})
    if args.quiet:
        print(out / "transcript.json")
    return 0


def cmd_textbook(args: argparse.Namespace) -> int:
    """把内容 JSON + 课件图渲染成一本讲义 PDF。"""
    review_mod = _lazy("review")
    pdf_mod = _lazy("pdf")

    content_path = Path(args.content).expanduser().resolve()
    if not content_path.exists():
        raise CvnError(f"找不到讲义内容文件：{content_path}")
    content = read_json(content_path, default=None)
    if not isinstance(content, dict) or not content.get("chapters"):
        raise CvnError(
            f"{content_path} 不是合法的讲义内容文件（需要 chapters 字段）。\n"
            "格式见 references/content-schema.md"
        )

    out_dirs = [Path(p).expanduser().resolve() for p in (args.out or [])]
    for d in out_dirs:
        if not d.exists():
            raise CvnError(f"课程目录不存在：{d}")

    banner("收集课件图")
    figures = []
    for d in out_dirs:
        slides_dir = d / "slides"
        if not slides_dir.exists():
            continue
        for png in sorted(slides_dir.glob("slide_*.png")):
            figures.append({"label": d.name, "title": png.name, "path": png})
    log(f"  · 共 {len(figures)} 张课件图可供引用")

    banner("渲染 HTML")
    html_text = review_mod.build_html(content, figures=figures)
    html_path = (out_dirs[0] if out_dirs else content_path.parent) / "复习讲义.html"
    review_mod.write_html(html_text, html_path)
    log(f"  · HTML：{html_path}")

    banner("打印 PDF")
    pdf_path = Path(args.pdf_path).expanduser() if args.pdf_path else html_path.with_suffix(".pdf")
    try:
        pdf_mod.html_to_pdf(html_path, pdf_path, chrome=args.chrome)
    except RuntimeError as exc:
        raise CvnError(
            f"{exc}\n提示：HTML 已生成，可用浏览器打开后「打印 → 存为 PDF」。"
        ) from exc

    size_kb = pdf_path.stat().st_size / 1024
    print("")
    print(f"完成：{pdf_path}（{size_kb:.0f} KB）")
    print(f"  章节 {len(content.get('chapters') or [])} 章")
    if args.quiet:
        print(pdf_path)
    return 0


def cmd_pdf(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    if not _frames_path(out).exists():
        raise _no_frames_error(out)

    notes = _lazy("notes")
    pdf_mod = _lazy("pdf")
    util_mod = _lazy("util")

    banner("准备 PDF 数据")
    data = notes.export_for_pdf(out, min_score=args.min_score)
    title = args.title or data.get("title") or out.name

    meta = read_json(out / "meta.json", default={}) or {}
    source_name = Path(str(meta.get("video") or "")).name

    banner("渲染 HTML")
    html_text = pdf_mod.build_html(
        title=title,
        pages=data["pages"],
        knowledge=data["knowledge"],
        emphasis=data["emphasis"],
        out_dir=out,
        source_name=source_name,
        duration=util_mod.human_duration(data.get("duration") or 0.0),
    )
    html_path = out / "course.html"
    pdf_mod.write_html(html_text, html_path)
    log(f"  · HTML：{html_path}")

    banner("打印 PDF")
    pdf_path = Path(args.pdf_path).expanduser() if args.pdf_path else (out / "课程笔记.pdf")
    try:
        pdf_mod.html_to_pdf(html_path, pdf_path, chrome=args.chrome)
    except RuntimeError as exc:
        raise CvnError(
            f"{exc}\n"
            "提示：HTML 已经生成好了，可以直接用浏览器打开再「打印 → 存为 PDF」。"
        ) from exc

    if not args.keep_html:
        pass  # 保留 HTML 无害，而且用户常想改样式
    _stage_done(out, "pdf", {"pages": len(data["pages"])})

    size_kb = pdf_path.stat().st_size / 1024 if pdf_path.exists() else 0
    print("")
    print(f"完成：{pdf_path}（{size_kb:.0f} KB）")
    print(f"  课件 {len(data['pages'])} 页 · 知识点 {len(data['knowledge'])} 条 · 重点 {len(data['emphasis'])} 条")
    if args.quiet:
        print(pdf_path)
    return 0


def cmd_lms(args: argparse.Namespace) -> int:
    """LMS 相关：登录 / 列课 / 抓直链。"""
    lms = _lazy("lms")
    act = getattr(args, "lms_cmd", None)
    site = args.site
    try:
        if act == "login":
            path = lms.login(site, args.course)
            print(f"会话已保存：{path}")
        elif act == "list":
            for x in lms.list_lessons(site, args.course):
                print(f"{x['activity_id']}\t{x['title']}")
        elif act == "urls":
            rows = lms.harvest_urls(site, args.course, only=args.only)
            ok = sum(1 for r in rows if r.get("pc_url"))
            print(f"\n有电脑端画面：{ok}/{len(rows)}")
            if args.out:
                write_json(Path(args.out).expanduser(), rows)
                print(f"已保存：{args.out}")
            else:
                for r in rows:
                    print(f"{r['title']}\n  {r.get('pc_url') or '(无)'}")
        else:
            raise CvnError(f"未知动作：{act}")
    except RuntimeError as exc:
        raise CvnError(str(exc)) from exc
    return 0


def cmd_merge(args: argparse.Namespace) -> int:
    out = Path(args.out).expanduser().resolve()
    if not _frames_path(out).exists():
        raise _no_frames_error(out)

    notes = _lazy("notes")
    banner("对齐 + 划重点")
    aligned = notes.align(out, min_score=args.min_score)
    _stage_done(out, "align", {"min_score": args.min_score})

    banner("生成笔记")
    result = notes.render(out, min_score=args.min_score, title=args.title)
    _stage_done(out, "merge", {"min_score": args.min_score})

    if args.inject_by_p:
        _inject_by_p(out, args.inject_by_p)

    if args.quiet:
        print(out / "course.md")
    else:
        _print_summary(out, result)
    return 0


def _inject_by_p(out: Path, by_p: str) -> None:
    """把课件内容注入 dsh-video-to-knowledge 的 by_p 素材。"""
    target = Path(by_p).expanduser().resolve()
    if not target.parent.exists() and not target.exists():
        raise CvnError(
            f"by_p 目录不存在：{target}\n"
            "顺序应为：先让插件跑出 runs/<videoId>/by_p，再执行本步骤。"
        )
    if "by_p" not in target.name:
        raise CvnError(
            f"拒绝注入：{target} 看起来不是 by_p 目录。\n"
            "请指向 <workDir>/runs/<videoId>/by_p，避免污染其它素材。"
        )
    data = read_json(out / "aligned.json", default={}) or {}
    injected = 0
    for slide in data.get("slides", []):
        p = target / f"p{slide['index']}.txt"
        if not p.exists():
            continue  # 只增强插件已产出的段落，不凭空造文件
        original = p.read_text(encoding="utf-8", errors="replace")
        marker = "【课件原文】"
        if marker in original:
            continue
        extra = []
        if slide.get("lines"):
            extra.append(marker)
            extra.extend(slide["lines"])
        if slide.get("figure_note"):
            extra.append("【图示】" + slide["figure_note"])
        if not extra:
            continue
        p.write_text(original.rstrip() + "\n\n" + "\n".join(extra) + "\n", encoding="utf-8")
        injected += 1
    if injected:
        log(f"已向 {target} 注入 {injected} 段课件内容")
    else:
        log(f"没有需要注入的段落（{target} 可能已是增强过的）")


def _print_summary(out: Path, result: dict) -> None:
    course = out / "course.md"
    know = out / "knowledge.md"
    print("")
    print("完成。产物：")
    for label, path in (
        ("课程笔记", course),
        ("知识点清单", know),
        ("知识点表格", out / "knowledge.csv"),
        ("划重点数据", out / "emphasis.json"),
        ("逐字稿字幕", out / "transcript.srt"),
        ("课件原图", out / "slides"),
    ):
        mark = "✔" if path.exists() else "·"
        print(f"  {mark} {label:<10} {path}")
    n_emph = len(result.get("emphasis", {}).get("items", [])) if isinstance(result, dict) else 0
    n_know = result.get("knowledge_count", 0) if isinstance(result, dict) else 0
    if n_emph or n_know:
        print("")
        print(f"  共 {n_know} 个知识点、{n_emph} 条重点")


def cmd_all(args: argparse.Namespace) -> int:
    # URL 不能走 Path().resolve()，否则 https:// 会被规范化成 https:/
    video = args.video.strip() if media.is_url(args.video) else str(
        Path(args.video).expanduser().resolve()
    )
    out = resolve_out_dir(video, args.out)
    out.mkdir(parents=True, exist_ok=True)
    _run_meta(out, video, args)

    # 0) 先体检，缺关键能力就早点说清楚
    report = env_mod.collect_report(check_network=False)
    if not report["ffmpeg"]["ok"]:
        raise CvnError(
            "缺少 ffmpeg/ffprobe，无法抽帧。\n"
            "  macOS：brew install ffmpeg\n"
            "  Ubuntu：sudo apt install ffmpeg"
        )
    if not report["ocr"]["any_ok"]:
        log("!! 没有可用 OCR：课件上的文字提取不到，笔记里只剩老师讲解。")
        log("   macOS 可执行 xcode-select --install 启用 Apple Vision。")
    if not report["asr"]["ok"] and not args.from_srt and not args.skip_asr:
        raise CvnError(
            "没有安装 faster-whisper，无法语音转写。三选一：\n"
            "  1) pip install faster-whisper\n"
            "  2) 用已有字幕：--from-srt 字幕.srt\n"
            "  3) 只要课件部分：--skip-asr"
        )

    # 1) 抽帧
    banner("① 抽帧 / 去重")
    media.extract_slides(
        out, video,
        interval=args.interval, strategy=args.strategy, pick=args.pick,
        change_ratio=args.change_ratio, top_ratio=args.top_ratio,
        min_hold=args.min_hold, min_page=args.min_page, crop=args.crop,
        mask_top=args.mask_top, mask_bottom=args.mask_bottom,
        max_width=args.max_width, force=args.force,
    )
    _stage_done(out, "frames", {"interval": args.interval, "pick": args.pick})

    # 2) OCR
    banner("② 课件识字")
    if report["ocr"]["any_ok"]:
        _lazy("ocr").run_ocr(out, engine=args.engine, vlm=args.vlm, force=args.force)
        _stage_done(out, "ocr", {"engine": args.engine, "vlm": args.vlm})
    else:
        write_json(out / "slides.json", {"engine": "none", "slides": []})
        _stage_done(out, "ocr", {"engine": "none"})

    # 3) ASR
    banner("③ 语音转写")
    if args.skip_asr:
        log("已跳过（--skip-asr），笔记里将没有老师讲解。")
        write_json(out / "transcript.json", {
            "engine": "skipped", "model": "", "language": args.language,
            "duration": 0, "segments": [],
        })
        _stage_done(out, "asr", {"via": "skipped"})
    elif args.from_srt:
        _lazy("asr").import_srt(out, args.from_srt)
        _stage_done(out, "asr", {"via": "srt"})
    else:
        _lazy("asr").run_asr(
            out, video, model=args.model, language=args.language,
            device=args.device, compute_type=args.compute_type, force=args.force,
            initial_prompt=args.initial_prompt,
        )
        _stage_done(out, "asr", {"model": args.model, "language": args.language})

    # 4) 对齐 + 划重点
    notes = _lazy("notes")
    banner("④ 对齐 + 划重点")
    notes.align(out, min_score=args.min_score)
    _stage_done(out, "align", {"min_score": args.min_score})

    # 5) 生成笔记
    banner("⑤ 生成笔记与知识点清单")
    result = notes.render(out, min_score=args.min_score, title=args.title)
    _stage_done(out, "merge", {"min_score": args.min_score})

    if args.inject_by_p:
        _inject_by_p(out, args.inject_by_p)

    if args.quiet:
        print(out / "course.md")
    else:
        _print_summary(out, result)
    return 0


# --------------------------------------------------------------------------
# 入口
# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    set_verbose(args.verbose)

    handlers = {
        "probe": cmd_probe,
        "setup": cmd_setup,
        "frames": cmd_frames,
        "ocr": cmd_ocr,
        "asr": cmd_asr,
        "merge": cmd_merge,
        "pdf": cmd_pdf,
        "textbook": cmd_textbook,
        "lms": cmd_lms,
        "all": cmd_all,
    }

    # setup 必须在当前解释器里跑（它就是来装环境的）
    if args.cmd == "setup":
        try:
            return cmd_setup(args)
        except CvnError as exc:
            print(f"\n✘ {exc}", file=sys.stderr)
            return 1

    # 本解释器缺 faster-whisper、但 skill 的 venv 里有 → 整条命令转交给 venv 重跑
    # （asr 之外的其他阶段不依赖它，但放在一起重跑能保证各阶段用的是同一套环境）
    if args.cmd in ("asr", "all") and not _importable("faster_whisper"):
        venv = _default_venv_dir()
        py = _venv_python(venv) if venv.exists() else None
        if py and _venv_has_faster_whisper(venv):
            if not args.quiet:
                print(f"（转写依赖在独立环境里，转交：{py}）", file=sys.stderr)
            argv_rest = list(sys.argv[1:])
            env = _asr_env(venv)
            try:
                proc = subprocess.run([str(py), str(_HERE / "course_kit.py"), *argv_rest], env=env)
                return proc.returncode
            except KeyboardInterrupt:
                return 130

    try:
        return handlers[args.cmd](args)
    except CvnError as exc:
        print(f"\n✘ {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:  # noqa: BLE001
        if getattr(args, "debug", False):
            traceback.print_exc()
        print(f"\n✘ 未预期的错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        print("  加 --debug 可看完整堆栈；加 -v 可看执行细节。", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
