---
name: course-video-notes
description: 把课程录像变成系统复习讲义。支持本地视频、网络直链、在线教学平台（LMS）课程回放（自动抓取屏幕画面，站点适配器可自行添加）。产出三种成品：逐页笔记（course.md）、知识点清单（knowledge.csv）、图文并茂的复习讲义 PDF（含考点、标准表述、经典引文、插图）。当用户丢来视频/链接/课程地址，想整理成笔记、提取 PPT 内容、找出考点、或要一份能直接背的复习资料时使用。触发词：视频笔记、课程录像、录播、在线课程、把这段视频整理成笔记、提取课件、找出重点考点、做个复习讲义。
metadata:
  version: "2.0.0"
---

# 课程录像 → 复习讲义

## 这个 skill 解决什么问题

用户手里有一段课程录像（本地文件、网络直链、或学校的在线回放），想要一份**能直接用于复习的材料**。

难点从来不是「转字幕」，而是三件事：

1. **画面里混着非课件内容**——录屏开头是桌面、中间切到浏览器、结尾是 PPT 编辑器。这些不是课件，却和课件一样是满屏画面，不看文字分不出来。
2. **PPT 的层级信息在像素里**——标题、要点、引文、出处，OCR 出来只是一堆视觉行，需要按版面重建才能还原成文章。
3. **「讲了什么」不等于「考什么」**——用户要的是考点、标准表述、易混辨析，不是课堂流水账。

所以本 skill 的流程是：

```
取素材 → 抽帧去重 → 课件识字 → （可选）语音转写 → 版面重建 → 编排成讲义
```

## 三种输入来源

### 1. 本地视频

```bash
python scripts/course_kit.py all --video 第3讲.mp4 --out out/第3讲
```

支持 mp4/mkv/mov/ts 等 ffmpeg 能解的格式。

### 2. 网络直链（不用先下载）

```bash
python scripts/course_kit.py frames --video "https://example.com/lecture.mp4?token=..." --out out/第1讲
```

ffmpeg 会**流式读取**，只取需要的字节区间（HTTP Range），不必先下载整个文件。实测 1 小时的课抽帧只需 3 分钟，流量远小于完整文件。

### 3. 在线教学平台（LMS）课程回放

不同学校的录播系统各不相同，本 skill 不绑定任何特定学校：在 `scripts/cvn/lms.py` 里参照 `SiteAdapter` 写一个适配器并注册到 `ADAPTERS`（通常只需改接口路径、字段名、以及哪一路是屏幕画面），然后用 `--site <名字>` 调用。先登录一次（弹出浏览器，用户自己操作），之后会话会被保存：

```bash
python scripts/course_kit.py lms login --site my-school --course 12345
```

然后列课、抓直链：

```bash
# 看看这门课有哪些课时
python scripts/course_kit.py lms list --site my-school --course 12345

# 抓全部课时的录像直链（默认只要屏幕画面）
python scripts/course_kit.py lms urls --site my-school --course 12345 -o urls.json
```

**关于「屏幕画面」**：很多录播系统每节课有两路录像——教室摄像头和电脑屏幕录制。课件都在后者。适配器通过 `screen_pattern` 正则识别哪一路是屏幕（例如按文件名前缀区分）。因为笔记要的是 PPT，不是老师的身影。

抓到的直链是 S3 预签名 URL，**有效期 6 天**。拿到后就是普通 HTTPS mp4，可以直接喂给 `frames`：

```bash
python scripts/course_kit.py frames --video "<urls.json 里第 N 条的 pc_url>" --out out/第1讲
```

## 快速开始

先体检，确认这台机器能跑哪些环节：

```bash
python scripts/course_kit.py probe
```

`probe` 会告诉你 ffmpeg/OCR/转写分别是否可用、缺什么怎么补。**永远先跑 probe**，它能省掉后面一半的报错。

然后一条命令跑完（不需要语音转写时加 `--skip-asr`，能省很多时间）：

```bash
python scripts/course_kit.py all --video <视频或直链> --out out/<课程名> --skip-asr
```

跑完得到 `out/<课程名>/` 下的：

- `course.md` —— 逐页笔记
- `knowledge.md` / `knowledge.csv` —— 知识点清单
- `slides/` —— 每一页课件的原图

## 产出三：复习讲义 PDF（重点推荐）

如果用户要的是「能背诵的复习资料」而不是「课堂记录」，用 `textbook`：

```bash
python scripts/course_kit.py textbook \
  --out out/第1讲 out/第2讲 out/第3讲 \
  --content 讲义内容.json \
  --pdf 复习讲义.pdf
```

`--out` 可以给多个课程目录，`--content` 是讲义内容 JSON（章节结构 + 正文）。

### 讲义内容怎么写

这是本 skill 最核心的一环——**由模型把课件内容重编成教材**，不是简单拼接。

**结构**：按学科逻辑分章，不按课堂时间顺序。例如可以是「为何学 → 学什么 → 怎么学 → 历史脉络 → 当代实践」。

**语言**：教材式叙述，**不要出现「课件显示」「老师讲到」「本页」这类元叙述**。直接陈述知识本身。

**必备的两个块类型**（这是「有考点」和「只讲了什么」的区别）：

`exam` 块 —— 考点，放在每节开头：

```json
{
  "type": "exam",
  "question": "如何理解「条件、过程、结果」三者的关系？",
  "items": [
    "「条件」是前提——它规定了后续发展得以成立的范围……",
    "「过程」是转化路径——它不是线性推进，而是逐层展开……",
    "「结果」是最终形态——条件与过程共同作用后的成熟状态……"
  ],
  "standard": "三者构成一个闭环：条件是前提，过程是方式，结果是归宿。"
}
```

`distinguish` 块 —— 易混辨析：

```json
{
  "type": "distinguish",
  "label": "辨析：民本 ≠ 民主",
  "text": "传统「民本」主体是君主（为民作主），现代「民主」主体是人民（由民作主）……"
}
```

完整块类型清单见 `references/content-schema.md`。

### 讲义排版

由 `cvn/review.py` 负责：封面、前言、目录、章节（考点框 / 定义框 / 引文块 / 对照表 / 要点框 / 补充框 / 辨析框）、复习要点。用本机 Chrome 无头打印，不需要 LaTeX / wkhtmltopdf。

**教材式排版**：宋体正文、苹方标题、深红主色、引文用金黄边框、补充说明用灰底。

## 产出二：逐页 PDF 笔记

如果用户想「对着录像复习、找回细节」，用 `pdf`：

```bash
python scripts/course_kit.py pdf --out out/<课程名> --title "课程名 · 第1讲"
```

每页课件配原图 + 版面还原的文字，老师讲解折叠成灰色小字。

**两种产出的区别**：

| | 逐页笔记（`pdf`） | 系统讲义（`textbook`） |
|---|---|---|
| 组织方式 | 按课堂时间顺序 | 按学科逻辑重组成章 |
| 适合 | 对照录像复习、找回细节 | 系统性复习、应考 |
| 讲解地位 | 作为注释保留 | 只用于补背景 |

## 分步跑（调试 / 只重做某一环）

每一步都能单独重跑，默认复用已有结果，`--force` 才重做：

```bash
python scripts/course_kit.py frames --video <视频> --out out/<课程名>
python scripts/course_kit.py ocr    --out out/<课程名>
python scripts/course_kit.py asr    --video <视频> --out out/<课程名>
python scripts/course_kit.py merge  --out out/<课程名> --min-score 3
```

典型场景：抽帧不满意只重跑 `frames --force`，后面的 `ocr` 会自动识别哪些页变了。

## 什么时候需要调参

| 现象 | 参数 |
|---|---|
| 课件翻页很快、漏页 | `--interval 1` |
| 两小时大课，想快一点 | `--interval 3` |
| PPT 有逐条动画，截到的是没出全的半页 | `--pick last` |
| 同一页被重复切成好几页 | `--change-ratio 0.15` |
| 页面漏检、不同页被并成一页 | `--change-ratio 0.06` |
| 目录里出现一堆只停留几秒的页 | `--min-page 20`（默认 12） |
| 画面底部有硬字幕被 OCR 当成内容 | `--mask-bottom 0.12` |
| 只关心中间的 PPT 区域（录屏带侧边栏） | `--crop 0.05,0.05,0.9,0.8` |
| 重点太少 / 太多 | `--min-score 2` / `--min-score 5` |
| 已有字幕，不想重跑转写 | `--from-srt 字幕.srt` |
| 只想要课件部分 | `--skip-asr` |
| 老师念的专有名词识别不准 | `--initial-prompt "课程名，术语1，术语2"` |
| 课件标题是艺术字，OCR 认成乱码 | `ocr --vlm`（用本地视觉模型读标题，需 Ollama） |
| whisper 模型跑不动 | `--model small`（默认 medium，CPU 上 1 小时课约 30~55 分钟） |

**抽帧阈值不用手调**。`--change-ratio` 默认自适应：纯 PPT 视频翻页幅度小（0.02~0.03），真人录屏幅度大（0.19~0.94），差一个量级，固定阈值必然两头不讨好。工具会先算本段素材的 ratio 分布再定阈值。

## 录屏类视频：只会留下真正的课件页

电脑内录的视频里往往混着桌面、浏览器、PPT 编辑界面——这些**不是课件**，但和课件一样是全屏 16:9，光看画面分不出来。

本工具会按画面上的文字判定页面类型（桌面 / 网页 / 编辑器 / 放映菜单 / 课件），**只把课件页写进笔记**。被跳过的页在 `aligned.json` 的 `skipped_pages` 里列出（带时间戳和原因）。

真实素材实测：60 分钟录屏，69 页 → 9 个课件页。

## 详细参考

| 文件 | 内容 |
|---|---|
| `references/content-schema.md` | 讲义内容 JSON 的完整格式（块类型、写作约定） |
| `references/DATA-CONTRACT.md` | 各阶段 JSON 的字段契约，改输出格式前必读 |
| `references/tuning.md` | 抽帧与划重点的调优手记（含实测数据） |
| `references/field-notes.md` | 真实素材的踩坑记录，历次测试的完整纪要 |

## 常识与边界

- **先 probe 再跑**。缺什么会提前说清楚，别等跑完才发现没有 OCR。
- **语音转写很慢**。60 分钟课用 medium 模型约 30~55 分钟。如果用户说过「讲解不重要」，直接 `--skip-asr`——课件本身就是主体。
- **whisper 的幻觉必须挡**。录屏开头静音时，whisper 会把 `initial_prompt` 当成语音吐出来（实测片段字词 100% 来自提示词）；长静音处还会编出一两分钟的超长片段。这两类都在 `asr.py` 里剔除，不要关掉。
- **OCR 原文不要改写**。课件上的公式、术语、编号就是复习依据。跨页自动纠错默认关闭——实测它会把「幻灯片第142张」改成「第190张」。
- **艺术字标题认不准是已知边界**。PPT 标题用 Word 艺术字时，Apple Vision 会输出「看起来像中文但错的字」。开 `--vlm` 用本地视觉模型（qwen3-vl:2b）能补救大部分，实测 11 页标题全部纠正。
- **讲义里的「标准表述」要谨慎**。如果 PPT 只有提纲没有成段定义，模型拟的标准表述可能和老师讲的不一致。写的时候标注哪些是补充，并提醒用户核对。
- **信息不足的地方标出来**，不要编。知识点清单里有「待补充 / 存疑」区，就是为此。

## 不要做

- 不要在没跑 `frames` 的情况下直接跑 `merge`——它依赖 `frames.json`
- 不要用 VLM 输出替代 OCR 原文去改写课件措辞，VLM 只补标题和图表说明
- 不要在讲义正文里写「课件显示」「老师讲到」——写成教材
- 不要在没有音轨的视频上硬跑 `asr`，用 `--skip-asr`
- 不要在用户没确认的情况下删掉 `out/` 目录重跑，增量复用能省很多时间
- 不要破解用户的浏览器 session——LMS 登录要用户自己操作，会话是加密存储的
