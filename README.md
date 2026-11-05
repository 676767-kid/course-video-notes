# course-video-notes

把课程录像变成**系统复习讲义**。

支持本地视频、网络直链、以及在线教学平台（LMS）的课程回放。输出三种成品：逐页笔记、知识点清单、图文并茂的复习讲义 PDF。

```
录像 ──▶ 抽帧去重 ──▶ 课件识字 ──▶ 版面重建 ──▶ 编排讲义
         自适应阈值    RapidOCR/Vision   还原标题层级   含考点与标准表述
         区分翻页/动画   +VLM 修艺术字     去页眉接断行
```

## 它解决什么问题

录屏转文字的工具很多，但「能用来复习」的很少。这里的三个关键处理：

**1. 只留真课件页。** 录屏开头是桌面、中间切到浏览器、结尾是 PPT 编辑器——这些和课件一样是满屏 16:9，光看画面分不出来。本工具按画面上的文字判定页面类型，只把课件页写进笔记。

**2. 按版面重建正文。** OCR 输出的是视觉行，混着时间戳、水印、断行。需要按版面把它重建成「标题 / 要点 / 引文 / 出处」的结构，才是可读的文章。

**3. 「讲了什么」≠「考什么」。** 讲义模式下，每个知识点会给出**设问 + 编号要点 + 可抄进答卷的标准表述**，而不是课堂流水账。

## 快速开始

```bash
# 1) 体检：这台机器能跑哪些环节
python3 scripts/course_kit.py probe

# 2) 一条命令跑完（不需要语音转写时加 --skip-asr，能省大量时间）
python3 scripts/course_kit.py all --video 第3讲.mp4 --out out/第3讲 --skip-asr

# 3) 出复习讲义
python3 scripts/course_kit.py textbook --out out/第3讲 \
  --content 讲义内容.json --pdf 复习讲义.pdf
```

产物在 `out/第3讲/`：

| 文件 | 内容 |
|---|---|
| `course.md` | 逐页笔记（课件原文 + 讲解） |
| `knowledge.md` / `knowledge.csv` | 知识点清单 |
| `slides/` | 每页课件原图 |
| `transcript.srt` | 字幕（如果跑了转写） |

## 三种输入

```bash
# 本地文件
python3 scripts/course_kit.py all --video 第3讲.mp4 --out out/第3讲

# 网络直链（流式读取，不用先下载整个文件）
python3 scripts/course_kit.py frames --video "https://example.com/lecture.mp4?token=..." --out out/第1讲

# 在线教学平台（列课 → 抓直链 → 处理；先写好本校适配器，见下）
python3 scripts/course_kit.py lms login --site my-school --course 12345
python3 scripts/course_kit.py lms list  --site my-school --course 12345
python3 scripts/course_kit.py lms urls  --site my-school --course 12345 -o urls.json
```

## 支持哪些平台

本工具**不绑定任何特定学校**。各校录播系统差异很大，需要为本站写一个**站点适配器**——通常只改三处：接口路径、字段名、哪一路是屏幕画面：

```python
# scripts/cvn/lms.py
MY_SCHOOL = SiteAdapter(
    name="my-school",
    base_url="https://lms.example.edu.cn",
    list_endpoint="/api/course/{course}/coursewares?page=1&page_size=100",
    screen_pattern=r"/screen[^/]*\.mp4",   # 哪一路是电脑屏幕
    camera_pattern=r"/camera[^/]*\.mp4",
)
ADAPTERS["my-school"] = MY_SCHOOL
```

然后在命令里加 `--site my-school`。适配器只负责描述差异，抓取逻辑是共用的。

> 如果你为本校写了适配器，欢迎 PR。

## 环境要求

| 能力 | 依赖 | 必需性 |
|---|---|---|
| 抽帧 | ffmpeg | **必需** |
| 课件识字 | RapidOCR（中文首选）/ Apple Vision / tesseract | 至少一个 |
| 艺术字标题 | Ollama + 一个视觉模型（如 `qwen3-vl:2b`） | 可选，`--vlm` 启用 |
| 语音转写 | faster-whisper | 可选，`course_kit.py setup` 自动装 |
| PDF 生成 | Chrome / Chromium / Edge | 出 PDF 时必需 |
| LMS 抓取 | Node + playwright-core，或 Python playwright | 只用学校回放时需要 |

安装提示：

```bash
brew install ffmpeg                     # macOS
pip install rapidocr-onnxruntime        # 中文 OCR，强烈推荐
pip install playwright && playwright install chromium
```

## 文档

- [`references/content-schema.md`](references/content-schema.md) —— 讲义内容 JSON 的格式与写作约定
- [`references/DATA-CONTRACT.md`](references/DATA-CONTRACT.md) —— 各阶段 JSON 字段契约
- [`references/tuning.md`](references/tuning.md) —— 抽帧与划重点调优手记（含实测数据）
- [`references/field-notes.md`](references/field-notes.md) —— 真实素材的踩坑记录

## 设计取舍

**为什么抽帧阈值自适应？** 纯 PPT 视频的翻页幅度是 0.02~0.03，真人录屏是 0.19~0.94，差一个量级。固定阈值必然两头不讨好，所以先算本段素材的分布再定阈值。

**为什么 OCR 要跨引擎？** Apple Vision 对中文印刷体很好，但对艺术字标题会输出「看起来像中文但错的字」；RapidOCR（PP-OCR 的 ONNX 版）中文更强但慢一些。默认 hybrid，必要时叠加本地视觉模型读标题。

**为什么不做自动纠错？** 试过用「同一门课里的高置信度文本」纠正低置信度错字，实测会把「幻灯片第142张」改成「第190张」——风险大于收益。宁可留错字让用户对照原图。

**为什么分词不靠外部词典？** 早期版本依赖 jieba 判断生僻词，但很多环境里装不上，导致判据静默失效。现在改成基于字符集（GB2312 常用字范围）和字重合度，无外部依赖。

## 已知限制

- 艺术字标题、繁体标题、被水印压住的标题，OCR 可能读错；开 `--vlm` 能救回大部分。
- 语音转写慢（CPU 上 1 小时课约 30~55 分钟）。如果老师口述对复习不重要，直接 `--skip-asr`。
- 需要登录/付费的课程，工具不破解鉴权——登录要用户自己在弹出的浏览器里完成。
- 讲义的「标准表述」由模型生成，可能和任课教师的说法有出入，需人工核对。

## License

MIT
