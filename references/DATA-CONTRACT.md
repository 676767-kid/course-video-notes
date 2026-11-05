# course-video-notes 数据契约（冻结版）

所有阶段（frames / ocr / asr / align / render）之间只通过下面这些 JSON 文件通信。
字段名、单位、排序规则**不得擅自改动**；要加字段只能追加，不能改语义。

## 0. 目录布局

```
out/<课程名>/
├── course.md            主产物：重点速览 + 课件目录 + 每页讲解
├── knowledge.md         知识点清单（便于二次加工）
├── knowledge.csv        同上，表格版（Excel/WPS 直接打开）
├── emphasis.json        划重点数据，给后续检索/排序
├── meta.json            本次运行的环境与参数快照
├── manifest.json        所有阶段产物索引 + 阶段完成标记
├── slides/              slide_0001.png ... （原图，可直接嵌文档）
├── transcript.json      ASR 逐字稿（带时间戳）
├── transcript.srt       标准字幕
└── frames.json          抽帧 + 去重结果
```

## 1. `manifest.json`

```json
{
  "version": 1,
  "video": "/abs/path/lecture.mp4",
  "video_id": "lecture",
  "duration": 3725.4,
  "stages": {
    "frames": {"done": true, "at": "2026-01-01T00:00:00", "params": {}},
    "ocr":    {"done": true, "at": "...", "params": {}},
    "asr":    {"done": true, "at": "...", "params": {}},
    "align":  {"done": true, "at": "..."},
    "merge":  {"done": true, "at": "..."}
  }
}
```
`stages.<name>.done` 为下游的硬前置：`merge` 之前必须有 `frames`、`ocr`、`align`、`asr`。

## 2. `frames.json`

```json
{
  "duration": 3725.4,
  "interval": 2.0,
  "strategy": "interval|scene|both",
  "count": 128,
  "slides": [
    {
      "index": 1,             // 1 起，连续，即 slides/slide_0001.png
      "file": "slides/slide_0001.png",
      "t": 0.0,               // 该页首次出现的秒数（page start）
      "t_end": 41.5,          // 下一页出现的秒数；最后一页 = duration
      "frame_t": 38.0,        // 实际截取的那一帧的秒数（--pick last 时为页面末尾）
      "similarity": 1.0       // 与上一页的相似度，1.0 = 新页
    }
  ]
}
```
排序：严格按 `t` 升序，`index` 与数组下标一一对应（index = i+1）。

## 3. `slides.json`（ocr 阶段产物，位于 out 根目录）

```json
{
  "engine": "rapidocr|hybrid|vision|vlm|none",  // hybrid = RapidOCR + Apple Vision 互补
  "slides": [
    {
      "index": 1,
      "lines": ["第三章 电枢反应", "1. 交轴电枢反应", "..."],  // 版面序，已清洗
      "text": "第三章 电枢反应 1.交轴电枢反应 ...",              // 行拼接成段
      "title": "第三章 电枢反应",                                 // 最可能的标题行，可为 ""
      "title_ocr": "第三章 电枢反戍",                             // 标题修复前的 OCR 原文（审计用，可选）
      "title_source": "课件版面",                                 // 课件版面 / 跨页标题 / VLM 读图 / 讲稿术语（可选）
      "has_figure": false,        // 检测到图表/公式区域
      "figure_note": "",          // VLM 补充说明，无则空串
      "ocr_conf": 0.93,           // 平均置信度，无则 0.0
      "kind": "slide",            // slide|desktop|browser|editor|ui|blank|unknown（分类阶段回填）
      "kind_reason": "",          // 判成该类型的证据词（可选）
      "low_conf_lines": [],       // 置信度明显偏低的 OCR 行（可选）
      "vlm_model": "",            // 实际使用的 VLM 模型名（可选）
      "_sig": "..."               // 增量缓存签名（内部字段，可忽略）
    }
  ]
}
```

## 4. `transcript.json`

```json
{
  "engine": "faster-whisper|srt-import",
  "model": "medium",              // 默认 medium；large-v3-turbo 更快但中文更碎
  "initial_prompt": "课程名，核心术语1，核心术语2，……",  // 实际使用的提示词（可选）
  "language": "zh",
  "duration": 3725.4,
  "segments": [
    {"i": 0, "start": 12.4, "end": 18.9, "text": "看这个图，考试必考。", "slide": 1}
  ]
}
```
`segments` 按 `start` 升序；`slide` 由 align 阶段回填（1 起，0 = 未归属）。

## 5. `aligned.json`（align 阶段产物）

```json
{
  "slides": [
    {
      "index": 1,
      "t": 0.0, "t_end": 41.5,
      "title": "第三章 电枢反应",
      "lines": ["..."],
      "text": "...",
      "has_figure": false, "figure_note": "",
      "speech": "老师在这一页说的话，按句拼接",
      "segments": [0, 1, 2],
      "emphasis": [
        {"text": "考试必考", "score": 5, "t": 18.9, "reason": "考试信号词+重复"}
      ],
      "key_terms": ["电枢反应", "交轴"]
    }
  ],
  "global_emphasis": [
    {"text": "换向与火花", "score": 9, "t": 120.0, "slide": 3, "reason": "出现 4 次"}
  ]
}
```

## 6. `emphasis.json`

```json
{
  "min_score": 3,
  "items": [
    {"text": "…", "score": 9, "t": 120.0, "slide": 3,
     "reasons": ["考试信号词", "重复 4 次"], "source": "speech|slide"}
  ]
}
```
`items` 按 `score` 降序，再按 `t` 升序。

## 7. `knowledge.csv` 列定义（固定，供二次加工）

```
id,知识点,一句话说明,所属页,时间,重要度,类型,前置知识点,易错点
```
- `id`：`K001` 递增
- `所属页`：整数页号（无则空）
- `时间`：`mm:ss` 或 `h:mm:ss`
- `重要度`：1–5 整数
- `类型`：`概念|公式|步骤|结论|辨析|案例`
- 其余字段可为空串

## 8. 退出码约定

`0` 成功；`1` 用户可见错误（CvnError，只打印原因不打印 traceback）；`2` 用法错误。
