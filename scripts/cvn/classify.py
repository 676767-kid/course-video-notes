"""页面类型判定：这一帧到底是「课件」，还是桌面 / 浏览器 / 编辑器。

录屏类课程视频里，开头往往是桌面（用户在调设备），中间可能打开浏览器查资料，
结尾可能切到 PPT 编辑界面。这些画面**不是课件**，却和课件长得一样是全屏 16:9，
光靠画面差异分不出来，必须看画面上的文字。

判据（多信号投票，任一命中即判为非课件）：

1. **界面特征词**：浏览器导航（"设为首页" "English Version" "站内搜索"）、
   桌面图标（"此电脑" "回收站" "腾讯会议"）、PPT 功能区（"视图" "开始" "插入"
   "幻灯片" "切换"）、放映控制（"上一张" "下一张" "自定义放映"）……
2. **文字密度过高**：课件页通常只有几行（标题 + 要点 + 页脚），
   而桌面/网页会印出几十行短词。行数多而长句少 = 界面而非课件。
3. **平均行长过短**：课件正文多是完整句子，界面元素多是 2~4 字词块。

判定结果写进 slides.json 的 `kind` 字段：

    "slide"    真课件（有讲义内容）
    "desktop"  桌面
    "browser"  浏览器 / 网页
    "editor"   PPT / 文档编辑界面
    "ui"       播放器控制条、放映菜单等
    "blank"    几乎无文字（黑屏、纯图）
    "unknown"  文字太少，判不出来

notes 层默认只把 `kind == "slide"` 的页写进课程笔记；其余页仍然保留在
frames.json / slides.json 里，用户想看可以在 knowledge.md 的「未计入」区看到。
"""

from __future__ import annotations

import re
from typing import Any, Sequence

# --------------------------------------------------------------------------
# 界面特征词
# --------------------------------------------------------------------------

# 浏览器 / 网页
_BROWSER_WORDS = (
    "设为首页", "English Version", "站内搜索", "隐私政策", "Cookie",
    "版权所有", "京ICP", "公安备案", "网站地图", "联系我们",
    "院系设置", "组织机构", "本科书院", "师资队伍", "人才培养",
    "科学研究", "医学教育", "学校概况", "站内", "登录",
    "注册", "更多", "分享", "关注", "推荐",
    "反馈", "设置", "切换帐号", "使用手机微信扫码", "wX.aq.com",
    "视频", "图片", "资讯", "地图", "文库", "百科", "翻译",
    # 教务/考勤系统：录屏里老师会打开点名页面，界面和课件一样是全屏 16:9，
    # 但内容是学生名单，不能当讲义写进笔记。
    "电子考勤管理平台", "课堂点名", "统一入口", "考勤管理",
    "学生名单", "签到", "缺勤", "出勤",
)
# 桌面
_DESKTOP_WORDS = (
    "此电脑", "回收站", "腾讯会议", "雨课堂", "微信网页版", "综合搜索",
    "新建文件夹", "重命名", "属性", "打开方式", "所有程序",
)
# 文档 / 演示编辑界面
_EDITOR_WORDS = (
    "视图", "开始", "插入", "设计", "切换", "动画", "幻灯片", "页面布局",
    "参考资料", "邮件", "审阅", "视图 工具", "重置", "新建幻灯片",
    "复制", "剪切", "格式刷", "字体", "字号", "段落",
)
# 放映 / 播放控制
_PLAYER_WORDS = (
    "上一张", "下一张", "自定义放映", "显示演示者视图", "放大",
    "上次查看的位置", "查看所有幻灯片", "屏幕", "全屏", "退出",
    "从头开始", "暂停", "播放", "进度条", "倍速", "静音",
)

_ALL_UI_WORDS = _BROWSER_WORDS + _DESKTOP_WORDS + _EDITOR_WORDS + _PLAYER_WORDS

# 按类别分别建正则，便于给出「为什么判成这一类」的理由
# 顺序即优先级：浏览器要在桌面之前判，因为网页导航里也常出现
# "下载 / 图片 / 设置 / 搜索" 这些桌面通用词。
_CATEGORY_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("editor", re.compile("|".join(map(re.escape, _EDITOR_WORDS)))),
    ("browser", re.compile("|".join(map(re.escape, _BROWSER_WORDS)))),
    ("desktop", re.compile("|".join(map(re.escape, _DESKTOP_WORDS)))),
    ("ui", re.compile("|".join(map(re.escape, _PLAYER_WORDS)))),
)

# 出现这些词基本只可能是网页/软件界面，课件里不会有
# 注意：每个分支都必须有实际内容。之前写成 r"|A|B" 会产生零宽匹配，
# findall 返回一堆空串，导致真课件页被误判成界面页。
_STRONG_UI = re.compile(
    r"English\s*Version"
    r"|站内搜索"
    r"|此电脑"
    r"|回收站"
    r"|上一张"
    r"|下一张"
    r"|自定义放映"
    r"|显示演示者视图"
    r"|幻灯片浏览"
    r"|剪贴板"
    r"|字号\s*[:：]"
    r"|微信网页版"
    r"|扫码"
    r"|wX\.aq\.com"
    r"|使用手机微信"
    r"|切换帐号"
    r"|反馈\s*设置"
    r"|软件\s*良医"
    r"|电子考勤管理平台"
    r"|课堂点名"
    r"|考勤管理平台"
    r"|统一入口\s*[-—]",
    re.I,
)

# 时间戳 / 水印行：几乎每页都有，不该算作内容行
_CHROME_LINE = re.compile(
    r"^\s*(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日|星期[一二三四五六日]|"
    r"[a-zA-Z]{2,6}\d?\s*[_\-]?\s*[xX×]?\s*\d{2,4}|©.*|\d{1,2}:\d{2}(?::\d{2})?)\s*$",
    re.I,
)


def _content_lines(lines: Sequence[str]) -> list[str]:
    """去掉时间戳/水印这类每页都有的固定行，剩下的才算内容。"""
    return [l for l in lines if l.strip() and not _CHROME_LINE.match(l.strip())]


def classify_page(
    lines: Sequence[str],
    *,
    max_content_lines: int = 14,
    min_avg_len: float = 7.0,
) -> dict[str, Any]:
    """判定一页的性质。

    返回 {"kind": str, "reason": str, "metrics": {...}}：
      kind    slide / desktop / browser / editor / ui / blank / unknown
      reason  中文说明（会写进 notes，方便用户理解为什么这页没进笔记）
      metrics 判定依据的数字（行数、平均行长、命中的界面词）
    """
    cleaned = [str(l).strip() for l in lines if str(l).strip()]
    content = _content_lines(cleaned)
    n_all = len(cleaned)
    n_content = len(content)
    avg_len = (sum(len(l) for l in content) / n_content) if n_content else 0.0
    long_lines = sum(1 for l in content if len(l) >= 12)

    blob = " ".join(cleaned)
    strong_hits = _STRONG_UI.findall(blob)
    strong_n = len(strong_hits)

    # 分类别命中数
    cat_hits: dict[str, list[str]] = {}
    for cat, pat in _CATEGORY_PATTERNS:
        found = pat.findall(blob)
        if found:
            cat_hits[cat] = sorted(set(found))[:4]

    metrics = {
        "lines": n_all,
        "content_lines": n_content,
        "avg_len": round(avg_len, 1),
        "long_lines": long_lines,
        "ui_hits": strong_n,
    }

    # --- 规则 0：几乎没有文字 ---
    # 注意：不能只看"行数"。不同 OCR 引擎输出的框粒度不一样——
    # Apple Vision 会把一行拆成好几个框（含时间戳/水印），
    # RapidOCR 则倾向把整行合成一个框。
    # 同一张课件，Vision 给 5 行、RapidOCR 只给 1 行；若按行数判会误杀。
    # 所以这里看的是**文字总量**：只有一行但有 20 多字，显然是正常课件。
    content_chars = sum(len(l) for l in content)
    if content_chars == 0:
        return {"kind": "blank", "reason": "画面无文字（黑屏或纯图）", "metrics": metrics}
    if n_content <= 1 and content_chars <= 8 and n_all <= 3:
        return {"kind": "blank", "reason": "画面几乎无文字（黑屏或纯图）", "metrics": metrics}

    # --- 规则 1：强界面特征词 ---
    # 两个及以上强特征词，或单类特征词密集出现 → 一定是界面
    if strong_n >= 1:
        kind = "ui"
        for cat in ("editor", "browser", "desktop", "ui"):
            if cat_hits.get(cat):
                kind = cat
                break
        return {
            "kind": kind,
            "reason": "画面含界面特征词：" + "、".join(
                sorted(set(strong_hits))[:3]
            ),
            "metrics": metrics,
        }

    # --- 规则 2：文字密度过高且几乎都是短词 ---
    if n_content >= max_content_lines and long_lines <= 1:
        return {
            "kind": "ui",
            "reason": f"文字行数过多（{n_content} 行）且几乎没有完整句，"
                      "像是软件界面而不是课件",
            "metrics": metrics,
        }

    # --- 规则 3：平均行长过短（满屏短词） ---
    if n_content >= 8 and avg_len < min_avg_len and long_lines == 0:
        return {
            "kind": "ui",
            "reason": f"共 {n_content} 行但平均每行仅 {avg_len:.1f} 字，"
                      "更像图标菜单而不是讲义",
            "metrics": metrics,
        }

    # --- 规则 4：单类界面词密集 ---
    for cat, words in cat_hits.items():
        if len(words) >= 3:
            return {
                "kind": cat,
                "reason": f"出现多处{_CAT_LABEL.get(cat, cat)}特征：" + "、".join(words),
                "metrics": metrics,
            }

    # --- 规则 5：只有一两个界面词，但行数也多 → 仍判界面 ---
    if strong_n >= 1 and n_content >= max_content_lines:
        return {
            "kind": "ui",
            "reason": f"含界面特征「{strong_hits[0]}」且文字行数偏多（{n_content} 行）",
            "metrics": metrics,
        }

    # --- 默认：真课件 ---
    if n_content == 0:
        return {"kind": "blank", "reason": "去掉页眉页脚后没有内容", "metrics": metrics}
    return {"kind": "slide", "reason": "", "metrics": metrics}


_CAT_LABEL = {
    "desktop": "桌面",
    "browser": "网页",
    "editor": "编辑器",
    "ui": "界面",
}


def is_content_page(kind: str) -> bool:
    """只有 slide 算真正的课件内容页。"""
    return kind == "slide"
