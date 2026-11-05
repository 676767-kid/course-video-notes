"""把一页课件的文字框重建成「可读的正文」。

为什么需要这一层
----------------
OCR 输出的是一堆**视觉行**，不是文章。直接按行拼接会得到：

    2026年09月
    17:02:09
    第—章绪论
    经济学的对象
    稀缺性与选择
    （排版噪声）)(D)()(品)((

用户要的是「PPT 上写了什么」，那就该是：

    第一章 绪论
    经济学的对象
    稀缺性与选择

本模块做四件事：
  1. **去页眉页脚**：时间戳、日期、机房号、页码、校名 logo
  2. **合并断行**：一句话被排成两行（PPT 里很常见）要接回去
  3. **识别层级**：标题 / 小标题 / 正文 / 引文 / 出处 / 页码
  4. **按版面排序**：自上而下、同行内自左而右

输出结构（写进 slides.json 的 `blocks` 字段）：

    [
      {"role": "heading",  "text": "核心问题是什么"},
      {"role": "subhead",  "text": "概念甲 · 概念乙 · 概念丙"},
      {"role": "bullet",   "text": "一个前提？一项条件？一种结果？"},
      {"role": "body",     "text": "其本质不是……"},
      {"role": "quote",    "text": "「若……之知……可以为……」"},
      {"role": "cite",     "text": "《某书·某篇》"},
    ]
"""

from __future__ import annotations

import re
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------
# 页眉页脚识别
# --------------------------------------------------------------------------

# 整行就是页眉/页脚（时间戳、日期、星期、机房号、页码、校名）
_CHROME_LINE = [
    # 日期：2026年01月01日 / 2026-01-01 / 2026/01/01
    re.compile(r"^\s*\d{4}\s*[-年/]\s*\d{1,2}\s*[-月/]\s*\d{1,2}\s*日?\s*$"),
    # 时间：17:02:09 / 17：02 / 16:46:33
    re.compile(r"^\s*\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?\s*$"),
    # 星期：星期二 / 周一
    re.compile(r"^\s*(?:星期|周)[一二三四五六日天]\s*$"),
    # 机房号/设备水印：形如 abc1_x307 / room-2307 / PC-X307
    re.compile(r"^\s*[a-zA-Z]{2,6}\s*\d?\s*[_\-]?\s*[xX×]?\s*\d{2,4}\s*$"),
    # 纯页码：3 / 3/28 / - 3 -
    re.compile(r"^\s*[-—–]?\s*\d{1,3}\s*[-—–]?\s*$"),
    re.compile(r"^\s*\d{1,3}\s*/\s*\d{1,3}\s*$"),
    # 校名 / 机构名（通用：独占一行的短页眉，如「某某大学」「XX UNIVERSITY」）
    re.compile(r"^\s*(?!(?:我|你|他|她|它|这|那|哪|如何|什么|为什么|怎样|为什|所谓)[^，。；：！？]{0,6}$)"
               r"[^，。；：！？,.;:!?]{0,24}(?:大学|大學|學院|学院|學校|学校|UNIVERSITY|COLLEGE|INSTITUTE)\s*$", re.I),
    # 不完整日期（2026年0 / 2026年09月 被截断）
    re.compile(r"^\s*\d{4}\s*年\s*\d{0,2}\s*月?\s*\d{0,2}\s*日?\s*$"),
    # 纯符号垃圾：)(D)()(品)(( / 装饰碎片
    re.compile(r"^[\s()（）\[\]{}<>,.?/\\|·•\-—_=+*#@%&~\"'`]*$"),
    re.compile(r"^\s*[)(（），,、·•]{2,}\s*$"),
]

# 一行里**混着**上述内容时，把它们切掉（保留剩下的正文）
_CHROME_FRAG = re.compile(
    r"\d{4}\s*年\s*\d{1,2}\s*月(?:\s*\d{1,2}\s*日)?"
    r"|\d{4}\s*[-/]\s*\d{1,2}\s*[-/]\s*\d{1,2}"
    r"|(?:星期|周)[一二三四五六日天]"
    r"|\d{1,2}\s*[:：]\s*\d{2}(?:\s*[:：]\s*\d{2})?"
    r"|[xX]\s*ier\s*1?\s*[_\-]?\s*[xX×]?\s*\d{0,4}",
    re.I,
)

# 那些"看起来像字但其实是装饰"的残渣
_JUNK_CHARS = re.compile(r"[)(（），,、·•\[\]{}<>|\\/]{3,}")


def is_chrome(line: str) -> bool:
    """整行是不是页眉/页脚。"""
    t = (line or "").strip()
    if not t:
        return True
    return any(p.match(t) for p in _CHROME_LINE)


def strip_chrome(line: str) -> str:
    """把一行里夹带的页眉片段切掉，返回剩下的正文（可能为空）。"""
    t = _CHROME_FRAG.sub(" ", line or "")
    t = _JUNK_CHARS.sub(" ", t)
    t = re.sub(r"\s{2,}", " ", t)
    # 句尾挂着的装饰残渣，如「……提纲》)（D)(品)（Q)」
    t = re.sub(r"[)(（），、|·•\s]{2,}$", "", t)
    t = re.sub(r"^[)(（），、|·•\s]{2,}", "", t)
    return t.strip(" .,:;、|-_·•")


# --------------------------------------------------------------------------
# 断行拼接
# --------------------------------------------------------------------------

# 上一行的结尾符号：说明这句话没说完，下一行要接上
_CONT_END = "，、；：,;:"
# 下一行的开头符号：说明它接着上一行
_CONT_START = "，。、；：）)」』”’,.;:"


_LABEL_LINE = re.compile(r"^[^：:]{2,20}[：:]")     # 「考勤：10分」这类标签行
_SHORT_ITEM = re.compile(r"^[^。！？]{2,14}$")          # 短条目


_OPEN_Q = "「『“‘\""
_CLOSE_Q = "」』”’\""


# 出处行：破折号+书名号，或「某某《某书》」这种署名+书名号
_CITE_START = re.compile(
    r"^\s*(?:——?\s*《|—\s*《|\(\s*《|（\s*《"
    r"|[^\s《》]{2,20}[（(]?《"          # 任意署名 + 书名号（不限具体人物）
    r"|《[^》]{1,30}》\s*$)"
)


def _looks_like_cite(text: str) -> bool:
    """这一行看起来是「出处」（——《某书·某篇》/ 某作者《某著作》）。"""
    return bool(_CITE_START.match((text or "").strip()))


def _quote_unclosed(text: str) -> bool:
    """这一行的引号是不是没闭合。"""
    t = text or ""
    opens = sum(t.count(c) for c in "「『“‘")
    closes = sum(t.count(c) for c in "」』”’")
    return opens > closes


def _is_standalone_item(text: str) -> bool:
    """这一行本身就是一个完整条目（不该被接到上一行）。"""
    t = (text or "").strip()
    if not t:
        return True
    if _BULLET_PREFIX.match(t):
        return True
    if _LABEL_LINE.match(t):      # 冒号标签行，独立成条
        return True
    if _SHORT_ITEM.match(t) and not t.endswith(tuple(_CONT_END)):
        return True
    return False


def _should_join(prev: str, nxt: str, prev_box: dict[str, Any] | None = None,
                 next_box: dict[str, Any] | None = None) -> bool:
    """判断 nxt 是不是 prev 这句话的续行。

    这里要非常克制：PPT 上的「要点列表」每一条本来就是独立的一句，
    乱接会把 6 条要点揉成一大段，用户反而看不出结构。
    只有在**明显是同一句被切断**时才接：

    1. 上一行以逗号/顿号/分号结尾（句子没说完）
    2. 下一行以标点开头（是上一行的尾巴）
    3. 上一行以「的/了/而/与/和」这类连接字结尾，且下一行不是新要点
    4. 版面证据：两行**紧挨着**（垂直间距小于半个行高），
       且下一行不是以项目符号/编号开头 —— 这是 PPT 自动换行的典型特征
    """
    p = (prev or "").rstrip()
    n = (nxt or "").lstrip()
    if not p or not n:
        return False

    # 引文续接：上一行开了引号还没闭合 → 这段引文没完，必须接回去。
    # 但要注意「已经闭合」的情况：闭引号之后紧跟的出处（——《某书》）
    # 不能粘进引文正文，所以这里要求下一行也不像出处。
    if _quote_unclosed(p) and not _looks_like_cite(n):
        return True

    # 下一行是新要点（项目符号/编号开头）→ 绝不接
    if _BULLET_PREFIX.match(n):
        return False
    # 下一行本身就是完整条目（冒号标签 / 短句）→ 不接
    if _is_standalone_item(n):
        return False
    # 上一行已经完整收尾 → 绝不接
    if p.endswith(("。", "！", "？", "!", "?", "」", "』", "”", "；", ";")):
        return False

    # 1) 上一行以句中标点结尾
    if p.endswith(tuple(_CONT_END)):
        return True
    # 2) 下一行以标点开头
    if n[0] in _CONT_START:
        return True
    # 3) 上一行以连接字结尾
    if p.endswith(("的", "了", "而", "与", "和", "或", "在", "是", "有", "为", "、", "，")):
        return True
    # 4) 版面证据：两行紧邻 + 上一行较长（说明是被宽度截断的）
    if prev_box and next_box:
        try:
            p_bottom = float(prev_box.get("y") or 0) + float(prev_box.get("h") or 0)
            n_top = float(next_box.get("y") or 0)
            p_h = float(prev_box.get("h") or 0) or 0.03
            gap = n_top - p_bottom
            # 紧挨着（间隙小于半行）且上一行不短
            if gap <= p_h * 0.6 and len(p) >= 16:
                return True
        except (TypeError, ValueError):
            pass
    return False


def join_wrapped(lines: Sequence[str]) -> list[str]:
    """把被 PPT 自动换行切断的句子接回去。"""
    out: list[str] = []
    for line in lines:
        t = (line or "").strip()
        if not t:
            continue
        if out and _should_join(out[-1], t):
            out[-1] = out[-1] + t
        else:
            out.append(t)
    return out


# --------------------------------------------------------------------------
# 层级判定
# --------------------------------------------------------------------------

_FULL_QUOTE = re.compile(r"^\s*[「『“\"'](.+)[」』”\"']\s*[。.]?\s*$")


def _is_full_quote(text: str) -> bool:
    """整行是不是一段完整的引文（而不是用引号强调的小标题）。

    判据：被一对引号整体包裹 **且** 有句子感（够长、含逗号句号、
    或是「某某说」引出的整段）。短促的「"某概念"」不算引文。
    """
    t = (text or "").strip()
    # 以引号开头就是引文候选（内部可能还有引号，不能用"整行一对引号"卡死）
    if not t or t[0] not in "「『“\"'":
        return False
    # 长引文：明显是一段话
    if len(t) >= 40:
        return True
    # 中等长度且有句子标点（说明是完整引用的一句）
    if len(t) >= 16 and re.search(r"[，。；,;]", t):
        return True
    return False


_BULLET_PREFIX = re.compile(r"^\s*(?:[•·▪◦‣∙]|[-–—*]|\d+[.、)]|[（(]\d+[）)]|[一二三四五六七八九十]+[、.])\s*")
_CITE = re.compile(
    r"^\s*(?:——《|—《|《|（《|\()"
    r"|^\s*[—–-]{1,2}\s*《"
    r"|^\s*[^\s《》]{2,20}[（(]?《"       # 任意署名 + 书名号（不限具体人物）
    r"|^\s*《[^》]{1,30}》\s*[（(]?[^)]{0,20}[）)]?\s*$"
)
_QUOTE_MARK = re.compile(r"^\s*[「『“\"']")
_PROPER_NOUN_LINE = re.compile(r"^\s*(?:一|二|三|四|五|六|七|八|九|十)[、.]\s*\S")


def guess_role(
    text: str,
    *,
    font_ratio: float,
    is_first: bool,
    y: float,
) -> str:
    """猜这一行在幻灯片里扮演什么角色。

    font_ratio = 该行字号 / 本页最大字号（0~1）。
    """
    t = (text or "").strip()
    if not t:
        return "body"
    if _CITE.search(t):
        return "cite"
    # 注意：只有「整行被一对引号包住」且内容较长，才算引文。
    # 「"某概念"」「某领域中的"某术语"」这类是 PPT 的**小标题**
    # （用引号强调），当成引用块会读起来很奇怪。
    if _is_full_quote(t):
        return "quote"
    # 字号明显大于正文 + 位置靠上 → 标题
    if font_ratio >= 0.85 and y <= 0.45 and len(t) <= 30:
        return "heading"
    if is_first and len(t) <= 24 and font_ratio >= 0.7:
        return "heading"
    if _BULLET_PREFIX.match(t):
        return "bullet"
    if _PROPER_NOUN_LINE.match(t):
        return "bullet"
    if font_ratio >= 0.7 and len(t) <= 18:
        return "subhead"
    return "body"


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def rebuild(
    rows: Sequence[tuple[str, dict[str, Any]]],
    *,
    max_font_ratio: float | None = None,
) -> list[dict[str, Any]]:
    """把 (文本, 框) 序列重建成带角色的正文块。

    rows 必须已按「自上而下、同行内自左而右」排好序。
    返回 [{"role", "text"}]，已去页眉页脚、已拼接断行。
    """
    if not rows:
        return []

    heights = [float(b.get("h") or 0.0) for _, b in rows if b]
    max_h = max(heights) if heights else 0.0
    if max_font_ratio is not None and max_h <= 0:
        max_h = max_font_ratio

    kept: list[tuple[str, dict[str, Any]]] = []
    for text, box in rows:
        raw = (text or "").strip()
        if not raw:
            continue
        if is_chrome(raw):
            continue
        cleaned = strip_chrome(raw)
        if not cleaned:
            continue
        # 清洗后只剩一两个字符、且原本是长行的，多半是装饰残留
        if len(cleaned) <= 1 and len(raw) > 6:
            continue
        # 几乎没有汉字、又都是符号/单字母的残渣（如「D)(品)（Q)」）
        han = len(re.findall(r"[\u4e00-\u9fff]", cleaned))
        if han == 0 and len(re.sub(r"[\s()（）\[\]{}<>,.?/\\|·•\-—=+*#@%&~\"'`]", "", cleaned)) <= 3:
            continue
        # 汉字极少但符号占多数
        if han <= 1 and len(cleaned) >= 3 and han / len(cleaned) < 0.25:
            continue
        # 汉字 ≤2 且整体很短、又不是有意义的词（「D 品」「)(Q)」这类残渣）
        if han <= 2 and len(cleaned) <= 4 and not re.search(r"[\u4e00-\u9fff]{2}", cleaned):
            continue
        kept.append((cleaned, box or {}))

    if not kept:
        return []

    # 先按行拼接断句（在同一个"视觉区域"内）
    merged: list[tuple[str, dict[str, Any]]] = []
    for text, box in kept:
        if merged and _should_join(merged[-1][0], text, merged[-1][1], box):
            prev_text, prev_box = merged[-1]
            merged[-1] = (prev_text + text, prev_box)
        else:
            merged.append((text, box))

    blocks: list[dict[str, Any]] = []
    heading_taken = False
    for i, (text, box) in enumerate(merged):
        h = float(box.get("h") or 0.0)
        line_conf = float(box.get("conf") or 0.0)
        ratio = (h / max_h) if max_h > 0 else 0.5
        role = guess_role(
            text,
            font_ratio=ratio,
            is_first=(i == 0),
            y=float(box.get("y") or 0.0),
        )
        # 一页只允许一个 heading：真正的主标题。
        # 后面字号也大的行（副标题、并列问句）降为 subhead/body，
        # 否则整页全是加粗，层级就没了。
        if role == "heading":
            if heading_taken:
                role = "subhead" if len(text) <= 24 else "body"
            else:
                heading_taken = True
        blocks.append({"role": role, "text": text, "conf": round(line_conf, 3)})
    return blocks


def blocks_to_text(blocks: Sequence[dict[str, Any]], *, sep: str = "\n") -> str:
    """把块拼成纯文本（给下游做检索/CSV 用）。"""
    return sep.join(str(b.get("text") or "") for b in blocks if b.get("text"))
