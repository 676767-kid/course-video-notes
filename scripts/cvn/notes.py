"""笔记生成层：把「课件识别结果 + 逐字稿」融合成可直接复习的课程笔记。

本模块承担流水线的后两个阶段：

    align   帧/OCR/ASR 三种产物按时间轴对齐，并完成「划重点」打分
            -> 写 out_dir/aligned.json（契约 §5）
    render  把 aligned.json 渲染成人能直接用的 Markdown / CSV
            -> 写 course.md、knowledge.md、knowledge.csv、emphasis.json（契约 §6/§7）

设计目标（产品视角，最终用户是中文大学生）：

1. course.md 要能「直接翻开就复习」：每页课件写了什么、老师讲了什么、哪句是重点。
2. knowledge.md / knowledge.csv 要能「二次加工」：一条一个知识点、一句话说明、
   带页号与时间戳、重要度 1-5、类型分类、易错点，可直接转 Anki 卡片或思维导图。
3. 划重点宁可少而准，不可满屏高亮；每条重点都必须能用一句话解释「为什么是重点」。

约束：
- 只用标准库（Pillow 也不需要，只在图片存在时按路径嵌入）。
- 只读写 out_dir 下的契约文件，不改契约字段语义；新增字段只做追加。
- 对齐阶段会按契约 §4 的注释把 segment["slide"] 回填进 transcript.json。

自测：`python3 -m cvn.notes /tmp/xxx`（见 main）。
"""

from __future__ import annotations

import argparse
import bisect
import csv
import re
import unicodedata
from collections import Counter, defaultdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # 既支持 `from cvn import notes`（包内），也支持 `python3 notes.py`（同目录脚本）
    from . import util
except ImportError:  # pragma: no cover - 直接脚本运行时走这里
    import util  # type: ignore[no-redef]

CvnError = util.CvnError

__all__ = ["align", "render", "main"]

# 重点分值的上下限：>= 10 已经是「老师把话说死了」，再高没有区分度。
_SCORE_MAX = 10

# 逐页划重点的条数上限：一页塞太多重点等于没有重点。
_MAX_PER_PAGE = 5
# 重点速览条数上限。
_MAX_OVERVIEW = 12


# ==========================================================================
# 一、信号词规则表
#
# 每条规则是 (正则, 权重, 说明模板)。说明模板里的 {tok} 会被替换成实际命中的词，
# 让用户一眼看到「为什么这句被标红」。
#
# 权重设计原则：
#   4 = 老师把考试/分数直接说出来了（必考、就考这个）
#   3 = 学习要求或要求背诵（记住、务必、考点、常考）
#   2 = 提醒注意 / 指向内容（注意、看这里、重点）
#   1 = 弱信号，单独不足以成为重点，但能和别的信号叠加（因果、序号、数字）
# 阈值默认 3：单独一个弱信号不会冒出来，避免满屏高亮。
# ==========================================================================

_EXAM_RULES: List[Tuple[re.Pattern, int, str]] = [
    (
        re.compile(r"必考|一定要考|考试就考|就考这|考的就是|必出|稳考|重点考察"),
        4,
        "考试信号词「{tok}」",
    ),
    (
        re.compile(r"期中|期末|考点|常考|会考|可能考|题型|易错|失分点|扣分点|考试"),
        3,
        "考试相关词「{tok}」",
    ),
    (
        re.compile(
            r"大家记住|记住|记牢|背下来|切记|务必|千万|一定要|必须掌握|必须会|划下来|标记下来"
            r"|再强调一遍|再次强调|我强调"
        ),
        3,
        "老师要求记住「{tok}」",
    ),
    (
        re.compile(r"注意|留神|小心|很容易错|容易搞混|容易弄错|不要混|别搞错"),
        2,
        "提醒注意「{tok}」",
    ),
    (re.compile(r"掌握|复习|作业|练习|重点"), 2, "学习要求「{tok}」"),
    (
        re.compile(r"看这里|看这个|看这张图|看屏幕|看黑板|这里很关键|这个很重要|本条很重要|看这里"),
        2,
        "老师指向内容「{tok}」",
    ),
]

_DEF_RULES: List[Tuple[re.Pattern, int, str]] = [
    (
        re.compile(r"是指|指的是|定义为|叫做|称为|即为|意思是"),
        2,
        "定义/解释句式「{tok}」",
    ),
    (re.compile(r"本质|实质|核心|关键在于|根本原因"), 2, "点明本质「{tok}」"),
    (re.compile(r"综上|总之|归纳一下|小结|结论|由此可见|可见|所以|因此"), 2, "结论/总结「{tok}」"),
    (re.compile(r"因为|由于|之所以|原因"), 1, "因果推理「{tok}」"),
    (re.compile(r"也就是|也就是说|换言之|换句话说|等于说"), 1, "同义复述「{tok}」"),
]

_ENUM_RULE = (
    re.compile(
        r"(?:^|[，。；、：\s])(第[一二三四五六七八九十]+|首先|其次|再次|最后|一是|二是|三是"
        r"|①|②|③|④|⑤)"
    ),
    1,
    "序号枚举「{tok}」",
)

_FORMULA_RULES: List[Tuple[re.Pattern, int, str]] = [
    (re.compile(r"[A-Za-zΦΨΩΩΔδ][A-Za-z0-9_]*\s*[=＝]|[=＝]\s*[A-Za-z0-9Φ]"), 2, "含公式「{tok}」"),
    (
        re.compile(r"\d+(?:\.\d+)?\s*(?:V|A|W|kW|MW|Ω|Hz|r/min|rpm|N·m|kW·h|℃|%|倍|匝|极)"),
        1,
        "含数值要点「{tok}」",
    ),
    (re.compile(r"\d"), 1, "含数字「{tok}」"),
]


# ==========================================================================
# 二、文本小工具
# ==========================================================================


def _norm(text: str) -> str:
    """NFKC 归一化（全角转半角），只用于匹配，不用于输出原文。"""
    return unicodedata.normalize("NFKC", text or "")


_MATCH_PUNCT_RE = re.compile(r"[\s，,。、；;！!？?“”‘’（）()《》【】\[\]·—…]+")


def _match_norm(text: str) -> str:
    """匹配用的紧凑形式：NFKC + 去空白/标点 + 小写。

    课件标题和讲稿的标点经常不一致（「核心问题是什么?」vs「核心问题是什么？」），
    不去标点会把同一个概念拆成两条知识点。保留 ASCII 的 . : / = + -，
    避免把公式/编号（1.2、A/B）误合并。
    """
    return _MATCH_PUNCT_RE.sub("", _norm(text)).lower()


def _near_title(line: str, title: str) -> bool:
    """这一行是不是标题本身的 OCR 变体（「核心问提是什么?」≈「核心问题是什么」）。

    这种行不能拿来当标题的说明：它只是同一句话被认错了一个字。
    """
    a, b = _match_norm(line), _match_norm(title)
    if not a or not b:
        return False
    if a == b:
        return True
    if abs(len(a) - len(b)) > 2:
        return False
    return SequenceMatcher(None, a, b).ratio() >= 0.65


# 允许句末没有标点（ASR 常见），也把分号当句末，方便逐句划重点。
_SENT_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")


def split_sentences(text: str) -> List[Tuple[str, int]]:
    """把一段文本切成句子，同时给出每句的字符起始偏移。

    与 util.sentences 的区别是保留偏移：ASR 片段内部按字数比例估算句子时间戳时
    需要偏移量，这样「重点」的时间点能落到秒级而不是整段开头。

    返回 [(句子, 起始偏移), ...]。
    """
    out: List[Tuple[str, int]] = []
    for m in _SENT_RE.finditer(text or ""):
        raw = m.group(0)
        seg = raw.strip()
        if not seg:
            continue
        lead = len(raw) - len(raw.lstrip())
        out.append((seg, m.start() + lead))
    return out


def _disp_width(text: str) -> int:
    """按终端/等宽字体估算显示宽度：CJK 全角算 2，其余算 1。"""
    return sum(2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1 for ch in text)


def _md_cell(text: str) -> str:
    """把任意文本塞进 Markdown 表格单元格：转义竖线、压掉换行。"""
    s = re.sub(r"\s*\n\s*", " ", str(text or "")).strip()
    return s.replace("|", "／")


def md_table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> List[str]:
    """渲染对齐的 Markdown 表格（按显示宽度补空格，源码看起来也整齐）。"""
    cells = [[_md_cell(c) for c in row] for row in rows]
    widths = [_disp_width(h) for h in headers]
    for row in cells:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], _disp_width(cell))

    def pad(text: str, width: int) -> str:
        return text + " " * max(0, width - _disp_width(text))

    lines = ["| " + " | ".join(pad(h, widths[i]) for i, h in enumerate(headers)) + " |"]
    lines.append("| " + " | ".join("-" * max(3, widths[i]) for i in range(len(headers))) + " |")
    for row in cells:
        padded = [pad(cell, widths[i]) for i, cell in enumerate(row)]
        lines.append("| " + " | ".join(padded) + " |")
    return lines


def _stars(n: int) -> str:
    """重要度 3 -> ★★★☆☆（一眼能看出轻重）。"""
    n = max(0, min(5, int(n)))
    return "★" * n + "☆" * (5 - n)


def _clamp(value: int, lo: int, hi: int) -> int:
    """把整数夹到 [lo, hi]，用于重要度等有硬范围的字段。"""
    return max(lo, min(hi, value))


def _pages_label(pages: Sequence[int]) -> str:
    """页号列表 -> CSV 里的「所属页」文本：1 页写 "3"，多页写 "1,3"（契约允许逗号列）。"""
    uniq = sorted({int(p) for p in pages})
    return ",".join(str(p) for p in uniq)


def _pages_text(p: Dict[str, Any]) -> str:
    """页号列表 -> Markdown 里的人话：「第 3 页」/「第 1、3 页」。"""
    uniq = sorted({int(x) for x in (p.get("pages") or [p.get("page") or 0])})
    uniq = [x for x in uniq if x > 0]
    if not uniq:
        return "（无页号）"
    if len(uniq) == 1:
        return f"第 {uniq[0]} 页"
    return "第 " + "、".join(str(x) for x in uniq) + " 页"


# ==========================================================================
# 三、术语抽取（纯 stdlib：CJK n-gram + 频次 + 停用词 + 嵌套剪枝）
# ==========================================================================

_CJK_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]+")
_LATIN_TERM = re.compile(r"(?<![A-Za-z])[A-Za-z][A-Za-z0-9]{1,}(?:[-_][A-Za-z0-9]+)*")
_GREEK_TERM = re.compile(r"[αβγδεζηθικλμνξοπρστυφχψωΔΦΨΩΣΠ]{1,3}")

# 语气/结构助词：术语里出现基本就是切错边界。
_PARTICLE_CHARS = set("的了着过吗呢吧啊呀嘛哦嗯")

# 术语不该以这些字开头或结尾（虚词、代词、连词、副词）。
_EDGE_BAD_CHARS = set(
    "是在和与也就都很要会可能没不你我他她它这那其之而但并或把被让给对从到为以及若则即且又还才只更最再等个些"
    "有看说讲做想觉得然后所以因为但是如果就是说"
)

# 高频功能词 / 口头禅：既不是学科术语，也不该进 key_terms。
_STOPWORDS = {
    "我们", "你们", "他们", "大家", "自己", "这个", "那个", "这些", "那些", "这样", "那样",
    "什么", "怎么", "为什么", "如何", "可以", "就是", "然后", "所以", "因为", "但是",
    "如果", "现在", "这里", "那里", "时候", "东西", "问题", "方面", "情况", "部分",
    "一下", "一点", "一个", "一些", "一样", "一直", "一起", "一般", "一定", "必须",
    "应该", "需要", "知道", "看到", "看到", "各位", "同学", "老师", "课堂", "上课",
    "今天", "刚才", "接下来", "下面", "上面", "前面", "后面", "左边", "右边",
    "不是", "没有", "已经", "还是", "或者", "而且", "并且", "不过", "虽然", "即使",
    "第一", "第二", "第三", "首先", "其次", "最后", "再次", "另外", "此外",
    "注意", "记住", "务必", "千万", "重点", "考点", "考试", "必考", "复习", "作业",
    "掌握", "背诵", "要求", "内容", "方法", "过程", "结果", "东西", "样子",
    "这一", "那二", "二二", "三三", "这种", "那种", "有点", "比较", "非常", "特别",
    "可能", "应该", "能够", "觉得", "认为", "发现", "出现", "进行", "得到", "变成",
    "小时", "分钟", "秒钟", "今天", "明天", "昨天", "以后", "以前", "之后", "之前",
}


def _cjk_ngrams(text: str, *, nmin: int = 2, nmax: int = 6) -> Iterable[str]:
    """枚举文本中所有 2..6 字的 CJK 连续子串。"""
    for run in _CJK_RUN.findall(text or ""):
        length = len(run)
        if length < nmin:
            continue
        for n in range(nmin, min(nmax, length) + 1):
            for i in range(0, length - n + 1):
                yield run[i : i + n]


def _plausible_term(term: str) -> bool:
    """判断一个 n-gram 像不像学科术语（粗筛，宁缺毋滥）。"""
    if len(term) < 2:
        return False
    if term in _STOPWORDS:
        return False
    if term[0] in _EDGE_BAD_CHARS or term[-1] in _EDGE_BAD_CHARS:
        return False
    if any(ch in _PARTICLE_CHARS for ch in term):
        return False
    # 全是虚词组成（如「这也」「就是说」）直接丢
    if all(ch in _EDGE_BAD_CHARS for ch in term):
        return False
    return True


class TermIndex:
    """全课术语索引：既供 key_terms 使用，也供「重复出现」加分使用。

    doc 定义为「一句讲解」或「一行课件」，这样 `count` 就是「出现次数」，
    `pages` 就是「跨了几页」，两个信号和用户直觉一致。
    """

    def __init__(self) -> None:
        self.count: Dict[str, int] = {}
        self.pages: Dict[str, set] = defaultdict(set)
        self.first_t: Dict[str, float] = {}
        self.first_page: Dict[str, int] = {}
        self.terms: List[str] = []          # 按长度降序，供子串匹配
        self._by_len_desc: List[str] = []

    # -- 构建 ---------------------------------------------------------------
    @classmethod
    def build(
        cls,
        docs: Sequence[Tuple[int, str, float]],
        *,
        min_count: int = 2,
        max_terms: int = 400,
    ) -> "TermIndex":
        """从 [(页号, 文本, 时间)] 构建索引。

        规则：n-gram 至少出现 min_count 次；再做「嵌套剪枝」——如果某个短词总是
        被一个出现次数相当的更长词包住（电枢 ⊂ 电枢反应），只保留长词，
        这样术语表读起来才像人话。
        """
        idx = cls()
        counts: Counter = Counter()
        pages: Dict[str, set] = defaultdict(set)
        first: Dict[str, Tuple[float, int]] = {}

        for page, text, t in docs:
            seen = set()
            for gram in _cjk_ngrams(text):
                if not _plausible_term(gram):
                    continue
                seen.add(gram)
            for gram in seen:
                counts[gram] += 1
                pages[gram].add(page)
                cur = first.get(gram)
                if cur is None or t < cur[0]:
                    first[gram] = (t, page)

        # 拉丁/希腊字母术语（Ea、PWM、Φ 等），也按出现次数筛
        for page, text, t in docs:
            for m in _LATIN_TERM.finditer(text or ""):
                term = m.group(0)
                if len(term) < 2 or term.lower() in _STOPWORDS:
                    continue
                counts[term] += 0
            for m in _GREEK_TERM.finditer(text or ""):
                counts.setdefault(m.group(0), 0)

        # 嵌套剪枝：长的优先，短词若是某个已收词的子串且出现次数不更多，就丢掉
        accepted: List[str] = []
        for cand in sorted(counts, key=lambda w: (-len(w), -counts[w], w)):
            if counts[cand] < min_count:
                continue
            if len(cand) >= 2 and any(cand in longer for longer in accepted):
                continue
            accepted.append(cand)
            idx.count[cand] = counts[cand]
            idx.pages[cand] = pages.get(cand, set())
            idx.first_t[cand] = first.get(cand, (0.0, 1))[0]
            idx.first_page[cand] = first.get(cand, (0.0, 1))[1]

        accepted.sort(key=lambda w: (-idx.count[w], -len(w), w))
        idx.terms = accepted[:max_terms]
        idx._by_len_desc = sorted(idx.terms, key=len, reverse=True)
        return idx

    # -- 查询 ---------------------------------------------------------------
    def find(self, text: str) -> List[str]:
        """返回文本中出现的术语（长词优先，被长词覆盖的短词不重复返回）。"""
        norm = _match_norm(text)
        if not norm:
            return []
        found: List[str] = []
        for term in self._by_len_desc:
            if term in norm:
                if any(term in longer for longer in found):
                    continue
                found.append(term)
        return found

    def rank_for(self, text: str) -> List[str]:
        """某页文本里出现过的术语，按全课频次降序（给 key_terms 用）。"""
        found = set(self.find(text))
        return sorted(found, key=lambda w: (-self.count[w], -len(w), w))


# ==========================================================================
# 四、划重点打分（本模块的核心）
# ==========================================================================


def _scan_rules(
    text: str,
    rules: Sequence[Tuple[re.Pattern, int, str]],
    claimed: List[Tuple[int, int]],
) -> Tuple[int, List[str]]:
    """逐条规则扫描，命中的片段不重复计分，返回 (加分, 原因列表)。

    claimed 记录已经计过分的位置区间，避免「考试必考」里 考试 和 必考
    这种重叠片段被重复叠加；不重叠的不同信号则允许叠加（本来就该更高分）。
    """
    score = 0
    reasons: List[str] = []
    for pattern, weight, label in rules:
        m = pattern.search(text)
        if not m:
            continue
        span = m.span()
        if span[0] == span[1]:
            continue
        if any(not (span[1] <= a or span[0] >= b) for a, b in claimed):
            continue
        claimed.append(span)
        score += weight
        tok = m.group(0).strip()
        reasons.append(label.format(tok=tok) if "{tok}" in label else label)
    return score, reasons


def _overlap_with_slide(text: str, slide_terms: Sequence[str], slide_lines: Sequence[str]) -> Tuple[int, List[str]]:
    """讲解与课件的重合度：老师念课件原文 = 强调。

    两级判断：
      * 整行几乎被念出来（长行子串）-> +3
      * 念到了课件上的术语（>=2 个 -> +2，1 个 -> +1）
    """
    norm = _match_norm(text)
    if not norm:
        return 0, []

    squashed_lines = [_match_norm(line) for line in slide_lines]
    for line in squashed_lines:
        if len(line) >= 6 and line in norm:
            return 3, ["基本念出课件原文"]

    hits = [t for t in slide_terms if _match_norm(t) in norm]
    if len(hits) >= 2:
        return 2, [f"念到课件术语「{hits[0]}」等 {len(hits)} 个"]
    if hits:
        return 1, [f"念到课件术语「{hits[0]}」"]
    return 0, []


def _repetition_bonus(text: str, terms: TermIndex) -> Tuple[int, List[str]]:
    """同一概念反复出现 -> 加分。全课出现次数越多、跨页越多，越像重点。"""
    hits = terms.find(text)
    if not hits:
        return 0, []
    best = max(hits, key=lambda w: (terms.count[w], len(terms.pages[w])))
    count = terms.count[best]
    pages = len(terms.pages[best])
    if count >= 5 or pages >= 4:
        return 2, [f"「{best}」全课出现 {count} 次（跨 {pages} 页）"]
    if count >= 3:
        return 1, [f"「{best}」全课出现 {count} 次（跨 {pages} 页）"]
    return 0, []


def _has_number(text: str) -> Tuple[int, List[str]]:
    """含公式或数字要点。"""
    norm = _norm(text)
    claimed: List[Tuple[int, int]] = []
    return _scan_rules(norm, _FORMULA_RULES, claimed)


def score_sentence(
    text: str,
    *,
    terms: TermIndex,
    slide_terms: Sequence[str],
    slide_lines: Sequence[str],
    source: str = "speech",
) -> Tuple[int, List[str]]:
    """给一句话打重点分，返回 (分数, 中文原因列表)。

    source="speech" 时启用来全部信号（考试词 / 念课件 / 重复 / 定义 / 序号 / 公式）；
    source="slide" 时只看句子本身的结构信号（课件上写死的定义、公式、并列枚举）。
    """
    norm = _norm(text)
    claimed: List[Tuple[int, int]] = []
    score = 0
    reasons: List[str] = []

    for rules in (_EXAM_RULES, _DEF_RULES):
        s, r = _scan_rules(norm, rules, claimed)
        score += s
        reasons.extend(r)

    s, r = _scan_rules(norm, [_ENUM_RULE], claimed)
    score += s
    reasons.extend(r)

    s, r = _has_number(norm)
    # 数字/公式的区间已经单独算过，这里只取分数和原因
    score += s
    reasons.extend(r)

    if source == "speech":
        s, r = _overlap_with_slide(text, slide_terms, slide_lines)
        score += s
        reasons.extend(r)
        s, r = _repetition_bonus(text, terms)
        score += s
        reasons.extend(r)

    return min(score, _SCORE_MAX), reasons


def _speech_units(page: Dict[str, Any], segments: Sequence[Dict[str, Any]]) -> List[Tuple[str, float]]:
    """把一页的讲解拆成 [(句子, 估算时间)]。

    有原始 segment 时按段内字符比例估算句子时间戳（比整段取开头更准）；
    没有 transcript 时退化成「整页 speech 从页首开始」。
    """
    units: List[Tuple[str, float]] = []
    idxs = page.get("segments") or []
    if idxs:
        for si in idxs:
            if si < 0 or si >= len(segments):
                continue
            seg = segments[si]
            start = float(seg.get("start") or 0.0)
            end = float(seg.get("end") or start)
            raw = str(seg.get("text") or "")
            pieces = split_sentences(raw)
            total = max(1, len(raw))
            for sent, off in pieces:
                frac = off / total
                units.append((sent, start + (end - start) * frac))
        if units:
            return units

    t0 = float(page.get("t") or 0.0)
    for sent, off in split_sentences(str(page.get("speech") or "")):
        units.append((sent, t0))
    return units


# 纯口语连接词/语气词：这些字本身不携带课程信息，
# 一句话如果基本由它们构成，就只是在「铺垫」，不该被标成重点。
_FILLER_RE = re.compile(
    r"所以|然后|那么|这个|那个|就是说|也就说|呢|啊|吧|嘛|哦|嗯|呃|"
    r"大家|咱们|我们|你们|他们|可以|可能|就是|觉得|其实|当然|"
    r"一些|一下|这样|那样|方面|内容|东西|事情|时候|地方|"
    r"对|跟|和|把|被|这|那|是|的|了|在|有|就|都|也|还|会|要|说|讲"
)


def _has_substance(text: str, terms: "TermIndex", slide_terms: Sequence[str]) -> bool:
    """这句话有没有「课程信息量」，够不够格当重点。

    背景：只靠信号词打分会把大量口语铺垫选进重点——
        「所以某人说到呢」          -> 命中"所以"，但不是重点
        「所以也就说,对相关的一些内容啊」 -> 同上
    真正的重点句应该至少满足一条：
      1. 命中了课件术语 / 本页关键词
      2. 去掉口语词之后还剩下成规模的实词（>=6 字）
      3. 句子里有定义/结论的实质内容（是指/定义为/本质/包括…）
    """
    core = _match_norm(text)
    if not core:
        return False

    # 去掉口语连接词后还剩多少「实词」——这是判断有没有信息量的主尺子。
    residue = _FILLER_RE.sub("", core)

    # 1) 命中课件术语，且术语要在句子里占份量。
    #    只命中一个 3 字术语不够：「所以某人说到呢」也含人名，
    #    但那句话的实际内容是"所以…说到呢"，是铺垫不是重点。
    hit_len = 0
    for term in slide_terms:
        t = _match_norm(term)
        if len(t) >= 2 and t in core:
            hit_len = max(hit_len, len(t))
    if hit_len and (hit_len >= 4 or hit_len >= max(3, len(residue))):
        return True

    # 2) 去掉口语词后仍有成规模实词——但要看**密度**，不能只看绝对长度。
    #    反例：「所以也就说,对相关的一些内容啊,就是我在教学的过程当中呢,
    #    可能就是因为以前的课程跟这个不一样」残留实词不少，但全句就是铺垫。
    if len(residue) >= 6:
        if len(core) < 30:
            return True
        return len(residue) / len(core) >= 0.5

    # 3) 定义/结论的实质句式（需要真的有内容，不能只有连接词）
    if re.search(r"是指|定义为|指的是|本质|实质|包括|分为|等于|意味着|叫做|称为", core):
        return len(residue) >= 4

    return False


def _score_page_emphasis(
    page: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
    terms: TermIndex,
    *,
    min_score: int,
) -> List[Dict[str, Any]]:
    """为一页课件产出划重点条目（讲解优先，其次课件本体）。

    返回 [{text, score, t, reason, reasons}]，按分数降序，最多 _MAX_PER_PAGE 条。
    """
    slide_lines = [str(x) for x in (page.get("lines") or [])]
    slide_terms = [str(t) for t in (page.get("key_terms") or [])]
    title = str(page.get("title") or "")
    title_norm = _match_norm(title)

    items: List[Dict[str, Any]] = []

    # 1) 老师讲解：主要来源
    for sent, t in _speech_units(page, segments):
        core = _match_norm(sent)
        if len(core) < 6:          # 「对」「嗯」「好」这类语气句没有信息量
            continue
        score, reasons = score_sentence(
            sent, terms=terms, slide_terms=slide_terms, slide_lines=slide_lines, source="speech"
        )
        if score < min_score:
            continue
        # 光有信号词不够：口语铺垫句不该占重点名额
        if not _has_substance(sent, terms, slide_terms):
            continue
        items.append(
            {
                "text": util.truncate(sent, 160),
                "score": score,
                "t": round(float(t), 1),
                "reasons": reasons,
                "source": "speech",
            }
        )

    # 2) 课件本体：课件上写死的定义/公式/枚举也值得标（但不重复标标题行）
    for line in slide_lines:
        core = _match_norm(line)
        if len(core) < 6 or core == title_norm:
            continue
        score, reasons = score_sentence(
            line, terms=terms, slide_terms=slide_terms, slide_lines=[], source="slide"
        )
        if score < min_score:
            continue
        items.append(
            {
                "text": util.truncate(line, 160),
                "score": score,
                "t": round(float(page.get("t") or 0.0), 1),
                "reasons": reasons,
                "source": "slide",
            }
        )

    # 同一句话（同义重复）只留分最高的那条
    best: Dict[str, Dict[str, Any]] = {}
    for item in items:
        key = _match_norm(item["text"])
        cur = best.get(key)
        if cur is None or item["score"] > cur["score"]:
            best[key] = item
    merged = sorted(best.values(), key=lambda x: (-x["score"], x["t"]))[:_MAX_PER_PAGE]

    for item in merged:
        # 契约 §5 用 reason（字符串），§6 用 reasons（数组），两个都写：前者兼容冻结契约，
        # 后者给前端/检索用。
        item["reason"] = "；".join(item["reasons"])
    return merged


def build_global_emphasis(terms: TermIndex) -> List[Dict[str, Any]]:
    """跨页重复出现的高频概念 -> 全课级重点（契约 §5 global_emphasis）。

    打分：2 * 出现次数 + 跨页数 - 2，上限 10。出现 4 次跨 3 页 = 9 分，
    和契约示例一致。
    """
    out: List[Dict[str, Any]] = []
    for term in terms.terms:
        count = terms.count.get(term, 0)
        pages = len(terms.pages.get(term, ()))
        if count < 3 or pages < 2:
            continue
        score = min(_SCORE_MAX, 2 * count + pages - 2)
        out.append(
            {
                "text": term,
                "score": score,
                "t": round(float(terms.first_t.get(term, 0.0)), 1),
                "slide": int(terms.first_page.get(term, 1)),
                "reason": f"出现 {count} 次，跨 {pages} 页",
                "reasons": [f"全课出现 {count} 次", f"跨 {pages} 页重复"],
            }
        )
    out.sort(key=lambda x: (-x["score"], x["t"]))
    return out[:40]


# ==========================================================================
# 五、align：时间轴对齐 + 划重点
# ==========================================================================


def _load_frames(root: Path) -> Dict[str, Any]:
    """读 frames.json，缺失或空页时给出可操作的中文错误。"""
    frames = util.read_json(root / "frames.json")
    if not frames:
        raise CvnError(f"{root}/frames.json 不存在或为空：请先运行抽帧阶段（frames）")
    slides = frames.get("slides") or []
    if not slides:
        raise CvnError(f"{root}/frames.json 里没有任何课件页，无法生成笔记")
    return frames


def _normalise_pages(frames: Dict[str, Any], ocr: Dict[str, Any]) -> List[Dict[str, Any]]:
    """合并 frames.json（时间轴）与 slides.json（识别文本）成统一的页列表。"""
    duration = float(frames.get("duration") or 0.0)
    ocr_by_index = {int(s.get("index") or 0): s for s in (ocr.get("slides") or [])}

    raw = sorted(frames.get("slides") or [], key=lambda s: (float(s.get("t") or 0.0), int(s.get("index") or 0)))
    pages: List[Dict[str, Any]] = []
    for pos, frame in enumerate(raw):
        index = int(frame.get("index") or (pos + 1))
        t = float(frame.get("t") or 0.0)
        t_end = frame.get("t_end")
        if t_end is None:
            nxt = raw[pos + 1] if pos + 1 < len(raw) else None
            t_end = float(nxt.get("t") or duration) if nxt else duration
        t_end = float(t_end)
        if t_end <= t:                       # 坏数据兜底：至少当成一页
            t_end = t + 1.0

        rec = ocr_by_index.get(index) or {}
        lines = [util.clean_line(str(x)) for x in (rec.get("lines") or [])]
        lines = [x for x in lines if x and not util.is_watermark(x)]
        text = util.join_lines(lines) if lines else util.clean_line(str(rec.get("text") or ""))
        title = util.clean_line(str(rec.get("title") or ""))

        pages.append(
            {
                "index": index,
                "t": round(t, 2),
                "t_end": round(t_end, 2),
                "title": title,
                "lines": lines,
                "blocks": list(rec.get("blocks") or []),
                "text": text,
                "has_figure": bool(rec.get("has_figure")),
                "figure_note": util.clean_line(str(rec.get("figure_note") or "")),
                "ocr_conf": float(rec.get("ocr_conf") or 0.0),
                "kind": str(rec.get("kind") or "unknown"),
                "kind_reason": util.clean_line(str(rec.get("kind_reason") or "")),
                "low_conf_lines": [str(x) for x in (rec.get("low_conf_lines") or [])],
                "speech": "",
                "segments": [],
                "emphasis": [],
                "key_terms": [],
            }
        )
    return pages


# 艺术字标题识别失败时，产出的是"看起来像中文但错的字"，
# 字符比例上看不出异常（cjk 占比仍然是 1.0），
# 所以没法靠字符统计判断——只能靠**跟其它页比对的信号**。

# 一行里出现这些组合，基本可以断定 OCR 认错了：
_GARBLE_PATTERNS = (
    # 汉字里夹孤立拉丁字母："为z学芬"
    re.compile(r"[\u4e00-\u9fff][A-Za-z][\u4e00-\u9fff]"),
    # 数字+字母+汉字："202m食閉"
    re.compile(r"[0-9]{1,4}[A-Za-z][\u4e00-\u9fff]"),
    # 单字重复堆叠（识别串行的典型特征）："怎科天" 类
    re.compile(r"[\u4e00-\u9fff]{2}[\u4e00-\u9fff]{2}"),
)


def _row_is_garbled(text: str) -> bool:
    """判断一整行是不是"标题 + 页眉被识别串行"的产物。

    特征：一行里同时出现「很长的汉字串」和「结尾挂着时间戳」，
    说明 OCR 把标题和右上角的页眉连成了一行。
    """
    t = util.clean_line(text or "")
    if not t:
        return True
    if re.search(r"\d{1,2}:\d{2}(?::\d{2})?", t):
        # 有时间戳：看汉字部分是否成句（含标点分隔的短语）
        han = re.sub(r"[^\u4e00-\u9fff,，。;；:：%（）()]", "", t)
        # 成句的标志：汉字之间有正常标点，或汉字段较短
        if len(han) <= 24 or re.search(r"[，。、；]", t):
            return False
        return True
    return False


def _title_row_suspect(text: str) -> bool:
    """标题行是否可疑（但还没到"明显乱码"的程度）。

    艺术字标题识别失败后的典型产物：
      「某门课程的名称与课时16446:39」
      ——中文长串 + 数字 + 时间戳，混在一起，没有一个正常标题会这样。
    判据：汉字数很多（>12）却夹着数字或时间戳，且没有像样的句子标点。
    """
    t = util.clean_line(text or "")
    if not t:
        return True
    han = len(re.findall(r"[\u4e00-\u9fff]", t))
    has_clock = bool(re.search(r"\d{1,2}:\d{2}", t))
    # 句子标点说明是正常正文。注意**半角标点也要算**——
    # RapidOCR 常输出半角逗号（「本课程总学分为2学分,总学时为32课时」），
    # 只认全角会把完全正确的标题误判成可疑。
    sentence_like = bool(re.search(r"[，。、；,.;:：]", t))
    # 只有「挂着时间戳」才是强信号——艺术字识别失败时，
    # 页眉的时间戳会跟标题串成一行。纯数字（目录页的「1」「2」）不算。
    if has_clock and han >= 6:
        return True
    # 没有任何正常标点、汉字很多、还夹着数字串 → 典型的识别串行
    if han >= 14 and not sentence_like and re.search(r"[0-9]{2,}", t):
        return True
    return False


def _looks_like_garbled(text: str) -> bool:
    """判断一行文字是否像"识别失败的乱码"。

    只能识别**结构性**异常（汉字夹字母、数字串行等）。
    纯汉字但内容错的（"某段乱码"）看不出来，
    那种情况靠"跟同页/同课程其它页比对"在别处兜底。
    """
    t = util.clean_line(text or "")
    if not t:
        return True
    if _GARBLE_PATTERNS[0].search(t) or _GARBLE_PATTERNS[1].search(t):
        return True
    return False


def _fix_title(page: Dict[str, Any], terms: "TermIndex") -> None:
    """标题不可靠时，用这一页讲解里的术语兜底。"""
    title = util.clean_line(str(page.get("title") or ""))
    # 只要这一行里混进了乱码片段，整行就不可信——
    # 「某门课程的名称与课时」里虽然有「总学时为32课时」是对的，
    # 但整行挂在目录上只会让人困惑，不如标出来让人自己看图。
    # 判断顺序很重要：
    #   1) 先看 title 本身。title 可信 → 什么都不用做，直接用。
    #   2) title 不可信 → 再退到"整行原文"判断（title 被截断过，长度不够）。
    # 反过来做会把「首行是页眉、标题在第二行」的好页面误伤。
    if title and not _looks_like_garbled(title) and not _title_row_suspect(title):
        return

    raw = util.clean_line(str((page.get("lines") or [""])[0]))
    if title and not _looks_like_garbled(title) and not _row_is_garbled(raw) \
            and not _title_row_suspect(raw):
        return
    has_garble = True
    # 兜底：取这一页讲稿里出现最多的、且不在页眉里的术语
    speech = str(page.get("speech") or "")
    if not speech:
        return
    ranked = terms.rank_for(speech) or []
    for term in ranked:
        term = util.clean_line(str(term))
        if len(term) >= 3 and not _looks_like_garbled(term):
            page["title_source"] = "讲解术语（课件标题未认准）"
            page["title"] = term
            return
    page["title_source"] = "未识别"


def _stages_related(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """两页内容是否真的相关（用于动画阶段合并的最后一道闸）。"""
    la = [_match_norm(x) for x in (a.get("lines") or []) if str(x).strip()]
    lb = [_match_norm(x) for x in (b.get("lines") or []) if str(x).strip()]
    if not la or not lb:
        return True                      # 薄页（只有标题）直接合
    joined_a, joined_b = " ".join(la), " ".join(lb)
    hit_a = sum(1 for x in la if x and x in joined_b)
    hit_b = sum(1 for x in lb if x and x in joined_a)
    overlap = max(hit_a / len(la), hit_b / len(lb))
    if overlap >= 0.3:
        return True
    # 两页都很薄（各 1~2 行）且标题一致 —— 是同一页的不同阶段
    if len(la) <= 2 and len(lb) <= 2:
        return True
    # 短页与长页、且有字重合
    if (len(la) <= 3 or len(lb) <= 3) and overlap > 0:
        return True
    return False


def merge_animation_stages(pages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """把「同一张幻灯片的逐条动画」合并成一页。

    背景：PPT 的逐条动画在抽帧阶段看起来像翻页——画面在变，其实同一页在"演"。
    实测 62 分钟录屏里，同一张幻灯片被切成 3~4 页，前几页只有部分文字，
    笔记就会出现好几段重复内容。

    判据：标题一致（或内容高度重合）+ 时间接近 + 内容相关。
    合并时保留文字最完整的那一帧，时间区间取并集。

    实现要点：拿「本组内任意一个原始阶段」去比对新页，而不是只比合并结果。
    否则会漏——合并后标题可能变成第一个阶段的标题，再跟后面阶段就比不上。
    """
    if not pages:
        return []

    out: List[Dict[str, Any]] = []
    group: List[Dict[str, Any]] = []      # 当前组的全部原始阶段

    for page in pages:
        if not out:
            out.append(page)
            group = [page]
            continue

        # 用组内任一阶段来判断"是不是同一张幻灯片的下一个阶段"
        matched = any(_same_slide_stage(m, page) for m in group)
        # 再确认内容确实相关（防止隔着长时间把不同页合了）
        related = any(_stages_related(m, page) for m in group)

        if not (matched and related):
            out.append(page)
            group = [page]
            continue

        # 合并：取文字最全的一页做底，时间取并集
        current = out[-1]
        richer = page if _text_len(page) >= _text_len(current) else current
        merged = dict(richer)
        merged["t"] = min(current["t"], page["t"])
        merged["t_end"] = max(current["t_end"], page["t_end"])
        merged["frames_merged"] = int(current.get("frames_merged", 1)) + int(
            page.get("frames_merged", 1)
        )
        src = list(current.get("merged_from") or [current.get("index")])
        src += list(page.get("merged_from") or [page.get("index")])
        merged["merged_from"] = sorted({int(x) for x in src if x})
        out[-1] = merged
        group.append(page)

    return out


def _text_len(page: Dict[str, Any]) -> int:
    return sum(len(str(x)) for x in (page.get("lines") or []))


def _same_slide_stage(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """判断 b 是不是 a 的下一个动画阶段（同一张幻灯片）。

    PPT 的"逐条动画"有两种表现，都要能合并：
      累加式：后一页包含前一页全部内容，再添几条
      替换式：后一页把前一页某几句换掉（先显示"自然属性"，
              再切到"社会属性"），内容不是包含关系

    所以判据是「标题相同 + 时间上紧邻 + 内容有相当比例重合」，
    而不是"必须包含"。标题相同 + 紧邻已经是很强的信号——
    同一张幻灯片连续讲两个不同话题的情况本就少见，
    即便误合并，也只是把同一页的两个阶段并成一页，损失很小。
    """
    ta = _match_norm(str(a.get("title") or ""))
    tb = _match_norm(str(b.get("title") or ""))
    if not ta or not tb:
        return False

    # 标题判定：完全一致 / 互相包含 / 字重合度高 都算同一个标题。
    # 注意还有一类情况：同一张幻灯片在不同阶段被 VLM 读出了不同标题
    # （「人之所以为人，究竟区别于动物的是什么」vs「人与动物最大的区别是什么」），
    # 这两句字面不同但语义是同一个提问，所以下面还有"内容重合"这条兜底。
    same_title = (
        ta == tb
        or ta in tb
        or tb in ta
        or _char_overlap(ta, tb) >= 0.8
    )
    if not same_title:
        # 标题不同时，退回看内容：同一页的动画阶段内容会高度重合
        la0 = [_match_norm(x) for x in (a.get("lines") or []) if str(x).strip()]
        lb0 = [_match_norm(x) for x in (b.get("lines") or []) if str(x).strip()]
        if la0 and lb0:
            joined_b0 = " ".join(lb0)
            joined_a0 = " ".join(la0)
            hit0 = max(
                sum(1 for x in la0 if x and x in joined_b0) / len(la0),
                sum(1 for x in lb0 if x and x in joined_a0) / len(lb0),
            )
            if hit0 >= 0.6:
                same_title = True
        if not same_title:
            return False

    # 时间上要足够接近：中间夹了很长的别的画面就不该合并。
    # 注意不能要求"严格紧邻"——中间可能隔着桌面/浏览器页（被判为跳过），
    # 于是 t_end 和下一页 t 之间会出现大段空隙，但仍然是同一张幻灯片。
    gap = float(b.get("t") or 0.0) - float(a.get("t_end") or 0.0)
    if gap > 600.0:
        return False

    la = [_match_norm(x) for x in (a.get("lines") or []) if str(x).strip()]
    lb = [_match_norm(x) for x in (b.get("lines") or []) if str(x).strip()]
    if not la or not lb:
        return True          # 一方没文字：靠标题+紧邻判断

    # 内容有相当比例重合（累加式会 100% 命中；替换式也常有公共行）
    joined_b = " ".join(lb)
    joined_a = " ".join(la)
    hit_a = sum(1 for line in la if line and line in joined_b)
    hit_b = sum(1 for line in lb if line and line in joined_a)
    best = max(hit_a / len(la), hit_b / len(lb))
    if best >= 0.5:
        return True
    # 一行都对不上，但标题相同且紧邻：仍可能是同一页的不同动画阶段
    # （替换式动画可能把整段文字都换了）
    return False


def _char_overlap(a: str, b: str) -> float:
    """两段文字的字重合度（多重集 Dice），0~1。"""
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    return 2.0 * inter / (len(a) + len(b))


def _assign_segments(
    pages: Sequence[Dict[str, Any]], segments: Sequence[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """把每条字幕落到页码，并回填 segment['slide']。

    规则（契约 §4/§5）：slide.t <= start < slide.t_end；早于第一页的归第 1 页；
    超出最后一页的归最后一页。用二分查找，1000+ 条字幕也很快。
    """
    if not pages:
        return []
    starts = [p["t"] for p in pages]
    last = len(pages) - 1

    for seg in segments:
        start = float(seg.get("start") or 0.0)
        pos = bisect.bisect_right(starts, start) - 1
        if pos < 0:
            pos = 0
        elif pos > last:
            pos = last
        # 页与页之间若留了空隙（抽帧去重造成），沿用前一页（内容还在屏上）
        page = pages[pos]
        if start >= page["t_end"] and pos + 1 <= last and start < pages[pos + 1]["t"]:
            pass
        if page.get("kind") == "slide":
            seg["slide"] = int(page["index"])
        else:
            # 这条字幕发生在桌面/浏览器等非课件画面上，不该算作某一页的讲解
            seg["slide"] = 0
    return list(segments)


def _join_speech(texts: Sequence[str]) -> str:
    """把多条字幕拼成一段讲解文本：缺标点的地方补句号，保证能按句切分。"""
    buf = ""
    for raw in texts:
        piece = util.clean_line(str(raw or ""))
        if not piece:
            continue
        if not buf:
            buf = piece
            continue
        if buf[-1] in "。！？!?；;，,、":
            buf += piece
        else:
            buf += "。" + piece
    return buf


def align(out_dir: str | Path, *, min_score: int = 3) -> Dict[str, Any]:
    """对齐三个上游产物并完成划重点打分，写 out_dir/aligned.json（契约 §5）。

    步骤：
      1. frames.json 提供页码与时间区间，slides.json 提供每页文本；
      2. transcript.json 的每条字幕按 start 落进页区间，回填 segment["slide"]；
      3. 每页 speech = 该页字幕按句拼接，segments = 该页字幕下标；
      4. 逐句划重点（信号词 / 念课件 / 重复 / 定义结论 / 序号 / 公式），
         每条给出 score 与 reasons；跨页重复的概念汇总成 global_emphasis；
      5. key_terms 取全课术语索引里在本页出现过的词，按频次排序。

    参数
    ----
    out_dir : 运行目录
    min_score : 重点阈值，默认 3（低于该分的句子不进 emphasis）

    返回
    ----
    摘要 dict：{"aligned": 路径, "slides": 页数, "segments": 字幕数,
              "assigned": 归属成功数, "emphasis": 重点条数,
              "global_emphasis": 全局重点条数}
    """
    root = Path(out_dir)
    if not root.exists():
        raise CvnError(f"目录不存在：{root}")

    frames = _load_frames(root)
    ocr = util.read_json(root / "slides.json", default={}) or {}
    transcript = util.read_json(root / "transcript.json", default={}) or {}
    segments = list(transcript.get("segments") or [])

    all_pages = _normalise_pages(frames, ocr)
    _assign_segments(all_pages, segments)

    # 只把「真课件页」写进笔记。
    # 录屏类视频里桌面/浏览器/PPT 编辑界面会占很大比重，这些画面不是课件，
    # 但它们在时间轴上是有效的——所以先用全量页做时间对齐，再筛页面。
    # 逐条动画会被抽帧当成多页，先合并回一张幻灯片
    content_pages = merge_animation_stages(
        [p for p in all_pages if p.get("kind") == "slide"]
    )
    skipped = [
        {
            "index": p["index"],
            "t": p["t"],
            "t_end": p["t_end"],
            "kind": p.get("kind", "unknown"),
            "reason": p.get("kind_reason", ""),
        }
        for p in all_pages
        if p.get("kind") != "slide"
    ]
    if not content_pages:
        # 一页都没识别出来时不能直接放弃：可能是分类器在这段素材上失灵了。
        util.warn("没有识别出任何「课件页」，改为把所有页面都当课件处理。"
                  "如需排除桌面/浏览器画面，请检查 slides.json 的 kind 字段。")
        content_pages = all_pages
        skipped = []
    pages = content_pages

    # 合并动画阶段后页区间变大了，字幕归属必须重算——
    # 否则原来落在"第 2 阶段"的字幕仍记着旧页码，合并后的页会丢讲解。
    if len(pages) != len([p for p in all_pages if p.get("kind") == "slide"]):
        for seg in segments:
            seg["slide"] = 0
        _assign_segments(pages, segments)

    # --- 术语索引：讲过的每一句话 + 课件上的每一行，都是一个 doc ---
    docs: List[Tuple[int, str, float]] = []
    for page in pages:
        for line in page["lines"]:
            docs.append((page["index"], line, page["t"]))
        for sent, t in _speech_units(page, segments):
            docs.append((page["index"], sent, t))
    terms = TermIndex.build(docs)

    # --- 每页 key_terms ---
    for page in pages:
        blob = "\n".join(page["lines"]) + "\n" + page["speech"]
        page["key_terms"] = terms.rank_for(blob)[:8]

    # --- 每页 speech / segments / emphasis ---
    for page in pages:
        idxs = [i for i, seg in enumerate(segments) if seg.get("slide") == page["index"]]
        page["segments"] = idxs
        page["speech"] = _join_speech([str(segments[i].get("text") or "") for i in idxs])
        page["emphasis"] = _score_page_emphasis(page, segments, terms, min_score=min_score)

    # --- 标题兜底 ---
    # 课件标题常用艺术字，OCR 认不准时会得到一串乱码。
    # 与其把乱码挂到目录上，不如用这一页老师讲得最多的术语当标题。
    # 必须放在 speech 算完之后——兜底要靠讲稿里的词。
    for page in pages:
        _fix_title(page, terms)

    global_emphasis = build_global_emphasis(terms)

    aligned = {
        "slides": pages,
        "global_emphasis": global_emphasis,
        "skipped_pages": skipped,
        "total_pages": len(all_pages),
    }
    path = util.write_json(root / "aligned.json", aligned)

    # 契约 §4 的注释要求 align 回填 segment["slide"]，写回 transcript.json
    if segments:
        transcript["segments"] = segments
        util.write_json(root / "transcript.json", transcript)

    _mark_manifest(root, "align")

    assigned = sum(1 for seg in segments if int(seg.get("slide") or 0) > 0)
    n_emp = sum(len(p["emphasis"]) for p in pages)
    util.vlog(
        f"align: {len(pages)} 页 / {len(segments)} 条字幕 / {n_emp} 条重点 / "
        f"{len(global_emphasis)} 个全局重点"
    )
    return {
        "aligned": str(path),
        "slides": len(pages),
        "segments": len(segments),
        "assigned": assigned,
        "emphasis": n_emp,
        "global_emphasis": len(global_emphasis),
    }


def _mark_manifest(root: Path, stage: str) -> None:
    """在已存在的 manifest.json 里标记阶段完成（不改动其它字段）。

    manifest.json 由编排层（course_kit.py）创建；这里只在文件已存在时补一条
    阶段完成记录，让「merge 之前必须有 align」这条契约硬前置能成立。
    """
    path = root / "manifest.json"
    manifest = util.read_json(path, default=None)
    if not isinstance(manifest, dict):
        return
    stages = manifest.setdefault("stages", {})
    if not isinstance(stages, dict):
        return
    entry = stages.setdefault(stage, {})
    if isinstance(entry, dict):
        entry["done"] = True
        entry["at"] = util.stamp()
    try:
        util.write_json(path, manifest)
    except OSError as exc:  # pragma: no cover - 只影响进度标记，不影响笔记
        util.vlog(f"manifest.json 写入失败（不影响笔记）：{exc}")


# ==========================================================================
# 六、知识点构建（便于二次加工）
# ==========================================================================

_KIND_ORDER = ["公式", "辨析", "案例", "步骤", "结论", "概念"]

_CHAPTER_RE = re.compile(r"^\s*(第\s*[0-9一二三四五六七八九十百]+\s*[章节讲部篇课]|chapter\s*\d+|part\s*\d+|专题|模块)", re.I)
_NUM_PREFIX_RE = re.compile(
    r"^\s*(?:[\(（]?\s*\d{1,2}\s*[\.\)）]|[一二三四五六七八九十]+\s*[、\.]|[①-⑳]|[-·•*])\s*"
)
_CHAPTER_NUM_PREFIX_RE = re.compile(r"^\s*\d+(?:\.\d+)+\s*")

_DEF_PATTERNS = [
    re.compile(r"^(?P<a>.{2,20}?)\s*(?:是指|指的是|定义为|叫做|称为|即)(?P<b>.{4,200})$"),
    # 「不是」里的「是」不是定义句：人的本质不是单个人…… 不能切成「人的本质不」
    re.compile(r"^(?P<a>.{2,20}?)\s*(?<![不不就])(?:是|表示|意味着)(?P<b>.{4,200})$"),
]

_ERROR_CUES = re.compile(
    r"易错|容易错|很容易|容易搞混|容易弄错|混淆|别搞错|不要混|误区|错误|弄错|搞混|不等于|千万不"
    r"|注意.{0,6}不|容易忽略"
)


def _strip_numbering(text: str) -> str:
    """去掉行首序号（1. / 一、 / ① / •），让知识点名更干净。"""
    s = _NUM_PREFIX_RE.sub("", text.strip())
    s = s.strip(" ：:·-—")
    return s


def _clean_name(text: str) -> str:
    """从一行课件里提炼知识点名称：去章节编号与序号，截掉冒号后的解释性尾巴。"""
    s = util.clean_line(text)
    # 必须先剥「4.1」再剥「4.」 —— 否则 "4.1 xxx" 会被当成序号 4. 而剩成 "1 xxx"
    s = _CHAPTER_NUM_PREFIX_RE.sub("", s)
    s = re.sub(r"^\s*\d{1,2}\s+", "", s)        # "5 机械特性" -> "机械特性"
    s = _strip_numbering(s)
    if not s:
        return ""
    # 「期末考试(占比50%)」→「期末考试」；括号里的比例不是名字的一部分
    s = re.sub(r"\s*[（(][^）)]{0,24}[）)]\s*$", "", s)
    s = re.sub(r"\s*[%％]\s*$", "", s)
    if not s:
        return ""
    # 「定义：XXX」这种行，取冒号前的部分当名称
    m = re.match(r"^(?P<head>.{2,24}?)[：:](?P<tail>.+)$", s)
    if m and len(_match_norm(m.group("tail"))) >= 4:
        s = m.group("head")
    s = s.strip(" ：:。；;、,.")
    return s


# 课件上这些行是版块标签而不是知识点，抽知识点时跳过。
# 两字的通用物理量/部件名：课件里出现频率极高，但单独成条对复习没帮助
# （真正要记的是「交轴电枢反应使磁场畸变」这种带谓词的结论）。
_TOO_GENERIC_2CHAR = {
    "磁场", "电流", "电压", "电枢", "磁通", "磁路", "火花", "电刷", "导体", "线圈",
    "绕组", "转速", "转矩", "效率", "功率", "负载", "空载", "励磁", "主极", "气隙",
    "交流", "直流", "电路", "电阻", "电容", "电感", "元件", "支路", "图", "表",
    "电动机", "发电机", "变压器", "电源", "磁动势", "电动势", "电枢反应",
}

_SKIP_LABEL_NAMES = {
    "本章重点", "本章小结", "本章要点", "小结", "总结", "目录", "例题", "作业",
    "思考题", "参考文献", "复习", "练习", "前言", "引言", "谢谢", "谢谢观看",
    # 这些是课件上的「标签」，本身不是知识点：真正的知识点是标签后面那句话
    "定义", "公式", "结论", "定理", "定律", "定理公式", "重要结论", "基本概念",
    "方法", "原因", "说明", "注意", "特点", "作用", "原理", "概念", "举例",
    # 课堂流程 / 版块标签：每页都有，但不是可复习的知识
    "提出问题", "解决问题", "课堂讨论", "小组讨论", "导入", "案例", "案例导入",
    "思考", "本节", "本节小结", "课堂练习", "讨论", "互动", "讲授", "主要内容",
    "学习目标", "教学要求", "课程性质", "课程目标", "考核方式", "成绩构成",
}

# 课程事务性信息（学分、学时、考试形式、作业、考勤……）：属于 course.md 的
# 「课程信息」，不是知识点。混进 knowledge.md 会把概念清单稀释成通知栏。
_ADMIN_LINE_RE = re.compile(
    r"学分|学时|课时|考核|考试|笔试|开卷|闭卷|成绩|作业|考勤|点名|签到"
    r"|小组讨论|课堂表现|第\s*\d+\s*周|分/次|占比|教学安排|课程安排|上课时间|选课"
    r"|选题范围|授课教师|教师|主讲|联系方式|电话|邮箱"
)

# 讲解里冒出来的「疑似术语」只要含这些字，基本就是半句话而不是术语。
_SPEECH_NAME_BAD_CHARS = set("的了着过吗呢吧啊呀嘛哦嗯是我你")

# 名称不该以这些字开头（代词/指示词）。
_BAD_NAME_HEAD = set("这那什怎哪我你他她它每各在对其此该")

# 口语里这些词开头的一定是半句话/连接词，不是术语
_SPEECH_BAD_PREFIX = (
    "虽然", "但是", "因为", "所以", "如果", "并且", "而且", "当然", "第一", "第二",
    "第三", "有些", "有的", "也不", "就是", "这个", "那个", "咱们", "大家", "我们",
    "你们", "他们", "或者说", "其实", "然后", "另外", "比如", "例如", "所谓", "还要",
)

# 术语中间出现这些字，基本说明这是从一句话里切出来的片段而不是术语：
# 学科术语（电枢反应 / 换向器 / 磁动势 / 电磁转矩）几乎不含这些虚词。
# 注意：不要收录「向/与/和/或/的」——它们是学科术语的正常组成字
# （换向、换向与火花、机械特性与电枢反应），收了会把真术语也毙掉。
_CLAUSE_CHARS = set("在是了也就都才又还只使将把被让给对从个种")

# 例外：这些是正经学科前缀，虽然首字本身是虚词/代词，但组合起来是术语
# （他励直流电动机、自励发电机、永磁同步电机）。
_GOOD_HEAD_COMPOUNDS = ("他励", "自励", "并励", "串励", "复励", "永磁", "它励")

# 数量词开头（「一台直流电机」「三种方式」）是把句子当术语了。
_CLASSIFIER_HEAD_RE = re.compile(r"^[一两三四五六七八九十几数多]\s*[台个种条项次只组步类些]")


def _is_good_name(name: str, *, from_speech: bool) -> bool:
    """判断候选名称能不能当知识点：挡住半句话、口头禅、OCR 碎片。

    from_speech=True 时按「老师口语」标准收紧（口语句子长、带虚词），
    因为讲解里能凑出「这一章」「我强调一遍」这类假术语。
    """
    s = util.clean_line(name).strip(" ：:。；;、,.")
    if not s:
        return False
    core = _match_norm(s)
    if len(core) < 2 or len(core) > 24:
        return False
    if core in _STOPWORDS or s in _SKIP_LABEL_NAMES:
        return False
    if s in _TOO_GENERIC_2CHAR:
        return False
    if util.is_watermark(s):
        return False
    if re.search(r"[。！？!?；;，,、（）()【】\[\]%:：/\\·=＝+＋]", s):  # 带句读/运算符的是一整句话或界面文本
        return False
    if core.isdigit():
        return False
    # 图表里四个两字标签横排（「尊重责任边界共情」）会被 OCR 连成一行，
    # 它不是术语；但如果结尾是学科名词（质量守恒定律），要留下。
    if re.fullmatch(r"(?:[\u4e00-\u9fff]{2}){3,}", s) and not re.search(
        r"(定律|定理|公式|方程|反应|效应|结构|系统|模型|方法|理论|关系|发展|教育"
        r"|人格|知识|能力|文化|社会|精神|价值|素养)$",
        s,
    ):
        return False
    # 名称中间的引号说明是从整句里切出来的（「可以为成人矣”日」）；
    # 首尾成对的引号是合法的（「“某概念”」）。
    inner = s[1:-1] if len(s) >= 2 and s[0] in "“\"'" and s[-1] in "”\"'" else s
    if re.search(r"[“”\"']", inner):
        return False
    if from_speech and s.startswith(_SPEECH_BAD_PREFIX):
        return False
    head_exempt = s.startswith(_GOOD_HEAD_COMPOUNDS)
    if not head_exempt and (s[0] in _BAD_NAME_HEAD or s[0] in _SPEECH_NAME_BAD_CHARS):
        return False
    if core[-1] in set("的了着过吗呢吧啊呀嘛"):
        return False
    # 句首数量词 / 中间夹虚词：都是从整句里切出来的碎片，不是术语
    if _CLASSIFIER_HEAD_RE.match(s):
        return False
    if s[0] in set("求看说讲做想计算证明假设如果由此"):
        return False
    if any(ch in _CLAUSE_CHARS for ch in s[1:]):
        return False
    # 「的」只允许出现在名词性后缀里：换向器的作用 / 人的本质 / 大学生的成长
    if "的" in s[1:]:
        if not re.search(
            r"的(作用|定义|含义|概念|方法|特点|原理|公式|分类|区别|意义|影响|条件|计算"
            r"|本质|属性|发展|关系|价值|内涵|特征|地位|目标|路径|层次|结构|类型|阶段"
            r"|表现|要求|原则|内容|问题|原因)$",
            s,
        ):
            return False
    if from_speech:
        if len(core) > 14:                          # 口语里超过 14 字的「术语」多半是半句话
            return False
        body = s[2:] if head_exempt else s
        if any(ch in _SPEECH_NAME_BAD_CHARS for ch in body):
            return False
        if re.search(r"而且|然后|但是|因为|所以|如果|就是|一下|一遍|一个|这个|那个|什么", s):
            return False
    return True


def _is_good_title(s: str) -> bool:
    """页标题能否当知识点：标题比正文行宽松（允许「的」「，」「？」）。"""
    s = util.clean_line(s).strip(" ：:。；;、,.")
    if not (3 <= len(s) <= 30):
        return False
    if _is_structural(s) or _ADMIN_LINE_RE.search(s):
        return False
    if re.search(r"[。！!；;]", s):
        return False
    if s[0] in _BAD_NAME_HEAD or s[0] in _SPEECH_NAME_BAD_CHARS:
        return False
    if util.is_watermark(s):
        return False
    return True


def _name_from_line(line: str, terms: Optional["TermIndex"] = None) -> Tuple[str, str]:
    """从一行课件文本推出 (知识点名, 一句话说明)；推不出来返回 ("", "")。

    优先级（越靠前越可信，宁可少而准）：
      1. 「名称：说明」「名称是指说明」这类显式结构 —— 直接拆；
      2. 行首就是已识别的全课术语（"机械特性方程 n = ..."）—— 用术语做名，整行做说明；
      3. 短行（<= 14 字）整行就是一个知识点名。

    长句、又没结构、行首也不是术语的，不当知识点：那是课件上的一句完整论述，
    硬截成短名字只会截出「这一章」「最后两条主线一定要记住」这种半句话。
    """
    raw = util.clean_line(line)
    if not raw:
        return "", ""
    name, desc = _split_desc(raw)
    if name and _is_good_name(name, from_speech=False):
        return name, desc

    # 人文类课的经典句式：「A，由B行」「某事物的本质不是……」
    # 「学习使人成为人」——术语在句首，后面就是解释，不需要冒号。
    m = re.match(r"^(?P<a>[\u4e00-\u9fff]{2,4})[，,](?P<b>.{6,80})$", raw)
    if m and re.search(r"[也矣乎者兮]", raw):
        a = _clean_name(m.group("a"))
        if _is_good_name(a, from_speech=False):
            return a, util.truncate(raw, 120)
    m = re.match(r"^(?P<a>[\u4e00-\u9fff]{2,10})不是(?P<b>.{6,80})$", raw)
    if m:
        a = _clean_name(m.group("a"))
        if _is_good_name(a, from_speech=False):
            # 保留「不是」：只取 b 会把否定定义翻成肯定，正好说反
            return a, util.truncate(raw, 120)
    m = re.match(r"^(?P<a>[\u4e00-\u9fff]{2,8})(?:使|让)(?P<b>.{6,80})$", raw)
    if m:
        a = _clean_name(m.group("a"))
        if _is_good_name(a, from_speech=False) and a not in _TOO_GENERIC_2CHAR:
            return a, util.truncate(raw, 120)

    norm_raw = _match_norm(raw)
    if terms is not None:
        for term in terms.find(raw):
            core = _match_norm(term)
            if len(core) < 3 or core == norm_raw:
                continue
            # 只有「行首即术语」才算这条行的主题，句中出现的不算
            if norm_raw.startswith(core) and _is_good_name(term, from_speech=False):
                return term, util.truncate(raw, 120)

    name = _clean_name(raw)
    if name and _is_good_name(name, from_speech=False) and len(_match_norm(name)) <= 14:
        return name, ""
    return "", ""


def _split_desc(line: str) -> Tuple[str, str]:
    """把「术语：说明」或「术语是指说明」拆成 (名称, 说明)。没拆出来就返回 ("", "")。"""
    s = _CHAPTER_NUM_PREFIX_RE.sub("", util.clean_line(line))
    s = _strip_numbering(s)
    for pattern in _DEF_PATTERNS:
        m = pattern.match(s)
        if m:
            a = m.group("a").strip(" ：:，,")
            b = m.group("b").strip(" 。；;，,")
            if len(_match_norm(a)) >= 2 and len(_match_norm(b)) >= 4:
                return _clean_name(a), util.truncate(b, 120)
    m = re.match(r"^(?P<head>.{2,24}?)[：:](?P<tail>.{4,})$", s)
    if m:
        head = _clean_name(m.group("head"))
        tail = m.group("tail").strip(" 。；;，,")
        if head and _match_norm(tail) != _match_norm(head):
            return head, util.truncate(tail, 120)
    return "", ""


def _title_desc(
    title: str,
    page: Dict[str, Any],
    segments: Sequence[Dict[str, Any]],
) -> str:
    """给页标题找一句最像「在解释它」的话（先讲解，后课件）。

    人文类课的标题常是问句（「某问题是什么」），老师不会逐字重复标题，
    所以用标题的二字片段做匹配，再给「最重要/关键/就是/是指/本质/立德」加分。
    找不到就返回空串，由调用方走指路说明，绝不硬编。
    """
    norm = _match_norm(title)
    grams = {
        norm[i : i + 2]
        for i in range(max(0, len(norm) - 1))
        if norm[i : i + 2]
        not in {"什么", "怎样", "如何", "意味", "着什", "的人", "样的", "为怎", "学意", "味着"}
    }
    if not grams:
        return ""

    best, best_score = "", 0.0
    # 注意：page["segments"] 存的是「本页字幕在全量列表里的下标」，
    # 而 _speech_units 需要的是全量 segment 字典列表——两者别搞混。
    for sent, _t in _speech_units(page, segments):
        clean = util.clean_line(sent)
        if len(clean) < 8 or _match_norm(clean) == norm:
            continue
        score = sum(1 for g in grams if g in _match_norm(clean))
        if not score:
            continue
        if re.search(r"最重要|关键|就是|是指|意味着|本质|包括|定义|立德|成人", clean):
            score += 2
        if len(clean) > 80:
            score -= 1
        if score > best_score:
            best, best_score = clean, score
    if best and best_score >= 3:
        return util.truncate(best, 120)

    best, best_score = "", 0.0
    for line in page.get("lines") or []:
        clean = util.clean_line(line)
        if _near_title(clean, title):
            continue
        if not (6 <= len(clean) <= 80):
            continue
        if _ADMIN_LINE_RE.search(clean) or util.is_watermark(clean):
            continue
        score = sum(1 for g in grams if g in _match_norm(clean))
        if re.search(r"更重要|不仅|而是|就是|是指|意味着|本质|包括|即|为", clean):
            score += 2
        if len(clean) > 40:
            score -= 2
        if score > best_score:
            best, best_score = clean, score
    return util.truncate(best, 120)


def _find_desc_in_speech(name: str, page: Dict[str, Any], segments: Sequence[Dict[str, Any]]) -> str:
    """在讲解里为知识点找一句「像在解释它」的话；找不到返回空串（不硬编）。

    宁可返回空串、标进「待补充」，也不要随便抓一句含有该词的话当说明——
    用户看到错的解释，比看到空白更糟。
    """
    norm_name = _match_norm(name)
    if norm_name:
        for sent, _t in _speech_units(page, segments):
            core = _match_norm(sent)
            if norm_name not in core or len(core) < 8:
                continue
            if not re.search(r"是指|指的是|定义为|叫做|称为|即为|意思是|也就是|相当于|用来|表示|就是", sent):
                continue
            # 术语出现在句首附近才更像「在给这个术语下定义」
            if core.find(norm_name) > max(6, len(core) // 3):
                continue
            return util.truncate(sent, 120)
    # 退化路径：课件行本身带说明结构（"换向器的作用：把交流变直流"）
    for line in page.get("lines") or []:
        head, desc = _split_desc(line)
        if head and desc and _match_norm(head) == norm_name:
            return desc
    return ""


def _find_pitfall(name: str, pages: Sequence[Dict[str, Any]], segments: Sequence[Dict[str, Any]]) -> str:
    """在讲解里找和该知识点相关的「易错点」原话；没有就留空，不编。"""
    norm_name = _match_norm(name)
    if not norm_name or len(norm_name) < 2:
        return ""
    for page in pages:
        for sent, _t in _speech_units(page, segments):
            core = _match_norm(sent)
            if norm_name not in core:
                continue
            if not _ERROR_CUES.search(sent):
                continue
            return util.truncate(sent, 100)
    return ""


# 章节/版块标记：这些行只用于给知识点分组，绝不单独成条。
_CHAPTER_MARK_RE = re.compile(
    r"^\s*(第\s*[0-9一二三四五六七八九十百]+\s*[章节讲部篇课]"
    r"|chapter\s*\d+|part\s*\d+|专题\s*\d*|模块\s*\d*)\s*$",
    re.I,
)
# 单独的版块标签（「本章小结」「绪论」等）也不是知识点。
_LABEL_ONLY = {
    "本章小结", "本章重点", "本章要点", "本章难点", "小结", "总结", "绪论", "绪论课",
    "目录", "习题", "作业", "思考题", "练习", "复习", "前言", "引言", "主要内容",
    "学习目标", "教学要求", "本章内容", "开场", "谢谢观看", "谢谢", "下课", "再见",
}


def _is_structural(name: str) -> bool:
    """判断一个名称是不是「章标题 / 版块标签」这类结构行（不该进知识点清单）。

    这类行的作用是给后面的内容分组，本身不是一个可复习的知识点；
    放进清单只会污染用户的背诵表。
    """
    raw = util.clean_line(name).strip(" ：:。；;、,.")
    if not raw:
        return True
    # 「第三章 电枢反应」-> 去掉章号后再判断，否则 "第三章" 和 "第三章节" 判断不一致
    s = _CHAPTER_NUM_PREFIX_RE.sub("", raw)
    s = re.sub(r"^\s*\d{1,2}\s+", "", s).strip(" ：:。；;、,.")
    if not s:
        return True
    if s in _LABEL_ONLY or raw in _LABEL_ONLY:
        return True
    if _CHAPTER_MARK_RE.match(s) or _CHAPTER_MARK_RE.match(raw):
        return True
    if re.match(r"^第\s*[0-9一二三四五六七八九十百]+\s*[章节讲部篇课]", s):
        return True
    # 「例题与本章小结」「小结与习题」这种复合版块
    if re.match(r"^(例题|习题|作业|思考题|练习题|复习题)", s):
        return True
    if s.endswith(("小结", "总结", "习题", "作业")) or re.search(r"(小结|总结)(与|和)(习题|作业|例题)", s):
        return True
    return False


def _half_sentence_with(term: str, text: str, *, limit: int = 60) -> str:
    """从一句话里截出「包含该术语的那半句」，到逗号/句号为止。

    这是最贴切的原话说明来源：用户忘了某术语是啥时，看到老师当场那句解释
    比看到一句我们自己编的概述有用得多。
    """
    norm_term = _match_norm(term)
    if not norm_term or not text:
        return ""
    raw = util.clean_line(text)

    # 说明句要「够格」：太短（"一条是机械特性"）或者只是把术语重复一遍
    # （"这叫可逆原理"）都不算解释，宁可继续往下降级。
    def usable(piece: str) -> bool:
        core = _match_norm(piece)
        if len(core) < 6:
            return False
        if core == norm_term:
            return False
        # 去掉术语本身后剩下的字太少 -> 只是复述术语，没有信息
        rest = core.replace(norm_term, "")
        if len(rest) < 4:
            return False
        # 目录式枚举（"后面还有 A、B、C"）只是片单，不解释任何东西，不算说明
        if re.search(r"(后面还|然后讲|先说|先讲|再讲|最后(讲|是)|还有|分别是|包括)", piece):
            return False
        if piece.count("、") >= 2:
            return False
        if re.search(r"一条是|一种是|一个是|第一种|第二种|主干|主线", piece):
            return False
        return True

    for sent in re.split(r"(?<=[。！？!?；;])", raw):
        if norm_term not in _match_norm(sent):
            continue
        # 优先取含术语的那一小句（到逗号为止）
        for piece in re.split(r"(?<=[，,、])", sent):
            piece = piece.strip(" ，,、。；;")
            if norm_term in _match_norm(piece) and usable(piece):
                return util.truncate(piece, limit)
        whole = sent.strip(" ，,、。；;")
        if usable(whole):
            return util.truncate(whole, limit)
    return ""


def _classify(name: str, desc: str, context: str) -> str:
    """判定知识点类型，取值必须是 概念|公式|步骤|结论|辨析|案例 之一。

    顺序有讲究：「辨析」「步骤」最具体，先判；「概念」兜底。
    """
    blob = f"{name}||{desc}||{context}"

    # 名称里写死了类型的，直接采信（「改善换向的方法」就是步骤，「A 与 B 的区别」就是辨析）
    if re.search(r"区别|不同|异同|对比|辨析|相同点|不同点|vs\.?", name, re.I):
        return "辨析"
    if re.search(r"方法|步骤|流程|程序|操作|接线|整定|起动|调速|措施|过程$", name):
        return "步骤"
    if re.search(r"[=＝]|公式|定律|定理|方程|表达式", name):
        return "公式"
    if re.search(r"例[题子如]|案例|实例|应用", name):
        return "案例"
    if re.search(r"结论|小结|综上|因此|所以", name):
        return "结论"

    if re.search(r"区别|不同之处|异同|对比|辨析|相同点|不同点|vs\.?", blob, re.I):
        return "辨析"
    if re.search(r"[=＝∑∫√]|公式|定律|定理|方程|表达式|系数|常数", blob):
        return "公式"
    if re.search(r"方法|步骤|流程|程序|操作|接线|整定|第[一二三四五六七八九十]步", blob):
        return "步骤"
    if re.search(r"例如|例题|例[子如]|如图|案例|实例|应用[中上]|工程上", blob):
        return "案例"
    if re.search(r"因此|所以|结论|综上|可知|可见|表明|说明|由此可见|总结", blob):
        return "结论"
    return "概念"


def build_knowledge(
    pages: Sequence[Dict[str, Any]],
    terms: TermIndex,
    global_emphasis: Sequence[Dict[str, Any]],
    segments: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """从课件行 + 讲解里的定义句 + 高频概念，抽出一份可二次加工的知识点清单。

    每条知识点包含：
      id / name / desc / page / t / importance / kind / prereq / pitfall /
      evidence（出处原话）/ flags（待补充原因列表）/ source（slide|speech|concept）

    设计取向（用户是要拿去做卡片/思维导图的）：
      * 一条一个知识点，章标题、版块标签一律不进；
      * 同一术语跨页出现合并成一条，所属页记成 "1,3"；
      * 「一句话说明」宁可朴素也不留空：讲解原话半句 -> 课件行原文 -> 「见第 N 页课件」；
      * 类型必须真的分出 概念/公式/步骤/结论/辨析/案例，不能全部落「概念」；
      * 信息不足的条目不编造内容，只把原因记进 flags，由 knowledge.md 集中列出。
    """
    global_terms = {_match_norm(str(item.get("text") or "")): item for item in global_emphasis}
    # 课件全课文本：用来挡住「只在口语里出现、课件上根本没有」的伪术语。
    # 人文类课的 n-gram 提取很容易把「并且咱们」「虽然咱」当概念，必须用课件兜底。
    slide_blob = _match_norm(
        " ".join(
            str(p.get("title") or "") + " " + " ".join(str(l) for l in (p.get("lines") or []))
            for p in pages
        )
    )
    candidates: Dict[str, Dict[str, Any]] = {}
    order: List[str] = []

    def add(name: str, *, page: Dict[str, Any], t: float, desc: str = "",
            source: str = "slide", evidence: str = "") -> None:
        """把候选知识点并入候选表（同名合并，跨页保留最早的出处页）。

        所有入口统一过一遍 _is_structural + _is_good_name：宁可少几条准的，
        也不要在清单里塞「第三章」「公式」这种脏条目。
        """
        name = util.clean_line(name).strip(" ：:。；;、,.")
        if not name or _is_structural(name):
            return
        if source == "title":
            if not _is_good_title(name):
                return
        elif not _is_good_name(name, from_speech=(source == "speech")):
            return
        key = _match_norm(name)
        if not key or len(key) < 2:
            return
        # 知识点必须能在课件上找到：只在口语里出现的 n-gram（「虽然咱」「第一次」
        # 「当然」）大多是半句话，不能当概念。宁可少几条，也不要污染清单。
        if key not in slide_blob:
            return
        # 两字术语（换向 / 饱和 / 调速）只有讲课里真的给出解释时才收；
        # 否则课件上到处是两字词，清单会被稀释成词表。
        if len(key) == 2 and not desc:
            vlog_skip = True  # noqa: F841  - 保持分支可读，实际直接返回
            return
        cur = candidates.get(key)
        if cur is None:
            candidates[key] = {
                "name": name,
                "desc": desc,
                "page": int(page["index"]),
                "t": float(t),
                "pages": {int(page["index"])},
                "source": source,
                "evidence": evidence,
                "page_refs": [page],
            }
            order.append(key)
            return
        cur["pages"].add(int(page["index"]))
        if page["index"] < cur["page"]:
            cur["page"] = int(page["index"])
            cur["t"] = float(t)
        cur["page_refs"].append(page)
        # 说明与出处：优先保留讲解来源（比课件行更口语、更像人话）
        if desc and (not cur["desc"] or source == "speech"):
            cur["desc"] = desc
        if evidence and (not cur["evidence"] or source == "speech"):
            cur["evidence"] = evidence

    # 1) 课件行：这是「老师要讲的东西」的骨架
    for page in pages:
        title_norm = _match_norm(str(page.get("title") or ""))
        for li, line in enumerate(page.get("lines") or []):
            if _is_structural(line):
                continue
            if _ADMIN_LINE_RE.search(line):
                continue
            name, desc = _name_from_line(line, terms)
            if not name:
                continue
            name_norm = _match_norm(name)
            # 标题被 OCR 截断成前缀（「某门课程名」「核心问题」）时别当知识点
            if title_norm and (
                title_norm.startswith(name_norm) or name_norm.startswith(title_norm)
            ):
                continue
            # 页面最上面的短行通常是版块标题 / 章节过渡句，不是知识点；
            # 但引号/书名号里的短句是经典原话或核心概念，要留下。
            if (
                li < 2
                and not desc
                and len(name_norm) <= 12
                and name[:1] not in "“\"《"
            ):
                continue
            # 短标题行（「某理论关于某问题的论述」）后面通常跟着解释段，
            # 取接下来的 3 行里第一句像解释的话做说明。
            if not desc and len(name_norm) <= 20:
                for nxt in (page.get("lines") or [])[li + 1 : li + 4]:
                    clean_nxt = util.clean_line(nxt)
                    if not (10 <= len(clean_nxt) <= 80):
                        continue
                    if _ADMIN_LINE_RE.search(clean_nxt) or util.is_watermark(clean_nxt):
                        continue
                    if _match_norm(clean_nxt) == name_norm:
                        continue
                    desc = clean_nxt
                    break
            if _match_norm(name) == title_norm:
                # 标题行不单独成条，只用来给该页条目分组（_chapter_of 会用到）
                continue
            add(name, page=page, t=page["t"], desc=desc, source="slide", evidence=line)

    # 1b) 页标题：人文类的标题往往就是这一节的核心概念
    #     （「何为教育」「如何成人」）。工科课件标题
    #     （「第三章 变压器」）会被 _is_structural 挡掉，不会混进来。
    for page in pages:
        title = util.clean_line(str(page.get("title") or "")).strip(" ：:。；;、,.")
        if not _is_good_title(title):
            continue
        if re.search(r"课程|学分|学时|课时|考核", title):
            continue
        # 没有标题的寄语/口号页：整段话是正文，不是概念名
        if title.startswith(("希望同学们", "让我们", "同学们", "大家", "亲爱的")):
            continue
        desc = _title_desc(title, page, segments)
        if not desc:
            for line in page.get("lines") or []:
                clean = util.clean_line(line)
                if _near_title(clean, title):
                    continue
                if _ADMIN_LINE_RE.search(clean) or util.is_watermark(clean):
                    continue
                if 6 <= len(clean) <= 60:
                    desc = clean
                    break
        add(title, page=page, t=page["t"], desc=util.truncate(desc, 120),
            source="title", evidence=title)

    # 2) 讲解里的定义句：「X 是指 Y」/「X 的定义是 Y」是最可靠的知识点来源    for page in pages:
        for sent, t in _speech_units(page, segments):
            cleaned = util.clean_line(sent)
            if _is_structural(cleaned):
                continue
            if _ADMIN_LINE_RE.search(cleaned):
                continue
            for pattern in _DEF_PATTERNS:
                m = pattern.match(cleaned)
                if not m:
                    continue
                name = _clean_name(m.group("a"))
                desc = util.truncate(m.group("b").strip(" 。；;，,"), 120)
                if not _is_good_name(name, from_speech=True) or len(_match_norm(desc)) < 4:
                    continue
                add(name, page=page, t=t, desc=desc, source="speech", evidence=sent)
                break

    # 3) 全课高频概念：即便课件只出现一个标题，也值得单独成条
    for term_key, item in global_terms.items():
        page_no = int(item.get("slide") or 1)
        page = next((p for p in pages if p["index"] == page_no), pages[0] if pages else None)
        if page is None:
            continue
        add(str(item.get("text") or ""), page=page, t=float(item.get("t") or page["t"]),
            source="concept", evidence="")

    # --- 组装 ---
    points: List[Dict[str, Any]] = []
    for key in order:
        cand = candidates[key]
        anchor = cand["page_refs"][0] if cand["page_refs"] else (pages[0] if pages else None)
        if anchor is None:
            continue
        name = cand["name"]
        name_norm = _match_norm(name)
        page_map = {int(p["index"]): p for p in pages}

        # 说明降级链（逐级降低「贴切度」，绝不返回空串）：
        #   1. 讲解里含该术语的那半句（最贴切，是老师原话）
        #   2. 课件上该术语所在的整行原文
        #   3. 「见第 N 页课件」这种指路说明，并标进「待补充」
        desc = cand["desc"]
        desc_source = "定义句" if desc else ""
        if not desc:
            # 两轮：先找「是指/定义为/叫做」这种真的在下定义的句子，
            # 找不到再接受任意含该术语的可用句（老师随口提到的也算参考）。
            for require_def in (True, False):
                for page_no in sorted(cand["pages"]):
                    p = page_map.get(page_no)
                    if not p:
                        continue
                    for sent, _t in _speech_units(p, segments):
                        if name_norm not in _match_norm(sent):
                            continue
                        if require_def and not re.search(
                            r"是指|指的是|定义为|叫做|称为|即为|意思是|也就是|相当于", sent
                        ):
                            continue
                        got = _half_sentence_with(name, sent)
                        # 说明几乎就是名称本身（"起动与调速"）等于没说
                        if got and len(_match_norm(got).replace(name_norm, "")) < 4:
                            got = ""
                        if got:
                            desc, desc_source = got, "原话"
                            break
                    if desc:
                        break
                if desc:
                    break
        if not desc:
            for page_no in sorted(cand["pages"]):
                p = page_map.get(page_no)
                if not p:
                    continue
                for line in p.get("lines") or []:
                    clean = util.clean_line(line)
                    core = _match_norm(clean)
                    if name_norm in core and len(core.replace(name_norm, "")) >= 4:
                        desc, desc_source = util.truncate(clean, 120), "课件原文"
                        break
                if desc:
                    break
        if not desc:
            desc, desc_source = f"见第 {anchor['index']} 页课件", "指路"

        # 「第三节，换向与火花，这一节同样是必考的」这类说明，句首的章节标记对用户没用
        desc = re.sub(r"^\s*第\s*[0-9一二三四五六七八九十百]+\s*[节讲课部]\s*[，,、]\s*", "", desc)
        desc = re.sub(r"^\s*(首先|然后|接下来|下面|好)[，,、]\s*", "", desc)

        # 最后一道关：说明如果去掉术语本身就没剩什么了，等于没解释 -> 走指路
        if desc_source != "指路" and len(_match_norm(desc).replace(name_norm, "")) < 4:
            desc, desc_source = f"见第 {anchor['index']} 页课件", "指路"
        # 引号顺序错乱（「…矣”日:“…」）说明是从整段引文里截出来的碎片，不是一句话说明
        _q = desc.strip("“”‘’\"'")
        if desc_source != "指路" and re.match(r"^[^““‘]*[”’]", _q):
            desc, desc_source = f"见第 {anchor['index']} 页课件", "指路"

        flags: List[str] = []
        if not desc:
            # 最后兜底：给出可跳转的指路说明，而不是空串
            flags.append("讲解中没找到对应解释，只有课件标题，需要自己补一句话说明")

        if len(name_norm) < 3:
            flags.append("术语偏短（2 字），可能被切碎，建议对照原图确认写法")
        if float(anchor.get("ocr_conf") or 0.0) and float(anchor.get("ocr_conf")) < 0.6:
            flags.append(f"该页识别置信度偏低（{float(anchor.get('ocr_conf')):.2f}），文字可能有误")
        if not (anchor.get("speech") or "").strip():
            flags.append("该页没有讲解字幕（可能是静音/翻页太快）")
        # 只要说明退化成「见第 N 页课件」，就说明没找到一句真正的解释；
        # 哪怕老师随口提过这个词，也不等于解释过，应该如实标进「待补充」。
        if desc_source == "指路":
            flags.append("课件与讲解都没有解释这个术语，建议自己补一句话说明")

        # 分类只看「知识点名 + 说明 + 它所在的那行课件」，不看整页：
        # 整页里总有公式，按整页判会把所有条目都打成「公式」。
        own_line = ""
        for line in anchor.get("lines") or []:
            if name_norm in _match_norm(line):
                own_line = util.clean_line(line)
                break
        kind = _classify(name, desc, own_line)

        pitfall = ""
        for page in pages:
            pitfall = _find_pitfall(name, [page], segments)
            if pitfall:
                break

        # 重要度：基础 2；全课高频 +1，跨页 +1，公式/辨析 +1，出现在划重点里抬到 4
        importance = 2
        if name_norm in global_terms:
            importance += 1
        if len(cand["pages"]) > 1:
            importance += 1
        if kind in ("公式", "辨析"):
            importance += 1
        if desc_source == "指路":
            importance = max(1, importance - 1)
        for page in pages:
            texts = [str(e.get("text") or "") for e in page.get("emphasis") or []]
            if any(name_norm in _match_norm(t) for t in texts):
                importance = max(importance, 4)
                break
        importance = _clamp(importance, 1, 5)

        points.append(
            {
                "name": name,
                "desc": desc,
                "page": int(anchor["index"]),
                "pages": sorted(cand["pages"]),
                "t": float(cand["t"]),
                "importance": importance,
                "kind": kind,
                "pitfall": pitfall,
                "evidence": cand["evidence"],
                "flags": flags,
                "source": cand["source"],
                "chapter": _chapter_of(anchor, pages),
            }
        )

    points.sort(key=lambda p: (p["page"], p["t"], -p["importance"], p["name"]))

    # --- 去冗余：短名被另一条完全包含、且自己没有独立说明，就并掉 ---
    kept: List[Dict[str, Any]] = []
    for p in points:
        key_p = _match_norm(p["name"])
        redundant = False
        for q in points:
            if q is p:
                continue
            key_q = _match_norm(q["name"])
            if key_q == key_p or key_p not in key_q:
                continue
            # 长名那条要有自己的说明，且短名没有，才认为短名是长名的一部分
            if q["desc"] and q.get("source") != "concept" and (
                not p["evidence"] or _match_norm(p["evidence"]) in _match_norm(q["evidence"])
            ):
                redundant = True
                break
        if not redundant:
            kept.append(p)
    points = kept

    # --- 编号 + 前置知识点 ---
    for i, p in enumerate(points):
        p["id"] = f"K{i + 1:03d}"

    for i, p in enumerate(points):
        blob = _match_norm(p["desc"] + p["name"])
        prereq = ""
        for q in points[:i]:
            key_q = _match_norm(q["name"])
            if len(key_q) < 2 or key_q in _match_norm(p["name"]):
                continue
            if key_q in blob:
                prereq = f"{q['id']} {q['name']}"
        p["prereq"] = prereq

    return points


def _chapter_of(page: Dict[str, Any], pages: Sequence[Dict[str, Any]]) -> str:
    """回溯找该页所属章节：最近一个「第X章 / 专题 / 模块」标题，找不到就用页标题。"""
    title = util.clean_line(str(page.get("title") or ""))
    if title and _CHAPTER_RE.match(title):
        return title
    for prev in reversed([p for p in pages if p["index"] <= page["index"]]):
        prev_title = util.clean_line(str(prev.get("title") or ""))
        if prev_title and _CHAPTER_RE.match(prev_title):
            return prev_title
    return title or f"第 {page['index']} 页"


# ==========================================================================
# 七、渲染：course.md / knowledge.md / knowledge.csv / emphasis.json
# ==========================================================================


def _collect_emphasis(pages: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """汇总全课重点条目（去重、排序），供 emphasis.json 与重点速览使用。"""
    best: Dict[str, Dict[str, Any]] = {}
    for page in pages:
        for item in page.get("emphasis") or []:
            key = _match_norm(item.get("text") or "")
            if not key:
                continue
            entry = {
                "text": item.get("text") or "",
                "score": int(item.get("score") or 0),
                "t": float(item.get("t") or 0.0),
                "slide": int(page["index"]),
                "reasons": list(item.get("reasons") or []),
                "source": item.get("source") or "speech",
            }
            cur = best.get(key)
            if cur is None or entry["score"] > cur["score"]:
                best[key] = entry
    items = list(best.values())
    items.sort(key=lambda x: (-x["score"], x["t"], x["slide"]))
    return items


def _course_title(root: Path, pages: Sequence[Dict[str, Any]]) -> str:
    """课程标题：优先 manifest 里的视频名，其次第一页标题，最后兜底。"""
    manifest = util.read_json(root / "manifest.json", default={}) or {}
    video = str(manifest.get("video") or "")
    if video:
        return util.slugify(video, fallback="课程笔记")
    for page in pages:
        title = util.clean_line(str(page.get("title") or ""))
        if title:
            return title
    return "课程笔记"


def _slide_image(out_dir: Path, index: int) -> str:
    """按契约的命名规则给出课件图片相对路径；文件不存在则返回空串。"""
    rel = f"slides/slide_{index:04d}.png"
    return rel if (out_dir / rel).exists() else ""


def render_slide_blocks(blocks: Sequence[Dict[str, Any]]) -> List[str]:
    """把一页的版面块渲染成 Markdown 行。

    这是笔记的主体：课件上写了什么，就按它原本的层级写出来。
    标题用加粗、要点用列表、引文用引用块、出处用小字。
    """
    out: List[str] = []
    for block in blocks or []:
        text = util.clean_line(str(block.get("text") or ""))
        if not text:
            continue
        role = str(block.get("role") or "body")
        if role == "heading":
            out.append(f"**{text}**")
            out.append("")
        elif role == "subhead":
            out.append(f"**{text}**")
            out.append("")
        elif role == "bullet":
            body = _BULLET_PREFIX_STRIP.sub("", text).strip()
            out.append(f"- {body or text}")
        elif role == "quote":
            out.append(f"> {text}")
            out.append("")
        elif role == "cite":
            out.append(f"> {text}")
            out.append("")
        else:
            # 正文行之间不插空行：PPT 上它们是一个条目列表，
            # 每行都加空行会撑成一大片空白，读起来很散。
            # 但要保证后面接引用块/标题时有空行，否则 Markdown 会粘连。
            out.append(text)
    # 规范化空行：Markdown 里引用块 / 加粗独立行前后都要有空行，
    # 否则会和上一行粘成同一段，渲染出来就乱了。
    cleaned: List[str] = []
    for line in out:
        needs_gap = line.startswith(">")
        if needs_gap and cleaned and cleaned[-1] != "":
            cleaned.append("")
        if cleaned and line == "" and cleaned[-1] == "":
            continue
        cleaned.append(line)
    while cleaned and cleaned[-1] == "":
        cleaned.pop()
    return cleaned


_BULLET_PREFIX_STRIP = re.compile(r"^\s*(?:[•·▪◦‣∙]|[-–—*])\s*")


def render_course_md(
    out_dir: Path,
    *,
    title: str,
    pages: Sequence[Dict[str, Any]],
    emphasis: Sequence[Dict[str, Any]],
    duration: float,
    segment_count: int,
    knowledge_count: int,
) -> str:
    """生成 course.md 文本（重点速览 → 课件目录 → 逐页正文 → 附录）。"""
    lines: List[str] = []
    add = lines.append

    add(f"# {title} · 课程笔记")
    add("")
    meta = [
        f"总时长 {util.human_duration(duration)}",
        f"课件 {len(pages)} 页",
        f"字幕 {segment_count} 条",
        f"重点 {len(emphasis)} 条",
        f"知识点 {knowledge_count} 个",
        f"生成于 {util.stamp()}",
    ]
    add("> " + " · ".join(meta))
    add("")
    add("- [重点速览](#重点速览)")
    add("- [课件目录](#课件目录)")
    add("- [逐页笔记](#逐页笔记)")
    add("- [附录](#附录)")
    add("")

    # ---------- 重点速览 ----------
    add("## 重点速览")
    add("")
    if emphasis:
        add(
            f"> 从全部讲解中按「考试信号词 / 念课件 / 重复出现 / 定义与结论句式 / 序号枚举 / 公式数字」"
            f"加权筛出，按重要度排序，共 {len(emphasis)} 条。"
        )
        add("")
        rows = []
        for i, item in enumerate(emphasis[:_MAX_OVERVIEW], 1):
            rows.append(
                [
                    str(i),
                    f"`{util.fmt_ts(item['t'])}`",
                    f"第 {item['slide']} 页",
                    util.truncate(item["text"], 60),
                    "；".join(item["reasons"]) or "综合得分",
                ]
            )
        lines.extend(md_table(["#", "时间", "页", "重点", "为什么是重点"], rows))
        if len(emphasis) > _MAX_OVERVIEW:
            add("")
            add(f"> 其余 {len(emphasis) - _MAX_OVERVIEW} 条见各页「划重点」与 `emphasis.json`。")
    else:
        add("> 本次没有达到阈值的重点句（阈值见 `emphasis.json` 的 `min_score`）。")
    add("")

    # ---------- 课件目录 ----------
    add("## 课件目录")
    add("")
    rows = []
    for page in pages:
        span = page["t_end"] - page["t"]
        rows.append(
            [
                f"第 {page['index']} 页",
                f"`{util.fmt_ts(page['t'])}`",
                util.truncate(page["title"] or "（未识别出标题）", 28),
                util.human_duration(span),
                str(len(page.get("emphasis") or [])),
            ]
        )
    lines.extend(md_table(["页", "开始", "标题", "讲解时长", "重点"], rows))
    add("")

    # ---------- 逐页笔记 ----------
    add("## 逐页笔记")
    add("")
    for page in pages:
        heading_title = page["title"] or "（未识别出标题）"
        add(f"### 第 {page['index']} 页 · {heading_title}")
        add("")
        add(
            f"`{util.fmt_ts(page['t'])} - {util.fmt_ts(page['t_end'])}`"
            f" · 讲解 {util.human_duration(page['t_end'] - page['t'])}"
        )
        add("")

        img = _slide_image(out_dir, page["index"])
        if img:
            add(f"![第 {page['index']} 页]({img})")
            add("")

        # ---------- 课件原文（主体）----------
        # 笔记以「PPT 上写了什么」为中心，老师讲解降为辅助说明。
        blocks = page.get("blocks") or []
        if blocks:
            lines.extend(render_slide_blocks(blocks))
            add("")
        elif page.get("lines"):
            for line in page["lines"]:
                add(f"- {line}")
            add("")

        # 图示
        if page.get("has_figure") or page.get("figure_note"):
            note = page.get("figure_note") or "该页有图表/示意图，见上方原图。"
            add(f"*图示*：{note}")
            add("")

        # ---------- 老师讲解（辅助，折叠） ----------
        if page.get("speech"):
            add("<details><summary>老师讲解（辅助，点击展开）</summary>")
            add("")
            sents = [s for s, _ in split_sentences(page["speech"])]
            para: List[str] = []
            for sent in sents:
                para.append(sent)
                if len(para) >= 3:          # 三句一段，避免大段豆腐块
                    add(" ".join(para))
                    add("")
                    para = []
            if para:
                add(" ".join(para))
                add("")
            add("</details>")
            add("")

        # 划重点
        if page.get("emphasis"):
            add("**划重点**")
            add("")
            for item in page["emphasis"]:
                stars = "🔥" * min(3, max(1, item["score"] // 3))
                add(f"- {stars} `{util.fmt_ts(item['t'])}` **{item['text']}**")
                add(f"  - 为什么是重点：{'；'.join(item['reasons'])}")
            add("")
        else:
            add("**划重点**：本页没有强信号句。")
            add("")
        add("---")
        add("")

    # ---------- 附录 ----------
    add("## 附录")
    add("")
    add(f"- 完整字幕：[`transcript.srt`](transcript.srt)（{segment_count} 条）")
    add("- 知识点清单：[`knowledge.md`](knowledge.md) · 表格版 [`knowledge.csv`](knowledge.csv)")
    add("- 重点数据：[`emphasis.json`](emphasis.json)")
    add("")
    return "\n".join(lines).rstrip() + "\n"


def render_knowledge_md(
    *,
    title: str,
    points: Sequence[Dict[str, Any]],
    pages: Sequence[Dict[str, Any]],
) -> str:
    """生成 knowledge.md：概览表 + 按章节分组 + 待补充/存疑区。"""
    lines: List[str] = []
    add = lines.append

    kinds = Counter(p["kind"] for p in points)
    kind_mix = " / ".join(f"{k} {kinds[k]}" for k in _KIND_ORDER if kinds.get(k))
    add(f"# {title} · 知识点清单")
    add("")
    add(
        f"> 共 {len(points)} 条 · {kind_mix or '无'} · 覆盖 {len({p['page'] for p in points})} 页"
        f" · 生成于 {util.stamp()}"
    )
    add("")
    add("> 用途：直接导入 Anki / 做成思维导图 / 当背诵表。一条一个知识点，附页号与时间戳便于回看录像。")
    add("")

    # ---------- 概览 ----------
    add("## 概览")
    add("")
    rows = []
    for p in points:
        rows.append(
            [
                p["id"],
                util.truncate(p["name"], 22),
                util.truncate(p["desc"], 34) or "—",
                p["kind"],
                _stars(p["importance"]),
                _pages_label(p.get("pages") or [p["page"] or 0]),
                util.fmt_ts(p["t"]),
            ]
        )
    lines.extend(md_table(["ID", "知识点", "一句话说明", "类型", "重要度", "页", "时间"], rows))
    add("")

    # ---------- 按章节 ----------
    add("## 按章节")
    add("")
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for p in points:
        grouped.setdefault(p["chapter"], []).append(p)

    for chapter, group in grouped.items():
        page_span = sorted({g["page"] for g in group})
        span = f"第 {page_span[0]} 页" if len(page_span) == 1 else f"第 {page_span[0]}–{page_span[-1]} 页"
        add(f"### {chapter}（{span}）")
        add("")
        for p in group:
            add(f"**{p['id']} · {p['name']}**  `{_stars(p['importance'])}` `{p['kind']}`")
            add("")
            add(f"- 一句话说明：{p['desc'] or '_待补充_'}")
            add(f"- 出处：{_pages_text(p)} · `{util.fmt_ts(p['t'])}`")
            if p.get("prereq"):
                add(f"- 前置知识点：{p['prereq']}")
            if p.get("pitfall"):
                add(f"- 易错点：{p['pitfall']}")
            if p.get("evidence") and p.get("source") in ("speech", "slide"):
                add(f"- 原文：{util.truncate(p['evidence'], 90)}")
            add("")

    # ---------- 待补充 / 存疑 ----------
    # 这是给用户「自己补」的入口，无论有没有条目都必须渲染出来，
    # 否则用户不知道有这个入口可用。
    flagged = [p for p in points if p.get("flags")]
    add("## 待补充 / 存疑")
    add("")
    if flagged:
        add(
            f"> 共 {len(flagged)} 条信息不足。笔记没有替你编内容，"
            f"请对照 `slides/` 原图或回看录像自行补齐（补齐后可参考 `knowledge.csv` 的列结构回填）。"
        )
        add("")
        rows = [[p["id"], p["name"], _pages_text(p), "；".join(p["flags"])] for p in flagged]
        lines.extend(md_table(["ID", "知识点", "页", "为什么存疑 / 待补什么"], rows))
        add("")
        add("<details><summary>逐条展开</summary>")
        add("")
        for p in flagged:
            add(f"- **{p['id']} {p['name']}**（{_pages_text(p)} · `{util.fmt_ts(p['t'])}`）")
            for flag in p["flags"]:
                add(f"  - {flag}")
        add("")
        add("</details>")
        add("")
    else:
        add("> 本次没有信息不足的条目：每条知识点都有讲解原话或课件原文支撑。")
        add(">")
        add("> 如果你在教材上看到需要补充的定义（比如更完整的公式条件），可以按")
        add("> `knowledge.csv` 的列格式手工追加一行，重要度自己打分即可。")
        add("")

    # ---------- 图示页提示 ----------
    figs = [p for p in pages if p.get("has_figure") or p.get("figure_note")]
    if figs:
        add("## 含图表的页")
        add("")
        for page in figs:
            note = page.get("figure_note") or "检测到图表/公式区域，讲解里未提供说明"
            add(f"- 第 {page['index']} 页（`{util.fmt_ts(page['t'])}`）：{note}")
        add("")

    return "\n".join(lines).rstrip() + "\n"


def write_knowledge_csv(path: Path, points: Sequence[Dict[str, Any]]) -> Path:
    """写 knowledge.csv：列固定为契约 §7，UTF-8 带 BOM（Excel/WPS 中文不乱码）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["id", "知识点", "一句话说明", "所属页", "时间", "重要度", "类型", "前置知识点", "易错点"]
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.writer(fh, quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)
        for p in points:
            writer.writerow(
                [
                    p["id"],
                    p["name"],
                    p["desc"],
                    _pages_label(p.get("pages") or [p.get("page") or 0]),
                    util.fmt_ts(p["t"]) if p.get("t") is not None else "",
                    str(int(p["importance"])),
                    p["kind"],
                    p.get("prereq") or "",
                    p.get("pitfall") or "",
                ]
            )
    tmp.replace(path)
    return path


def write_emphasis_json(path: Path, items: Sequence[Dict[str, Any]], *, min_score: int) -> Dict[str, Any]:
    """写 emphasis.json（契约 §6）：items 按 score 降序，再按 t 升序。

    返回写出去的那个 dict（renders 的调用方要直接读 items 长度）。
    """
    payload = {
        "min_score": int(min_score),
        "items": [
            {
                "text": item["text"],
                "score": int(item["score"]),
                "t": item["t"],
                "slide": int(item["slide"]),
                "reasons": list(item["reasons"]),
                "source": item["source"],
            }
            for item in items
        ],
    }
    util.write_json(path, payload)
    return payload


def export_for_pdf(out_dir: str | Path, *, min_score: int = 3) -> Dict[str, Any]:
    """给 PDF 渲染准备数据：页（含 blocks）、知识点、重点、标题、时长。

    和 render() 复用同一套 align/build_knowledge，保证 PDF 与 Markdown
    内容一致，不会出现"两个版本对不上"。
    """
    root = Path(out_dir)
    # 一定要重新 align，不能直接复用已有的 aligned.json：
    # 用户可能先跑了 --skip-asr（没有讲解），之后补跑 asr 再出 PDF，
    # 这时旧 aligned.json 里每页 speech 还是空的，PDF 就会变成"没有重点"。
    # align 本身很快（几百毫秒），每次重算最省心。
    align(root, min_score=min_score)
    aligned = util.read_json(root / "aligned.json", default=None)
    if not isinstance(aligned, dict) or not aligned.get("slides"):
        raise CvnError("aligned.json 里没有课件页，无法生成 PDF")

    pages = aligned["slides"]
    global_emphasis = list(aligned.get("global_emphasis") or [])
    frames = util.read_json(root / "frames.json", default={}) or {}
    transcript = util.read_json(root / "transcript.json", default={}) or {}
    segments = list(transcript.get("segments") or [])

    duration = float(frames.get("duration") or transcript.get("duration") or 0.0)
    if not duration and pages:
        duration = float(pages[-1].get("t_end") or 0.0)

    docs: List[Tuple[int, str, float]] = []
    for page in pages:
        for line in page.get("lines") or []:
            docs.append((int(page["index"]), str(line), float(page.get("t") or 0.0)))
        for sent, t in _speech_units(page, segments):
            docs.append((int(page["index"]), sent, t))
    terms = TermIndex.build(docs)
    points = build_knowledge(pages, terms, global_emphasis, segments)
    emphasis_items = _collect_emphasis(pages)

    # 给每页补一个人读的时间戳，PDF 里直接用
    out_pages: List[Dict[str, Any]] = []
    for i, page in enumerate(pages, 1):
        rec = dict(page)
        rec["t_human"] = util.fmt_ts(float(page.get("t") or 0.0))
        rec["seq"] = i
        out_pages.append(rec)

    for item in emphasis_items:
        item["t_human"] = util.fmt_ts(float(item.get("t") or 0.0))

    return {
        "title": _course_title(root, pages),
        "pages": out_pages,
        "knowledge": points,
        "emphasis": emphasis_items,
        "duration": duration,
    }


def render(out_dir: str | Path, *, min_score: int = 3, title: Optional[str] = None) -> Dict[str, Any]:
    """读 aligned.json，渲染 course.md / knowledge.md / knowledge.csv / emphasis.json。

    若 aligned.json 不存在会自动先跑一次 align（方便单测与手动补跑）。

    参数
    ----
    out_dir : 运行目录
    min_score : 重点阈值（与 align 保持一致，只影响 emphasis.json 里的记录值）
    title : 课程名。None 时按 manifest 视频名 -> 首页标题 自动推断；
            传入则直接使用（CLI `--title` 让用户覆盖自动识别的课程名）

    返回
    ----
    摘要 dict：
      course / knowledge_md / knowledge_csv : 三个产物的路径（str）
      emphasis   : 写进 emphasis.json 的那个 dict（含 "min_score" 与 "items" 列表）
      knowledge_count / emphasis_count : 知识点条数 / 重点条数（int）
    """
    root = Path(out_dir)
    aligned_path = root / "aligned.json"
    aligned = util.read_json(aligned_path, default=None)
    if not isinstance(aligned, dict) or not aligned.get("slides"):
        util.vlog("aligned.json 不存在或为空，先自动运行 align")
        align(root, min_score=min_score)
        aligned = util.read_json(aligned_path, default=None)
    if not isinstance(aligned, dict) or not aligned.get("slides"):
        raise CvnError("aligned.json 里没有课件页，无法生成笔记（请检查 frames.json / slides.json）")

    pages = aligned["slides"]
    global_emphasis = list(aligned.get("global_emphasis") or [])

    frames = util.read_json(root / "frames.json", default={}) or {}
    transcript = util.read_json(root / "transcript.json", default={}) or {}
    segments = list(transcript.get("segments") or [])
    duration = float(frames.get("duration") or transcript.get("duration") or 0.0)
    if not duration and pages:
        duration = float(pages[-1].get("t_end") or 0.0)

    # --- 术语索引（知识点抽取要用全课频次）---
    docs: List[Tuple[int, str, float]] = []
    for page in pages:
        for line in page.get("lines") or []:
            docs.append((int(page["index"]), str(line), float(page.get("t") or 0.0)))
        for sent, t in _speech_units(page, segments):
            docs.append((int(page["index"]), sent, t))
    terms = TermIndex.build(docs)

    course_name = util.clean_line(title) if title else _course_title(root, pages)
    emphasis_items = _collect_emphasis(pages)
    points = build_knowledge(pages, terms, global_emphasis, segments)

    course_md = render_course_md(
        root,
        title=course_name,
        pages=pages,
        emphasis=emphasis_items,
        duration=duration,
        segment_count=len(segments),
        knowledge_count=len(points),
    )
    knowledge_md = render_knowledge_md(title=course_name, points=points, pages=pages)

    course_path = util.write_text(root / "course.md", course_md)
    knowledge_md_path = util.write_text(root / "knowledge.md", knowledge_md)
    knowledge_csv_path = write_knowledge_csv(root / "knowledge.csv", points)
    emphasis_payload = write_emphasis_json(root / "emphasis.json", emphasis_items, min_score=min_score)

    _mark_manifest(root, "merge")

    util.vlog(
        f"render: {len(emphasis_items)} 条重点 / {len(points)} 个知识点 -> course.md, knowledge.md, knowledge.csv"
    )
    return {
        "course": str(course_path),
        "knowledge_md": str(knowledge_md_path),
        "knowledge_csv": str(knowledge_csv_path),
        "emphasis": emphasis_payload,
        "knowledge_count": len(points),
        "emphasis_count": len(emphasis_items),
    }


# ==========================================================================
# 八、命令行入口（只用于自测 / 手动补跑；正式流程由 course_kit.py 调度）
# ==========================================================================


def main(argv: Optional[Sequence[str]] = None) -> int:
    """命令行入口：python3 -m cvn.notes <out_dir> [--only align|render] [--min-score N]"""
    parser = argparse.ArgumentParser(
        prog="cvn.notes",
        description="把课件识别 + 逐字稿融合成课程笔记与知识点清单",
    )
    parser.add_argument("out_dir", help="运行目录（含 frames.json / slides.json / transcript.json）")
    parser.add_argument("--only", choices=["align", "render"], help="只跑某个阶段")
    parser.add_argument("--min-score", type=int, default=3, help="重点阈值，默认 3")
    parser.add_argument("--title", default=None, help="课程名（覆盖自动识别）")
    parser.add_argument("-v", "--verbose", action="store_true", help="打印详细日志")
    try:
        args = parser.parse_args(list(argv) if argv is not None else None)
    except SystemExit:
        return 2

    util.set_verbose(args.verbose)
    try:
        if args.only in (None, "align"):
            info = align(args.out_dir, min_score=args.min_score)
            util.log(f"✅ align 完成：{info['slides']} 页 / {info['emphasis']} 条重点")
        if args.only in (None, "render"):
            out = render(args.out_dir, min_score=args.min_score, title=args.title)
            util.log(
                f"✅ render 完成：{out['knowledge_count']} 个知识点 / "
                f"{len(out['emphasis']['items'])} 条重点"
                f"\n   {out['course']}\n   {out['knowledge_md']}\n   {out['knowledge_csv']}"
            )
    except CvnError as exc:
        util.warn(str(exc))
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
