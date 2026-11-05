"""跨页纠错：用全课里的高置信度文本，修正同一门课中的低置信度错字。

为什么需要这一层
----------------
RapidOCR 对中文很强，但仍会犯「字形近似的错」：

    页 17（conf 1.00）：更重要的是：核心原则是什么？
    页 20（conf 0.92）：为核心原列？            ← 「则」错成「列」，还丢了开头

同一门课的课件是**同一套术语反复出现**的，所以「用全课多数版本去纠正
少数错误版本」是可靠信号，而且完全不引入外部知识——只在本课程内部自洽。

做法
----
1. 收集全课所有文字行，按「去标点后完全相同」分组，组内取置信度最高的
   版本作为该句的**规范形式**；
2. 对每一行，找它的规范形式；若当前行与规范形式只差个别字，就用规范形式替换；
3. 另外做一遍**常见形近字纠正**：只在「同一课程里已确认的正确写法」出现时
   才替换，避免瞎改。

不做的事
--------
- 不纠正课件独有的专名（除非本课程别处出现过正确写法）；
- 不改数字、符号、英文；
- 所有替换都记进 `corrections` 字段，可回溯、可关闭。
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, Iterable, List, Sequence, Tuple

_CJK = re.compile(r"[\u4e00-\u9fff]")
_PUNCT = re.compile(r"[\s，。、；：？！,.;:?!\"'“”‘’（）()《》「」『』\-—–·•]+")

# 常见形近字对（左 = OCR 容易认错的结果，右 = 正确字）。
# 只在「本课程别处出现过右字」时才启用，减少误伤。
_CONFUSABLE = {
    "列": "则",     # 核心原列 -> 核心原则
    "大": "太",     # 大学 与 太学（仅当本课出现过「太」）
    "入": "人",
    "八": "人",
    "己": "已",
    "末": "未",
    "卡": "卞",     # 卡庄子 -> 卞庄子
    "日": "曰",     # 「……矣」日：「……」 -> 曰
    "遞": "递",
    "換": "换",
    "載": "载",
}


def _norm(text: str) -> str:
    return _PUNCT.sub("", text or "").lower()


def build_canonical(
    lines: Iterable[Tuple[str, float]],
) -> Dict[str, Tuple[str, float]]:
    """按「去标点文本」聚合，取置信度最高的版本作为规范形式。

    参数：[(文本, 置信度), ...]
    返回：{归一化文本: (规范文本, 置信度)}
    """
    buckets: Dict[str, List[Tuple[str, float]]] = {}
    for text, conf in lines:
        key = _norm(str(text or ""))
        if len(key) < 2:
            continue
        buckets.setdefault(key, []).append((str(text), float(conf or 0.0)))
    out: Dict[str, Tuple[str, float]] = {}
    for key, items in buckets.items():
        best = max(items, key=lambda x: x[1])
        out[key] = best
    return out


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离（文本很短，O(n*m) 足够）。"""
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(
                prev[j] + 1,          # 删除
                cur[j - 1] + 1,       # 插入
                prev[j - 1] + (ca != cb),  # 替换
            ))
        prev = cur
    return prev[-1]


def correct_line(
    text: str,
    conf: float,
    canonical: Dict[str, Tuple[str, float]],
    *,
    min_conf_gain: float = 0.03,
) -> str | None:
    """给一行文字找「更可信的同句版本」，返回替换后的文本或 None。"""
    raw = str(text or "").strip()
    key = _norm(raw)
    if len(key) < 4:
        return None

    exact = canonical.get(key)
    if exact is not None:
        cand, cconf = exact
        if cand != raw and cconf > conf + min_conf_gain:
            return cand
        return None

    # 近似匹配只允许**逐字修正**，绝不整句替换。
    #
    # 为什么这么保守：一开始我做了"整行/片段替换"，结果把
    #   「核心问题是什么」-> 「更重要的是：核心问题是什么？」（整句被换掉）
    #   「2026/」    -> 「2025-2026聚焦建设更高水平平安中国 2025/」
    # 这类破坏性改动。纠错的价值在于「改几个错字」，不在「换一句话」。
    #
    # 所以在所有"置信度更高的近似行"里找同一个位置上的差异，
    # 只有当 长度相同、且只有 1~2 个字不同 时，才逐字替换。
    best: Tuple[str, float, int] | None = None
    for other_key, (cand, cconf) in canonical.items():
        if cconf <= conf + min_conf_gain:
            continue
        if len(other_key) != len(key):
            continue          # 长度必须一致，否则可能整句不同
        diff = [(i, a, b) for i, (a, b) in enumerate(zip(key, other_key)) if a != b]
        if not diff or len(diff) > 2:
            continue
        # 数字和拉丁字母一律不改：实测「幻灯片第142张」会被改成「190张」，
        # 这类改动看起来"更通顺"但其实改掉了真实信息，比不改更糟。
        if any(a.isdigit() or b.isdigit() or a.isascii() or b.isascii() for _i, a, b in diff):
            continue
        # 只修汉字之间的形近错
        if not all(("\u4e00" <= a <= "\u9fff") and ("\u4e00" <= b <= "\u9fff")
                   for _i, a, b in diff):
            continue
        if best is None or cconf > best[1]:
            best = (cand, cconf, len(diff))

    if best is None:
        return None

    cand, _cconf, _d = best
    # 把 cand 里的字按位置替换进原文（保持原文的标点与结构）
    if len(_norm(cand)) != len(key):
        return None
    out_chars: List[str] = []
    src_chars = list(cand)
    key_pos = 0
    for ch in raw:
        if _PUNCT.match(ch):
            out_chars.append(ch)
            continue
        if key_pos < len(src_chars):
            out_chars.append(src_chars[key_pos] if key_pos < len(src_chars) else ch)
        key_pos += 1
    fixed = "".join(out_chars)
    return fixed if fixed != raw else None


_GLYPH_CONFUSABLE = {
    # OCR 高频形近字对：错字 -> 正字。只在"正字在本课程出现过"时启用。
    "天": "人",
    "入": "人",
    "八": "人",
    "大": "太",
    "己": "已",
    "末": "未",
    "卡": "卞",
    "日": "曰",
    "换": "換",
}


def fix_glyph_by_confidence(
    pages: Sequence[Dict[str, Any]],
    *,
    min_gain: float = 0.08,
) -> int:
    """用「同一课程里置信度更高的近似写法」纠正形近字错。

    为什么单独做这一条：整句替换风险太大（会把数字、专名改错），
    但**只改一个形近字**风险很小、收益很直观：

        第17页 conf=0.99  「更重要的是：核心原则是什么？」
        第18页 conf=0.84  「核心原列是什么？」        ← 「则」被认成「列」

    判据（三条都满足才改）：
      1. 两行去掉标点后长度相同，只差 1 个字；
      2. 差异字在 `_GLYPH_CONFUSABLE` 里，且正字在本课程别处出现过；
      3. 正确版本的置信度比当前高 `min_gain` 以上。
    """
    if not pages:
        return 0

    pool: list[tuple[str, float]] = []
    for page in pages:
        for block in page.get("blocks") or []:
            text = str(block.get("text") or "")
            if not text:
                continue
            c = block.get("conf")
            pool.append((text, float(c) if c is not None else 0.5))

    # 归一到「去标点文本 -> (原文, 最高置信度)」
    best: Dict[str, tuple[str, float]] = {}
    for text, conf in pool:
        key = _norm(text)
        if len(key) < 4:
            continue
        cur = best.get(key)
        if cur is None or conf > cur[1]:
            best[key] = (text, conf)

    fixed_count = 0
    for page in pages:
        blocks = page.get("blocks") or []
        changed = False
        for block in blocks:
            text = str(block.get("text") or "")
            conf = float(block.get("conf") or 0.5)
            key = _norm(text)
            if len(key) < 4:
                continue
            applied = False
            for other_key, (other_text, other_conf) in best.items():
                if other_key == key:
                    continue
                if other_conf <= conf + min_gain:
                    continue
                # 优先：整行等长、只差 1 字
                if len(other_key) == len(key):
                    diff = [(a, b) for a, b in zip(key, other_key) if a != b]
                    if len(diff) == 1 and _GLYPH_CONFUSABLE.get(diff[0][0]) == diff[0][1]:
                        new_text = _replace_char(text, diff[0][0], diff[0][1], key, other_key)
                        if new_text and new_text != text:
                            page.setdefault("corrections", []).append(
                                {"from": text, "to": new_text,
                                 "reason": f"形近字 {diff[0][0]}→{diff[0][1]}"}
                            )
                            block["text"] = new_text
                            changed = True
                            fixed_count += 1
                            applied = True
                            break
                        continue
                # 其次：本行是更长高置信行的一个片段，逐字比对找出形近错
                # 例：本行「核心问提是什么」是「更重要的是：核心问题是什么」的一段
                if len(other_key) >= len(key) + 2:
                    for start in range(0, len(other_key) - len(key) + 1):
                        piece = other_key[start:start + len(key)]
                        diff = [(a, b) for a, b in zip(key, piece) if a != b]
                        if len(diff) != 1:
                            continue
                        wrong, right = diff[0]
                        if _GLYPH_CONFUSABLE.get(wrong) != right:
                            continue
                        pos = key.index(wrong)
                        new_text = _replace_at(text, pos, wrong, right)
                        if new_text and new_text != text:
                            page.setdefault("corrections", []).append(
                                {"from": text, "to": new_text,
                                 "reason": f"形近字 {wrong}→{right}（参照高置信度原句）"}
                            )
                            block["text"] = new_text
                            changed = True
                            fixed_count += 1
                            applied = True
                            break
                    if applied:
                        break
            if applied:
                continue
        if changed and page.get("corrections"):
            # 同步修正 lines / title 里出现同样错字的地方
            for rec in page["corrections"]:
                wrong = rec["from"]
                right = rec["to"]
                if page.get("title") == wrong:
                    page["title"] = right
                page["lines"] = [right if x == wrong else x for x in (page.get("lines") or [])]
    return fixed_count


def _replace_at(text: str, index: int, wrong: str, right: str) -> str | None:
    """把原文里「第 index 个实字」从 wrong 换成 right（跳过标点计数）。"""
    count = 0
    out = list(text)
    for pos, ch in enumerate(text):
        if _PUNCT.match(ch):
            continue
        if count == index:
            if ch != wrong:
                return None
            out[pos] = right
            return "".join(out)
        count += 1
    return None


def _replace_char(text: str, wrong: str, right: str, key: str, other_key: str) -> str | None:
    """在原文里把「对应位置」的那个错字换成正字（保留标点与结构）。"""
    # 找出差异在去标点串里的下标
    idx = next((i for i, (a, b) in enumerate(zip(key, other_key)) if a != b), None)
    if idx is None:
        return None
    # 把原文按"非标点字符"计数，定位到第 idx 个实字
    count = 0
    out = list(text)
    for pos, ch in enumerate(text):
        if _PUNCT.match(ch):
            continue
        if count == idx:
            if ch != wrong:
                return None
            out[pos] = right
            return "".join(out)
        count += 1
    return None


def apply_corrections(
    pages: Sequence[Dict[str, Any]],
    *,
    enabled: bool = False,
) -> int:
    """对 slides.json 里的 pages 做跨页纠错，返回修正条数。

    **默认关闭**。实测下来这条路的收益不足以抵消风险：
    同一门课里"多数版本正确"这个假设并不总成立，而且容易把
    数字、专名改错（「幻灯片第142张」→「第190张」）。

    真要开启（`enabled=True`）时也只做「长度相同、差 1~2 个汉字、
    不涉及数字/英文」的逐字修正，改动记录在 `corrections` 里可回溯。
    """
    if not enabled or not pages:
        return 0

    # 1) 收集全课所有行（带置信度）
    # 用**每行自己的**置信度，不要用整页平均——
    # 同一页里正确的行和认错的行置信度差别很大，用平均值会把错误版本
    # 当成"高置信度规范形式"，纠错方向就反了（实测踩过）。
    pool: List[Tuple[str, float]] = []
    for page in pages:
        page_conf = float(page.get("ocr_conf") or 0.5)
        for block in page.get("blocks") or []:
            text = str(block.get("text") or "")
            if not text:
                continue
            c = block.get("conf")
            pool.append((text, float(c) if c is not None else page_conf))
        if not page.get("blocks"):
            for line in page.get("lines") or []:
                pool.append((str(line), page_conf))
    canonical = build_canonical(pool)
    if not canonical:
        return 0

    # 2) 逐页修正
    total = 0
    for page in pages:
        conf = float(page.get("ocr_conf") or 0.5)
        records: List[Dict[str, str]] = []

        new_lines: List[str] = []
        for line in page.get("lines") or []:
            original = str(line)
            fixed = correct_line(original, conf, canonical)
            if fixed and fixed != original:
                records.append({"from": original, "to": fixed})
                new_lines.append(fixed)
            else:
                new_lines.append(original)
        if new_lines:
            page["lines"] = new_lines

        new_blocks: List[Dict[str, Any]] = []
        for block in page.get("blocks") or []:
            b = dict(block)
            original = str(b.get("text") or "")
            block_conf = float(b.get("conf") or conf)
            fixed = correct_line(original, block_conf, canonical)
            if fixed and fixed != original:
                records.append({"from": original, "to": fixed})
                b["text"] = fixed
            new_blocks.append(b)
        if new_blocks:
            page["blocks"] = new_blocks

        title = str(page.get("title") or "")
        fixed_title = correct_line(title, conf, canonical)
        if fixed_title and fixed_title != title:
            records.append({"from": title, "to": fixed_title})
            page["title"] = fixed_title

        if records:
            page["corrections"] = records
            total += len(records)
    return total
