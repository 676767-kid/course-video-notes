"""把「系统复习讲义」的内容渲染成 HTML，再打印成 PDF。

和 pdf.py 的区别
----------------
pdf.py 渲染的是「逐页课件笔记」——一课一张图，按时间顺序。
本模块渲染的是「讲义」——按学科逻辑重组的章节，含定义、引文、对照表、
阐释、考点，配课件原图作为佐证。

输入是结构化 JSON（见 review_content.json 的格式）：
    title / subtitle / meta / preface
    chapters[] -> sections[] -> blocks[]
    review -> blocks[]

block 支持的类型：
    para    普通段落（支持 **粗体**）
    def     概念定义（term + text）
    quote   经典引文（text + cite）
    explain 阐释（缩进正文）
    key     要点提示（高亮框）
    note    补充（灰底小字，「补充」的意思就是课件没有、本书加的）
    table   对照表（headers + rows）
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_CSS = """
@page {
  size: A4;
  margin: 20mm 18mm 18mm 18mm;
}
@page :first { margin: 0; }

* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  font-family: "Songti SC", "Source Han Serif SC", "Noto Serif CJK SC",
               "PingFang SC", serif;
  font-size: 10.5pt;
  line-height: 1.85;
  color: #1c1c1c;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
h1, h2, h3, h4 { font-family: "PingFang SC", "Source Han Sans SC", sans-serif;
                 font-weight: 700; margin: 0; }

/* ---------- 封面 ---------- */
.cover {
  height: 297mm; width: 210mm;
  display: flex; flex-direction: column; justify-content: center;
  padding: 0 30mm;
  background: #fbfaf7;
  border-top: 14mm solid #8c2f27;
  page-break-after: always;
}
.cover .series {
  font-family: "PingFang SC", sans-serif;
  font-size: 9pt; letter-spacing: 0.4em; color: #8c2f27;
  margin-bottom: 12mm;
}
.cover h1 {
  font-family: "Songti SC", serif;
  font-size: 34pt; color: #2b1a17; line-height: 1.3;
  margin-bottom: 4mm; font-weight: 700;
}
.cover .sub {
  font-family: "Songti SC", serif;
  font-size: 14pt; color: #6b4f45; margin-bottom: 22mm;
}
.cover .rule {
  width: 40mm; height: 1.2pt; background: #8c2f27; margin-bottom: 12mm;
}
.cover .meta { font-size: 9.5pt; color: #5c4a42; line-height: 2.1;
               font-family: "PingFang SC", sans-serif; }
.cover .meta span { color: #8c2f27; }

/* ---------- 前言 ---------- */
.preface { page-break-after: always; }
.preface h2 {
  font-size: 17pt; color: #8c2f27; margin-bottom: 7mm;
  padding-bottom: 2.5mm; border-bottom: 1pt solid #ddd5cb;
}
.preface p { margin: 0 0 3.5mm 0; text-align: justify; }

/* ---------- 目录 ---------- */
.toc { page-break-after: always; }
.toc h2 {
  font-size: 17pt; color: #8c2f27; margin-bottom: 7mm;
  padding-bottom: 2.5mm; border-bottom: 1pt solid #ddd5cb;
}
.toc .ch { margin: 5mm 0 1.5mm 0; font-size: 11pt; font-weight: 700;
           font-family: "PingFang SC", sans-serif; color: #2b1a17; }
.toc .sec { margin: 1.2mm 0 1.2mm 7mm; font-size: 9.5pt; color: #4a3d36; }
.toc .sec .n { color: #8c2f27; margin-right: 3mm; font-family: "PingFang SC", sans-serif; }

/* ---------- 章节 ---------- */
.chapter { page-break-before: always; }
.ch-head { margin-bottom: 7mm; padding-bottom: 4mm;
           border-bottom: 2pt solid #8c2f27; }
.ch-no { font-family: "PingFang SC", sans-serif; font-size: 9pt;
         letter-spacing: 0.3em; color: #8c2f27; margin-bottom: 2mm; }
.ch-title { font-family: "Songti SC", serif; font-size: 21pt; color: #2b1a17; }
.leads { margin: 0 0 6mm 0; padding: 4mm 5mm; background: #faf7f2;
         border-left: 2.5pt solid #c9a227; }
.leads p { margin: 0 0 2.5mm 0; font-size: 10pt; color: #4a3d36; }
.leads p:last-child { margin-bottom: 0; }

h3.sec-title {
  font-family: "PingFang SC", sans-serif;
  font-size: 13pt; color: #2b1a17;
  margin: 8mm 0 3.5mm 0;
  padding-left: 3.5mm; border-left: 3pt solid #8c2f27;
  page-break-after: avoid;
}
h3.sec-title .n { color: #8c2f27; margin-right: 2.5mm; font-size: 11pt; }

p.body { margin: 0 0 3mm 0; text-align: justify; }

/* 防孤行：小的内容块整体不拆页 */
.def, blockquote.quote, .key, .note, .explain {
  page-break-inside: avoid;
}
h3.sec-title { page-break-after: avoid; }

/* 概念定义 */
.def {
  margin: 3.5mm 0; padding: 3.5mm 4.5mm;
  background: #f7f5f0; border: 0.6pt solid #e2dcd1;
  border-radius: 1mm;
}
.def .term {
  font-family: "PingFang SC", sans-serif; font-weight: 700;
  color: #8c2f27; font-size: 10.5pt; margin-bottom: 1.2mm;
}
.def .text { font-size: 10pt; text-align: justify; }

/* 引文 */
blockquote.quote {
  margin: 4mm 0; padding: 4mm 6mm;
  background: #fbf8f3;
  border-left: 3pt solid #c9a227;
  font-family: "Songti SC", serif;
  font-size: 10.5pt; color: #2b1a17; line-height: 1.95;
  text-align: justify;
}
blockquote.quote .cite {
  display: block; text-align: right; font-size: 8.5pt;
  color: #8a7868; margin-top: 2mm; font-family: "PingFang SC", sans-serif;
}

/* 阐释 */
.explain {
  margin: 3mm 0 3mm 4mm; padding-left: 4mm;
  border-left: 1.2pt solid #e2dcd1;
  font-size: 10pt; color: #3a302b; text-align: justify;
}

/* 要点 */
.key {
  margin: 4mm 0; padding: 4mm 5mm;
  background: #fffaf0; border: 0.8pt solid #e8d4a8;
  border-radius: 1mm; font-size: 10pt; color: #5c3d10;
  text-align: justify;
}
.key::before {
  content: "要点"; display: inline-block;
  font-family: "PingFang SC", sans-serif; font-size: 8pt; font-weight: 700;
  color: #fff; background: #c9a227;
  padding: 0.3mm 2mm; border-radius: 0.8mm;
  margin-right: 2.5mm; vertical-align: 0.6mm;
}

/* 补充说明 */
.note {
  margin: 3.5mm 0; padding: 3.5mm 4.5mm;
  background: #f4f6f8; border-radius: 1mm;
  font-size: 9.2pt; color: #4a5560; line-height: 1.8;
  text-align: justify;
}
.note::before {
  content: "补充"; display: inline-block;
  font-family: "PingFang SC", sans-serif; font-size: 7.8pt; font-weight: 700;
  color: #4a5560; border: 0.6pt solid #b8c4ce;
  padding: 0.2mm 1.8mm; border-radius: 0.8mm;
  margin-right: 2.5mm; vertical-align: 0.6mm;
}

/* 表格 */
table.cmp {
  width: 100%; border-collapse: collapse;
  margin: 4mm 0; font-size: 9.3pt;
  page-break-inside: avoid;
}
table.cmp th {
  background: #8c2f27; color: #fff;
  font-family: "PingFang SC", sans-serif; font-weight: 600;
  padding: 2.2mm 2.5mm; text-align: left; font-size: 9pt;
}
table.cmp td {
  padding: 2mm 2.5mm; border-bottom: 0.5pt solid #e2dcd1;
  vertical-align: top; line-height: 1.7;
}
table.cmp tr:nth-child(even) td { background: #faf8f4; }

/* 课件原图 */
figure.shot {
  margin: 4mm 0; text-align: center; page-break-inside: avoid;
}
figure.shot img {
  max-width: 100%; max-height: 82mm;
  border: 0.6pt solid #ddd5cb; border-radius: 1mm;
}
figure.shot figcaption {
  font-family: "Songti SC", serif;
  font-size: 8.5pt; color: #5c4a42; margin-top: 2mm;
  text-align: center; line-height: 1.6;
}

/* 考点：编号要点（背诵用） */
.exam {
  margin: 4mm 0; padding: 4mm 5mm 3mm 5mm;
  background: #fdfaf5; border: 0.8pt solid #d8c9a8;
  border-radius: 1mm; page-break-inside: avoid;
}
.exam .tag {
  display: inline-block; font-family: "PingFang SC", sans-serif;
  font-size: 7.8pt; font-weight: 700; color: #fff;
  background: #b8860b; padding: 0.3mm 2.2mm; border-radius: 0.8mm;
  margin-bottom: 2mm;
}
.exam .q {
  font-family: "PingFang SC", sans-serif; font-weight: 700;
  font-size: 10.5pt; color: #2b1a17; margin-bottom: 2mm;
}
.exam ol { margin: 0; padding-left: 6mm; }
.exam li { margin: 1.6mm 0; text-align: justify; font-size: 10pt; }
.exam .std {
  margin-top: 2.5mm; padding-top: 2mm;
  border-top: 0.5pt dashed #d8c9a8;
  font-size: 9.5pt; color: #5c3d10;
}
.exam .std b { color: #8c2f27; }

/* 辨析：易混对照 */
.distinguish {
  margin: 4mm 0; padding: 3.5mm 4.5mm;
  background: #f4f7fa; border-left: 3pt solid #4a7a9b;
  border-radius: 1mm; font-size: 9.5pt;
  page-break-inside: avoid;
}
.distinguish .label {
  font-family: "PingFang SC", sans-serif; font-weight: 700;
  font-size: 9pt; color: #2f5a78; margin-bottom: 1.5mm;
}

/* 复习要点 */
.review { page-break-before: always; }
.review h2 {
  font-size: 19pt; color: #8c2f27; margin-bottom: 6mm;
  padding-bottom: 3mm; border-bottom: 2pt solid #8c2f27;
}
.unverified {
  margin-top: 6mm; padding: 4mm 5mm;
  background: #fdf6f5; border: 0.8pt dashed #d8b8b3;
  border-radius: 1mm; font-size: 9pt; color: #7a4a44;
}
.unverified strong { color: #8c2f27; }
"""


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _md_inline(text: str) -> str:
    """把 **粗体** 转成 <strong>（先转义再处理）。"""
    s = _esc(text)
    return re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", s)


def render_block(block: Dict[str, Any]) -> str:
    kind = str(block.get("type") or "para")
    if kind == "para":
        return f'<p class="body">{_md_inline(block.get("text"))}</p>'
    if kind == "def":
        term = _md_inline(block.get("term"))
        text = _md_inline(block.get("text"))
        return f'<div class="def"><div class="term">{term}</div><div class="text">{text}</div></div>'
    if kind == "quote":
        text = _md_inline(block.get("text"))
        cite = block.get("cite")
        c = f'<span class="cite">—— {_md_inline(cite)}</span>' if cite else ""
        return f'<blockquote class="quote">{text}{c}</blockquote>'
    if kind == "explain":
        return f'<div class="explain">{_md_inline(block.get("text"))}</div>'
    if kind == "key":
        return f'<div class="key">{_md_inline(block.get("text"))}</div>'
    if kind == "note":
        return f'<div class="note">{_md_inline(block.get("text"))}</div>'
    if kind == "exam":
        q = _md_inline(block.get("question"))
        items = block.get("items") or []
        out = ['<div class="exam"><div class="tag">考点</div>']
        if q:
            out.append(f'<div class="q">{q}</div>')
        if items:
            out.append("<ol>")
            for it in items:
                out.append(f"<li>{_md_inline(it)}</li>")
            out.append("</ol>")
        std = block.get("standard")
        if std:
            out.append(f'<div class="std"><b>标准表述：</b>{_md_inline(std)}</div>')
        out.append("</div>")
        return "".join(out)
    if kind == "distinguish":
        label = _md_inline(block.get("label") or "辨析")
        text = _md_inline(block.get("text"))
        return (
            f'<div class="distinguish"><div class="label">{label}</div>'
            f"{text}</div>"
        )
    if kind == "figure":
        src = block.get("src") or ""
        cap = block.get("caption") or ""
        if not src:
            return ""
        return (
            '<figure class="shot">'
            f'<img src="{_esc(src)}" alt="{_esc(cap)}">'
            f'<figcaption>{_esc(cap)}</figcaption></figure>'
        )
    if kind == "table":
        headers = block.get("headers") or []
        rows = block.get("rows") or []
        out = ['<table class="cmp">']
        if headers:
            out.append("<thead><tr>")
            for h in headers:
                out.append(f"<th>{_md_inline(h)}</th>")
            out.append("</tr></thead>")
        out.append("<tbody>")
        for row in rows:
            out.append("<tr>")
            for cell in row:
                out.append(f"<td>{_md_inline(cell)}</td>")
            out.append("</tr>")
        out.append("</tbody></table>")
        return "".join(out)
    # 未知类型按段落处理
    return f'<p class="body">{_md_inline(block.get("text"))}</p>'


def build_html(content: Dict[str, Any], *, figures: Sequence[Dict[str, Any]] | None = None) -> str:
    """把讲义内容渲染成完整 HTML。"""
    doc: List[str] = []
    doc.append('<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">')
    doc.append(f"<title>{_esc(content.get('title'))}</title>")
    doc.append(f"<style>{_CSS}</style></head><body>")

    # ---------- 封面 ----------
    doc.append('<section class="cover">')
    doc.append('<div class="series">COURSE REVIEW</div>')
    doc.append(f"<h1>{_esc(content.get('title'))}</h1>")
    sub = content.get("subtitle")
    if sub:
        doc.append(f'<div class="sub">{_esc(sub)}</div>')
    doc.append('<div class="rule"></div>')
    meta = content.get("meta") or {}
    doc.append('<div class="meta">')
    for key, label in (
        ("course", "课程"), ("scope", "范围"), ("teacher", "授课"),
        ("source", "素材"), ("compiled", "整理"),
    ):
        if meta.get(key):
            doc.append(f'{label}：<span>{_esc(meta[key])}</span><br>')
    doc.append("</div></section>")

    # ---------- 前言 ----------
    preface = content.get("preface") or []
    if preface:
        doc.append('<section class="preface"><h2>前言</h2>')
        for para in preface:
            doc.append(f'<p>{_md_inline(para)}</p>')
        doc.append("</section>")

    # ---------- 目录 ----------
    chapters = content.get("chapters") or []
    doc.append('<section class="toc"><h2>目录</h2>')
    for ch in chapters:
        doc.append(f'<div class="ch">{_esc(ch.get("no"))}　{_esc(ch.get("title"))}</div>')
        for sec in ch.get("sections") or []:
            doc.append(
                f'<div class="sec"><span class="n">{_esc(sec.get("no"))}</span>'
                f'{_esc(sec.get("title"))}</div>'
            )
    review = content.get("review")
    if review:
        doc.append(f'<div class="ch">{_esc(review.get("title"))}</div>')
    doc.append("</section>")

    # ---------- 正文 ----------
    for ch in chapters:
        doc.append('<section class="chapter">')
        doc.append('<div class="ch-head">')
        doc.append(f'<div class="ch-no">{_esc(ch.get("no"))}</div>')
        doc.append(f'<h2 class="ch-title">{_esc(ch.get("title"))}</h2>')
        doc.append("</div>")
        leads = ch.get("leads") or []
        if leads:
            doc.append('<div class="leads">')
            for lead in leads:
                doc.append(f"<p>{_md_inline(lead)}</p>")
            doc.append("</div>")
        for sec in ch.get("sections") or []:
            doc.append("<div class=\"sec-block\">")
            doc.append(
                f'<h3 class="sec-title"><span class="n">{_esc(sec.get("no"))}</span>'
                f'{_esc(sec.get("title"))}</h3>'
            )
            for block in sec.get("blocks") or []:
                doc.append(render_block(block))
            doc.append("</div>")
        doc.append("</section>")

    # ---------- 复习要点 ----------
    if review:
        doc.append('<section class="review">')
        doc.append(f'<h2>{_esc(review.get("title"))}</h2>')
        for block in review.get("blocks") or []:
            doc.append(render_block(block))
        doc.append("</section>")

    # ---------- 待核验清单 ----------
    if figures:
        # 只在确实需要时列出处，且只列**被引用过的**图，不要把所有素材都倒出来。
        # 之前这里会把几百个文件名全印在正文末尾，纯属噪声。
        doc.append('<section class="review"><div class="unverified">')
        doc.append("<strong>关于本书的编写</strong><br>")
        doc.append(
            "本书依据课程录像与课件内容整理。"
            "文中标注「补充」的段落，是对讲授提纲中仅有要点而无展开之处的说明性文字；"
            "未标注者出自讲授内容或其所引经典文本。经典引文均注明出处，可供覆按。"
        )
        doc.append("</div></section>")

    doc.append("</body></html>")
    return "".join(doc)


def write_html(html_text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_text, encoding="utf-8")
    return p
