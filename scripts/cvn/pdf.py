"""把课程笔记渲染成「图文并茂」的 HTML，再交给 Chrome 打印成 PDF。

设计取向
--------
用户要的是一份**能直接看、能发给同学**的讲义：每一页课件配原图，
下面是课件上写的字（按版面还原的层级），讲解折叠成小字注释。

排版原则：
  - 一页课件 = 一个"节"，图片和文字放在同一屏附近，不要跨页断开
  - 课件原图最大宽度按 A4 正文宽（约 170mm）适配
  - 不引入外部字体/CDN，Chrome 直接用它本机字体
  - 中文字体优先无衬线（苹方/思源黑体等），行高给足（1.75）
  - 页码、页眉用 CSS `@page` + `position: fixed` 实现

为什么走 HTML 而不是 reportlab：
  中文排版、图片自适应、折叠块这些用 CSS 几行就够，
  用 reportlab 要手算每一个坐标，反而更容易出错。
"""

from __future__ import annotations

import html
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

_CSS = """
@page {
  size: A4;
  margin: 14mm 14mm 14mm 14mm;
}
* { box-sizing: border-box; }
html, body {
  margin: 0; padding: 0;
  font-family: "PingFang SC", "Hiragino Sans GB", "Source Han Sans SC",
               "Noto Sans CJK SC", "Microsoft YaHei", sans-serif;
  font-size: 10pt;
  line-height: 1.65;
  color: #1a1a1a;
  -webkit-print-color-adjust: exact;
  print-color-adjust: exact;
}
h1, h2, h3, h4 { font-weight: 700; line-height: 1.35; margin: 0; }

/* ---------- 封面 ---------- */
.cover {
  height: 240mm;
  display: flex; flex-direction: column; justify-content: center;
  border-left: 6pt solid #2c5f8d;
  padding-left: 12mm;
}
.cover .kicker {
  font-size: 10pt; letter-spacing: 0.3em; color: #2c5f8d;
  text-transform: uppercase; margin-bottom: 8mm;
}
.cover h1 {
  font-size: 26pt; color: #10243a; margin-bottom: 6mm; max-width: 150mm;
}
.cover .meta { font-size: 9.5pt; color: #5a6b7c; line-height: 2; }
.cover .meta b { color: #2c5f8d; font-weight: 600; }

/* ---------- 目录 ---------- */
.toc { page-break-before: always; }
.toc h2 { font-size: 16pt; margin-bottom: 6mm; color: #10243a; }
.toc ol { padding-left: 6mm; }
.toc li { margin: 2.2mm 0; font-size: 10pt; }
.toc .t { color: #7a8b9c; font-size: 9pt; margin-left: 3mm; }

/* ---------- 每个课件页 ---------- */
.slide {
  page-break-before: always;
  /* 不要整节 avoid：课件正文可能很长，硬塞一页会被截断。
     改为让标题+图片不被拆开，正文允许自然流动。 */
}
.slide-head, .shot { page-break-after: avoid; }
.slide-head {
  display: flex; align-items: baseline; gap: 4mm;
  border-bottom: 1.5pt solid #2c5f8d;
  padding-bottom: 2mm; margin-bottom: 4mm;
}
.slide-no {
  font-size: 9pt; font-weight: 700; color: #fff; background: #2c5f8d;
  padding: 0.8mm 2.4mm; border-radius: 1mm; white-space: nowrap;
}
.slide-title { font-size: 15pt; font-weight: 700; color: #10243a; flex: 1; }
.slide-time { font-size: 8.5pt; color: #8a9aa8; white-space: nowrap; }
.cont {
  font-size: 8pt; font-weight: 500; color: #7a8b9c;
  border: 0.5pt solid #c9d6e2; border-radius: 1mm;
  padding: 0.2mm 1.4mm; margin-left: 2mm;
  vertical-align: middle;
}

.shot { margin: 0 0 4mm 0; text-align: center; page-break-inside: avoid; }
.shot img {
  max-width: 100%; max-height: 88mm;
  border: 0.5pt solid #d6dde4; border-radius: 1mm;
}
.shot figcaption { font-size: 8pt; color: #98a6b2; margin-top: 1.5mm; }

/* ---------- 课件正文（主体） ---------- */
.content { margin-bottom: 4mm; }
.block-heading {
  font-size: 13pt; font-weight: 700; color: #10243a;
  margin: 2mm 0 1.5mm 0;
}
.block-subhead {
  font-size: 11pt; font-weight: 600; color: #2c5f8d;
  margin: 2.5mm 0 1.5mm 0;
}
.block-body { margin: 1mm 0; text-indent: 0; }
.block-bullet {
  margin: 1mm 0 1mm 5mm; padding-left: 3mm;
  border-left: 1.5pt solid #c9d6e2;
}
.block-quote {
  margin: 3mm 0; padding: 3mm 5mm;
  background: #f5f8fb; border-left: 3pt solid #2c5f8d;
  font-size: 10pt; color: #33475b;
}
.block-cite {
  text-align: right; font-size: 8.5pt; color: #7a8b9c;
  margin: -1mm 0 3mm 0;
}

/* ---------- 讲解（辅助） ---------- */
.speech {
  margin-top: 3mm; padding: 2.5mm 3.5mm;
  background: #fafbfc; border: 0.5pt dashed #d6dde4;
  border-radius: 1.5mm;
  font-size: 8.8pt; color: #66757f; line-height: 1.7;
}
.speech .label {
  font-size: 8pt; font-weight: 700; color: #8a9aa8;
  letter-spacing: 0.1em; margin-bottom: 1.5mm;
}
.speech p { margin: 0 0 1.5mm 0; }

/* ---------- 知识点一览 ---------- */
.knowledge { page-break-before: always; }
.knowledge h2 { font-size: 16pt; margin-bottom: 5mm; color: #10243a; }
table.kn { width: 100%; border-collapse: collapse; font-size: 9pt; }
table.kn th {
  text-align: left; padding: 2mm; background: #eef3f8;
  color: #10243a; border-bottom: 1pt solid #c9d6e2; font-weight: 600;
}
table.kn td {
  padding: 1.8mm 2mm; border-bottom: 0.5pt solid #e8edf2;
  vertical-align: top;
}
table.kn tr { page-break-inside: avoid; }
.kn-id { color: #2c5f8d; font-weight: 700; white-space: nowrap; }
.kn-stars { color: #e8a33d; white-space: nowrap; }

/* ---------- 重点 ---------- */
.emphasis { page-break-before: always; }
.emphasis h2 { font-size: 16pt; margin-bottom: 5mm; color: #10243a; }
.emp-item {
  margin: 0 0 3mm 0; padding: 3mm 4mm;
  background: #fff9f0; border-left: 3pt solid #e8a33d;
  border-radius: 1.5mm; page-break-inside: avoid;
}
.emp-item .txt { font-size: 10pt; color: #3a2f22; font-weight: 600; }
.emp-item .why { font-size: 8.5pt; color: #8a7048; margin-top: 1mm; }
.emp-item .src { font-size: 8pt; color: #a89678; margin-top: 0.8mm; }

.empty { color: #98a6b2; font-size: 9pt; font-style: italic; }
"""


def _esc(text: Any) -> str:
    return html.escape(str(text or ""), quote=True)


def _stars(n: int) -> str:
    n = max(1, min(5, int(n or 1)))
    return "★" * n + "☆" * (5 - n)


def _image_src(out_dir: Path, index: int) -> str | None:
    for name in (f"slide_{index:04d}.png",):
        p = out_dir / "slides" / name
        if p.exists():
            return p.as_uri()
    return None


def render_block(block: Dict[str, Any]) -> str:
    """把一个版面块渲染成 HTML。"""
    text = _esc(block.get("text"))
    if not text:
        return ""
    role = str(block.get("role") or "body")
    if role == "heading":
        return f'<div class="block-heading">{text}</div>'
    if role == "subhead":
        return f'<div class="block-subhead">{text}</div>'
    if role == "bullet":
        return f'<div class="block-bullet">{text}</div>'
    if role == "quote":
        return f'<div class="block-quote">{text}</div>'
    if role == "cite":
        return f'<div class="block-cite">{text}</div>'
    return f'<div class="block-body">{text}</div>'


def _render_speech(speech: str) -> str:
    """老师讲解：折成小字，深色注释样式。"""
    if not speech:
        return ""
    # 按句切段，三句一段，避免大段文字
    parts = [p.strip() for p in re.split(r"(?<=[。！？])", speech) if p.strip()]
    paras: List[str] = []
    buf: List[str] = []
    for p in parts:
        buf.append(p)
        if len(buf) >= 3:
            paras.append("".join(buf))
            buf = []
    if buf:
        paras.append("".join(buf))
    body = "".join(f"<p>{_esc(p)}</p>" for p in paras[:12])
    return (
        '<div class="speech">'
        '<div class="label">老师讲解（辅助）</div>'
        f"{body}</div>"
    )


def build_html(
    *,
    title: str,
    pages: Sequence[Dict[str, Any]],
    knowledge: Sequence[Dict[str, Any]],
    emphasis: Sequence[Dict[str, Any]],
    out_dir: Path,
    source_name: str = "",
    duration: str = "",
) -> str:
    """拼出完整 HTML。"""
    missing_images = 0
    doc: List[str] = []
    doc.append("<!doctype html><html lang=\"zh-CN\"><head><meta charset=\"utf-8\">")
    doc.append(f"<title>{_esc(title)}</title>")
    doc.append(f"<style>{_CSS}</style></head><body>")

    # ---------- 封面 ----------
    doc.append('<section class="cover">')
    doc.append('<div class="kicker">Course Notes</div>')
    doc.append(f"<h1>{_esc(title)}</h1>")
    doc.append('<div class="meta">')
    if source_name:
        doc.append(f"素材：<b>{_esc(source_name)}</b><br>")
    if duration:
        doc.append(f"时长：<b>{_esc(duration)}</b><br>")
    doc.append(f"课件：<b>{len(pages)}</b> 页　")
    doc.append(f"知识点：<b>{len(knowledge)}</b> 条　")
    doc.append(f"重点：<b>{len(emphasis)}</b> 条")
    doc.append("</div></section>")

    # ---------- 目录 ----------
    doc.append('<section class="toc"><h2>目录</h2><ol>')
    for i, page in enumerate(pages, 1):
        t = page.get("title") or f"第 {page['index']} 页"
        stamp = page.get("t_human") or ""
        doc.append(
            f'<li>{_esc(t)}<span class="t">{_esc(stamp)}</span></li>'
        )
    doc.append("</ol></section>")

    # ---------- 逐页 ----------
    # 同一标题的连续页（同一张幻灯片分多屏讲）在标题后标「续」，
    # 读者一眼就知道这是同一主题的延续，而不是重复内容。
    prev_title = ""
    for i, page in enumerate(pages, 1):
        doc.append('<section class="slide">')
        title_txt = page.get("title") or f"第 {page['index']} 页"
        stamp = page.get("t_human") or ""
        is_cont = bool(prev_title) and title_txt == prev_title
        prev_title = title_txt
        doc.append('<div class="slide-head">')
        doc.append(f'<span class="slide-no">课件 {i}</span>')
        suffix = '<span class="cont">续</span>' if is_cont else ""
        doc.append(f'<span class="slide-title">{_esc(title_txt)}{suffix}</span>')
        if stamp:
            doc.append(f'<span class="slide-time">{_esc(stamp)}</span>')
        doc.append("</div>")

        src = _image_src(out_dir, int(page["index"]))
        if not src:
            missing_images += 1
        if src:
            doc.append('<figure class="shot">')
            doc.append(f'<img src="{src}" alt="课件第 {i} 页">')
            doc.append(
                f'<figcaption>原始画面 · slide_{int(page["index"]):04d}.png</figcaption>'
            )
            doc.append("</figure>")

        doc.append('<div class="content">')
        blocks = page.get("blocks") or []
        if blocks:
            for b in blocks:
                doc.append(render_block(b))
        elif page.get("lines"):
            for line in page["lines"]:
                doc.append(f'<div class="block-body">{_esc(line)}</div>')
        else:
            doc.append('<div class="empty">（这一页没有识别到文字）</div>')
        doc.append("</div>")

        doc.append(_render_speech(str(page.get("speech") or "")))
        doc.append("</section>")

    # ---------- 知识点 ----------
    doc.append('<section class="knowledge"><h2>知识点一览</h2>')
    if knowledge:
        doc.append('<table class="kn"><thead><tr>')
        for h in ("编号", "知识点", "一句话说明", "类型", "重要度", "页"):
            doc.append(f"<th>{h}</th>")
        doc.append("</tr></thead><tbody>")
        for k in knowledge:
            doc.append("<tr>")
            doc.append(f'<td class="kn-id">{_esc(k.get("id"))}</td>')
            doc.append(f'<td><b>{_esc(k.get("name"))}</b></td>')
            doc.append(f'<td>{_esc(k.get("desc"))}</td>')
            doc.append(f'<td>{_esc(k.get("kind"))}</td>')
            doc.append(f'<td class="kn-stars">{_stars(k.get("importance") or 1)}</td>')
            doc.append(f'<td>{_esc(k.get("pages"))}</td>')
            doc.append("</tr>")
        doc.append("</tbody></table>")
    else:
        doc.append('<div class="empty">本次没有抽出知识点。</div>')
    doc.append("</section>")

    # ---------- 重点 ----------
    doc.append('<section class="emphasis"><h2>重点速览</h2>')
    if emphasis:
        for item in emphasis:
            doc.append('<div class="emp-item">')
            doc.append(f'<div class="txt">{_esc(item.get("text"))}</div>')
            why = item.get("reason") or "；".join(item.get("reasons") or [])
            if why:
                doc.append(f'<div class="why">为什么是重点：{_esc(why)}</div>')
            src_bits = []
            if item.get("t_human"):
                src_bits.append(f'时间 {item["t_human"]}')
            if item.get("slide"):
                src_bits.append(f'课件 {item["slide"]}')
            if src_bits:
                doc.append(f'<div class="src">{" · ".join(src_bits)}</div>')
            doc.append("</div>")
    else:
        doc.append('<div class="empty">本次没有达到阈值的重点。</div>')
    doc.append("</section>")

    if missing_images:
        import sys as _sys
        print(
            f"[cvn] 警告：{missing_images} 页找不到课件原图（slides/ 里没有对应 png），"
            "PDF 会是纯文字版。请确认 slides/ 目录跟着 out 目录一起在。",
            file=_sys.stderr,
        )

    doc.append("</body></html>")
    return "".join(doc)


def write_html(html_text: str, path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(html_text, encoding="utf-8")
    return p


def find_chrome() -> str | None:
    """找一个可用的 Chrome/Chromium 可执行文件。"""
    import shutil

    for cand in (
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
    ):
        if Path(cand).exists():
            return cand
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def html_to_pdf(
    html_path: str | Path,
    pdf_path: str | Path,
    *,
    chrome: str | None = None,
    timeout: float = 300.0,
) -> Path:
    """用无头 Chrome 把 HTML 打印成 PDF。"""
    import subprocess
    import tempfile

    exe = chrome or find_chrome()
    if not exe:
        raise RuntimeError(
            "没找到 Chrome/Chromium/Edge。装一个，或用 --pdf-engine wkhtmltopdf。"
        )
    html_p = Path(html_path).resolve()
    pdf_p = Path(pdf_path).resolve()
    pdf_p.parent.mkdir(parents=True, exist_ok=True)

    # 用独立的 user-data-dir：避免和你正在用的 Chrome 抢锁，
    # 也避免它的 Crashpad 上报超时拖慢进程（实测能快 4 倍）。
    with tempfile.TemporaryDirectory(prefix="cvn-chrome-") as profile:
        cmd = [
            exe,
            "--headless",
            "--disable-gpu",
            "--disable-crash-reporter",
            "--disable-breakpad",
            "--no-first-run",
            "--no-default-browser-check",
            f"--user-data-dir={profile}",
            "--no-pdf-header-footer",
            f"--print-to-pdf={pdf_p}",
            html_p.as_uri(),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        if not pdf_p.exists():
            tail = ((proc.stderr or "") + (proc.stdout or "")).strip().splitlines()[-5:]
            raise RuntimeError("Chrome 打印 PDF 失败：\n" + "\n".join(tail))
    return pdf_p
