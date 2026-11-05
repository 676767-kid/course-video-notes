"""Shared helpers for course-video-notes (cvn).

Every module in the package imports from here so that JSON on disk, log lines,
and timestamp formatting stay consistent across the pipeline stages
(probe -> frames -> ocr -> asr -> align -> render).

Nothing in this module may import a third-party package: it must work on a bare
CPython so that `course_kit.py probe` can always run and explain what is missing.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Iterable, Sequence

# --------------------------------------------------------------------------
# errors / logging
# --------------------------------------------------------------------------


class CvnError(RuntimeError):
    """A user-facing failure with an actionable message (no traceback needed)."""


_VERBOSE = False


def set_verbose(value: bool) -> None:
    global _VERBOSE
    _VERBOSE = bool(value)


def log(msg: str = "") -> None:
    """Diagnostics go to stderr so stdout stays parsable."""
    print(msg, file=sys.stderr, flush=True)


def vlog(msg: str) -> None:
    if _VERBOSE:
        log(msg)


def banner(title: str) -> None:
    log("")
    log(f"== {title} " + "=" * max(0, 58 - len(title)))


def warn(msg: str) -> None:
    log(f"!! {msg}")


# --------------------------------------------------------------------------
# json / paths
# --------------------------------------------------------------------------


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_json(path: str | os.PathLike[str], default: Any = None) -> Any:
    p = Path(path)
    if not p.exists():
        return default
    try:
        with p.open("r", encoding="utf-8") as fh:
            return json.load(fh)
    except json.JSONDecodeError as exc:
        raise CvnError(f"{p} 不是合法 JSON：{exc}") from exc


def write_json(path: str | os.PathLike[str], obj: Any) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2, sort_keys=False)
        fh.write("\n")
    tmp.replace(p)
    return p


def write_text(path: str | os.PathLike[str], text: str) -> Path:
    p = Path(path)
    ensure_dir(p.parent)
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(p)
    return p


def slugify(name: str, fallback: str = "course") -> str:
    """Turn a video filename into a safe directory name, keeping CJK intact."""
    stem = Path(name).stem
    stem = unicodedata.normalize("NFC", stem)
    stem = re.sub(r"[\/\\:\*\?\"<>\|\x00-\x1f]", "", stem)
    stem = re.sub(r"\s+", " ", stem).strip(" .")
    return stem[:80] or fallback


# --------------------------------------------------------------------------
# timestamps
# --------------------------------------------------------------------------


def fmt_ts(seconds: float) -> str:
    """12.5 -> '00:12' ; 3725 -> '1:02:05' (compact, for headings)."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def fmt_srt_ts(seconds: float, sep: str = ",") -> str:
    """12.5 -> '00:00:12,500' (SRT) or '00:00:12.500' (VTT)."""
    ms = int(round(max(0.0, seconds) * 1000))
    h, rem = divmod(ms, 3600_000)
    m, rem = divmod(rem, 60_000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}"


_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})(?:[.,](\d{1,3}))?")


def parse_ts(value: str) -> float:
    """Parse 'HH:MM:SS,mmm' / 'MM:SS' / '1:02:05.5' into seconds."""
    text = (value or "").strip()
    m = _TS_RE.fullmatch(text)
    if not m:
        raise CvnError(f"无法解析时间戳：{value!r}")
    hh = int(m.group(1) or 0)
    mm = int(m.group(2))
    ss = int(m.group(3))
    frac = (m.group(4) or "0").ljust(3, "0")
    return hh * 3600 + mm * 60 + ss + int(frac) / 1000.0


def youtube_ts(seconds: float) -> str:
    """t=1h2m5s style, handy as a clickable placeholder in Markdown."""
    total = max(0, int(round(seconds)))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    out = ""
    if h:
        out += f"{h}h"
    if h or m:
        out += f"{m}m"
    out += f"{s}s"
    return out


# --------------------------------------------------------------------------
# subprocess
# --------------------------------------------------------------------------


def which(name: str) -> str | None:
    return shutil.which(name)


def run_cmd(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run a command, capturing text output; raise CvnError with stderr tail."""
    vlog("$ " + " ".join(str(a) for a in args))
    try:
        proc = subprocess.run(
            [str(a) for a in args],
            capture_output=True,
            text=True,
            timeout=timeout,
            input=input_text,
            errors="replace",
        )
    except FileNotFoundError as exc:
        raise CvnError(f"找不到命令：{args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CvnError(f"命令超时（{timeout}s）：{' '.join(map(str, args))}") from exc
    if check and proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-6:]
        raise CvnError(
            f"命令失败（exit {proc.returncode}）：{' '.join(map(str, args))}\n" + "\n".join(tail)
        )
    return proc


def run_cmd_bytes(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    check: bool = True,
    stdin_bytes: bytes | None = None,
) -> tuple[int, bytes, str]:
    """Run a command returning raw stdout bytes plus decoded stderr.

    抽帧要读 ffmpeg 的 rawvideo 裸像素流，绝不能经过文本解码——否则
    0x80 以上的字节会被替换，灰度指纹会全部变成 0。
    """
    vlog("$ " + " ".join(str(a) for a in args))
    try:
        proc = subprocess.run(
            [str(a) for a in args],
            input=stdin_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=timeout,
        )
    except FileNotFoundError as exc:
        raise CvnError(f"找不到命令：{args[0]}") from exc
    except subprocess.TimeoutExpired as exc:
        raise CvnError(f"命令超时（{timeout}s）：{' '.join(map(str, args))}") from exc
    err = (proc.stderr or b"").decode("utf-8", "replace")
    if check and proc.returncode != 0:
        tail = err.strip().splitlines()[-6:]
        raise CvnError(
            f"命令失败（exit {proc.returncode}）：{' '.join(map(str, args))}\n" + "\n".join(tail)
        )
    return proc.returncode, proc.stdout or b"", err


def is_writable_dir(path: str | os.PathLike[str]) -> bool:
    """True if we can create a probe file in `path` (creates the dir if needed)."""
    p = Path(path)
    try:
        p.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=p, prefix=".wtest", delete=True):
            return True
    except (OSError, PermissionError):
        return False


def pick_temp_root() -> Path:
    """First writable location among $TMPDIR and /tmp."""
    for cand in (os.environ.get("TMPDIR"), tempfile.gettempdir(), "/tmp"):
        if not cand:
            continue
        p = Path(cand) / "cvn"
        if is_writable_dir(p):
            return p
    raise CvnError("找不到可写的临时目录（$TMPDIR / /tmp 都被拒绝）")


# --------------------------------------------------------------------------
# text cleanup
# --------------------------------------------------------------------------


_CJK = r"\u3400-\u4dbf\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af"

# Lines that a lecturer/slide template repeats on every page and that carry no
# teaching content: page numbers, school names, dates, "谢谢观看", copyright rows.
_WATERMARK_PATTERNS = [
    re.compile(r"^\s*[-—–]?\s*\d{1,3}\s*[-—–]?\s*$"),                    # bare page num
    re.compile(r"^\s*(第\s*\d+\s*页|共\s*\d+\s*页|Page\s*\d+.*)\s*$", re.I),
    re.compile(r"^\s*(\d{1,3}\s*/\s*\d{1,3})\s*$"),                      # 3/28
    re.compile(r"^\s*©.*$"),
    re.compile(r"^\s*(版权所有|保留所有权利|翻版必究).*$"),
    re.compile(r"^\s*(谢谢(观看|聆听|欣赏)?|感\s*谢\s*观\s*看|THE\s+END)\s*[!！.。]*\s*$", re.I),
    re.compile(r"^\s*\d{4}\s*[-/年]\s*\d{1,2}\s*[-/月]\s*\d{1,2}\s*日?\s*$"),
    re.compile(r"^\s*[\w.+-]+@[\w-]+\.[\w.]+\s*$"),
    re.compile(r"^\s*(https?://|www\.)\S+\s*$", re.I),
    # 录屏水印 / 教室编号 / 完整时间戳（RapidOCR 会把这些当正文行读出来）
    re.compile(r"^\s*x\s*ier\s*[\d\-_×x]*\s*$", re.I),
    re.compile(r"^\s*cier\s*1?\s*[x×]?\s*307\s*$", re.I),
    re.compile(r"^\s*(?!(?:我|你|他|她|它|这|那|哪|如何|什么|为什么|怎样|所谓)[^，。；：！？]{0,6}$)"
               r"[^，。；：！？,.;:!?]{0,24}(?:大学|大學|学院|學院|学校|學校|UNIVERSITY|COLLEGE|INSTITUTE)\s*$", re.I),
    re.compile(r"^\s*20\d{2}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日.*$"),
    # OCR 把装饰图案读成一串括号/符号：没有汉字、字母数字也极少
    re.compile(r"^\s*[^\w\u4e00-\u9fff]{2,}\s*$"),
]

# A slide's title/logo row often appears as one lonely short line at the top.
_TRAILING_PUNCT = "，。、；：,.;:·-—_|/\\"


def clean_line(line: str) -> str:
    """Normalise one OCR/ASR line without changing its wording."""
    s = unicodedata.normalize("NFC", line)
    s = s.replace("\u3000", " ").replace("\xa0", " ")
    s = s.replace("\ufeff", "")
    # full-width ASCII -> ASCII keeps maths readable, but leaves CJK punctuation alone
    s = s.translate(_FW_ASCII)
    s = re.sub(r"[ \t]{2,}", " ", s).strip()
    return s


_FW_ASCII = {0xFF01 + i: 0x21 + i for i in range(94)}


def is_watermark(line: str) -> bool:
    s = clean_line(line)
    if not s:
        return True
    return any(p.match(s) for p in _WATERMARK_PATTERNS)


def clean_ocr_lines(lines: Iterable[str], *, drop_watermarks: bool = True) -> list[str]:
    """Normalise, drop watermarks, and collapse duplicate consecutive lines."""
    out: list[str] = []
    for raw in lines:
        for piece in str(raw).splitlines():
            s = clean_line(piece)
            if not s:
                continue
            if drop_watermarks and is_watermark(s):
                continue
            if out and out[-1] == s:      # OCR double-read of the same row
                continue
            out.append(s)
    return out


def join_lines(lines: Sequence[str]) -> str:
    """Join OCR rows into prose: no space between CJK, space between latin words."""
    if not lines:
        return ""
    buf = lines[0]
    for nxt in lines[1:]:
        prev_ch = buf[-1:] if buf else ""
        next_ch = nxt[:1]
        if re.match(r"[%s]" % _CJK, prev_ch) and re.match(r"[%s]" % _CJK, next_ch):
            buf += nxt
        elif re.search(r"[A-Za-z0-9,.;:)\]]$", buf) and re.match(r"[A-Za-z0-9(\[]", next_ch):
            buf += " " + nxt
        else:
            buf += "" if prev_ch in _TRAILING_PUNCT else " "
            buf += nxt
    return buf.strip()


def has_cjk(text: str) -> bool:
    return bool(re.search(r"[%s]" % _CJK, text or ""))


def squash(text: str) -> str:
    """Lowercase, strip all whitespace/punct — for fuzzy equality tests."""
    s = unicodedata.normalize("NFKC", text or "").lower()
    return re.sub(r"[^0-9a-z%s]+" % _CJK, "", s)


def similarity(a: str, b: str) -> float:
    """Cheap character-level Dice coefficient on squashed text (0..1)."""
    sa, sb = squash(a), squash(b)
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    if sa == sb:
        return 1.0
    from collections import Counter

    ca, cb = Counter(sa), Counter(sb)
    overlap = sum((ca & cb).values())
    return 2.0 * overlap / (len(sa) + len(sb))


def sentences(text: str) -> list[str]:
    """Split Chinese/latin prose into sentences, keeping punctuation."""
    parts = re.split(r"(?<=[。！？!?；;])|\n+", text or "")
    return [p.strip() for p in parts if p and p.strip()]


def truncate(text: str, limit: int = 120, ellipsis: str = "…") -> str:
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    return text[: max(1, limit - 1)].rstrip() + ellipsis


def parse_time_span(value: str) -> tuple[float, float] | None:
    """Parse a CLI range like '10:00-15:30' or '600-930' into seconds."""
    m = re.fullmatch(r"\s*([\d:.,]+)\s*(?:[-~—]\s*([\d:.,]+))?\s*", value or "")
    if not m:
        return None
    start = parse_ts(m.group(1))
    end = parse_ts(m.group(2)) if m.group(2) else float("inf")
    if end <= start:
        raise CvnError(f"时间区间非法：{value!r}")
    return start, end


def stamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def human_duration(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.0f} 秒"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m} 分 {s:02d} 秒"
    h, m = divmod(m, 60)
    return f"{h} 小时 {m:02d} 分"
