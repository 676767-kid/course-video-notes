"""ocr.py —— 课件截图的文字识别（course-video-notes 的「识别层」之一）。

职责（对应 references/DATA-CONTRACT.md 第 3 节）：
    读 out_dir/frames.json 里的 slide 列表（或直接扫 out_dir/slides/*.png），
    对每张幻灯片图片做文字识别，写出 out_dir/slides.json：

        {
          "engine": "rapidocr|hybrid|vision|vlm|none",
          "slides": [
            {"index": 1, "lines": [...], "text": "...", "title": "...",
             "has_figure": false, "figure_note": "", "ocr_conf": 0.93}
          ]
        }

设计要点（给后来改代码的人）：
1. **引擎可降级**。auto 顺序：RapidOCR+Vision（hybrid）> RapidOCR > Swift Vision
   > pyobjc Vision > tesseract > none。
   一个都不可用时不是崩溃，而是 engine="none" + 每页空文本，下游 align/render
   仍然能跑（只是没有课件文字）。这样用户在任意机器上都能拿到一份「讲解稿」。
2. **批量**。macOS Vision 走 swiftc 编译的小程序，一次进程吃下所有图片路径
   （stdin 每行一个），100+ 页也不会一页一进程。pyobjc / tesseract 同理批量。
3. **增量**。默认读回已有 slides.json，只对缺的页 / 图片 mtime 变化的页重做；
   force=True 才全量重算。缓存键记录在 slides.json 每页的 `_sig` 字段里
   （下划线开头，属于契约允许的「追加字段」，不改变既有语义）。
4. **绝不抛裸 traceback**。所有失败路径都转成 util.CvnError（中文可读）或降级。
5. 只依赖标准库；Pillow / pyobjc / tesseract 都是「有则有、无则退」。

对外接口：
    run_ocr(out_dir, *, engine="auto", vlm=False, force=False) -> dict
    probe_ocr_engines() -> dict
    detect_figure(path) -> bool                       （Pillow 启发式，单独可用）
    build_title(lines, boxes) -> str                  （标题启发式，单独可用）
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

try:  # 允许 `python3 scripts/cvn/ocr.py ...` 直接跑，也允许 `from cvn import ocr`
    from . import util
    from . import classify
    from . import enhance
    from . import layout
    from . import crossfix
    from . import ocr_rapid
    from . import vlm
except ImportError:  # pragma: no cover - 仅在脚本直跑时触发
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import util  # type: ignore[no-redef]
    import classify  # type: ignore[no-redef]
    import enhance  # type: ignore[no-redef]
    import layout  # type: ignore[no-redef]
    import crossfix  # type: ignore[no-redef]
    import ocr_rapid  # type: ignore[no-redef]
    import vlm  # type: ignore[no-redef]

# --- 可选依赖：全部 try/except，缺了就走降级路径 ---------------------------

try:
    from PIL import Image  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover - 取决于机器
    Image = None  # type: ignore[assignment]
    _HAS_PIL = False

# jieba 只用来给「标题是否像人话」打分：错字标题（「天写动物最天的这别」）
# 会切出一堆零频词，正确标题不会。没有 jieba 就退回字符级判断，不阻塞主流程。
try:  # pragma: no cover - 取决于机器
    import jieba  # type: ignore

    _HAS_JIEBA = True
except Exception:  # pragma: no cover
    jieba = None  # type: ignore[assignment]
    _HAS_JIEBA = False

# pyobjc 的 Vision 绑定在部分机器上可用（有 pip 权限时 pip install pyobjc-framework-Vision）
try:  # pragma: no cover - 本机没有
    import Vision as _PyVision  # type: ignore
    import Quartz as _PyQuartz  # type: ignore

    _HAS_PYOBJC = True
except Exception:
    _PyVision = None  # type: ignore[assignment]
    _PyQuartz = None  # type: ignore[assignment]
    _HAS_PYOBJC = False


SWIFT_SOURCE = Path(__file__).resolve().parent / "vision_ocr.swift"

# 识别语言：中文简体优先。不要在 VNRecognizeTextRequest 上加 recognitionCandidates，
# 那个属性不存在（踩过坑）。
VISION_LANGUAGES = ("zh-Hans", "en-US")

# 一次喂给 Vision 程序的图片数量。太多会让 Swift 端内存里堆积结果，
# 太少又多开进程；50 张一批实测速度最好。
VISION_BATCH = 50

# 编译产物缓存目录名（放在 out_dir 下，保证可写；见 _binary_cache_dir）
CACHE_DIRNAME = ".cache"

ENGINES = ("vision", "tesseract", "none")


# ==========================================================================
# 引擎探测
# ==========================================================================


def _swiftc_path() -> str | None:
    """找 swiftc：优先 xcrun，其次 PATH，最后几个已知的固定位置。"""
    p = util.which("swiftc")
    if p:
        return p
    for cand in (
        "/usr/bin/swiftc",
        "/Library/Developer/CommandLineTools/usr/bin/swiftc",
        "/Applications/Xcode.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/swiftc",
    ):
        if Path(cand).exists():
            return cand
    return None


def probe_ocr_engines() -> dict:
    """探测本机可用的 OCR 引擎，供 CLI 的 `doctor` / 启动横幅打印。

    返回：
        {
          "engine": "vision" | "tesseract" | "none",   # auto 最终会选谁
          "vision": {"available": bool, "via": "swift"|"pyobjc"|None, "reason": str},
          "tesseract": {"available": bool, "path": str|None, "langs": [...]},
          "pillow": bool,                              # has_figure 启发式要不要跑
          "swiftc": str|None,
        }
    这个函数只做「轻量探测」，不会编译 Swift 程序（编译发生在 run_ocr 里，
    因为编译产物要落到 out_dir 的缓存目录）。
    """
    swiftc = _swiftc_path()
    rapid = ocr_rapid.probe()
    info: dict[str, Any] = {
        "engine": "none",
        "swiftc": swiftc,
        "pillow": _HAS_PIL,
        "vision": {"available": False, "via": None, "reason": ""},
        "tesseract": {"available": False, "path": None, "langs": []},
        "rapidocr": {
            "available": bool(rapid.get("available")),
            "python": rapid.get("python") or "",
            "reason": rapid.get("reason") or "",
        },
    }

    if sys.platform != "darwin":
        info["vision"]["reason"] = "Apple Vision 只在 macOS 上可用（当前系统不是 darwin）"
    elif swiftc and SWIFT_SOURCE.exists():
        info["vision"] = {"available": True, "via": "swift", "reason": f"swiftc: {swiftc}"}
    elif _HAS_PYOBJC:
        info["vision"] = {"available": True, "via": "pyobjc", "reason": "pyobjc Vision 绑定已安装"}
    elif swiftc and not SWIFT_SOURCE.exists():
        info["vision"]["reason"] = f"找不到 {SWIFT_SOURCE.name}，无法编译 Vision 小工具"
    elif not swiftc and _HAS_PYOBJC is False:
        info["vision"]["reason"] = "既没有 swiftc 也没有 pyobjc 的 Vision 绑定"

    tess = util.which("tesseract")
    if tess:
        langs: list[str] = []
        try:
            proc = subprocess.run(
                [tess, "--list-langs"], capture_output=True, text=True, timeout=10, errors="replace"
            )
            langs = [
                ln.strip()
                for ln in (proc.stdout or "").splitlines()[1:]
                if ln.strip() and not ln.startswith("List of")
            ]
        except Exception:  # 探测失败不影响主流程
            langs = []
        info["tesseract"] = {"available": True, "path": tess, "langs": langs}

    if info["rapidocr"]["available"]:
        info["engine"] = "rapidocr"
    elif info["vision"]["available"]:
        info["engine"] = "vision"
    elif info["tesseract"]["available"]:
        info["engine"] = "tesseract"
    return info


# ==========================================================================
# Swift Vision 小工具的编译与缓存
# ==========================================================================


def _binary_cache_dir(out_dir: Path) -> Path:
    """选一个可写目录放编译产物。

    顺序有讲究：**优先用跨课程共享的临时目录**，其次才是 out_dir/.cache。
    因为首次编译要给 Vision/AppKit 建模块缓存（冷启动约 50 秒，热了约 1 秒），
    如果按课程目录各存一份，用户每换一门课就要再等 50 秒。

    ~/.cache 与 ~/Library/Caches 在很多沙箱/受管 Mac 上不可写（本机就是），
    所以候选顺序是：$TMPDIR/cvn/ocr-bin -> /tmp/cvn-ocr-bin -> out_dir/.cache。
    全不可写就抛 CvnError（而不是等 swiftc 报一句难懂的话）。
    """
    candidates: list[Path | None] = []
    if _safe_temp_root():
        candidates.append(util.pick_temp_root() / "ocr-bin")
    candidates.extend(
        [
            Path("/tmp/cvn-ocr-bin"),  # 显式兜底
            out_dir / CACHE_DIRNAME,
        ]
    )
    for cand in candidates:
        if cand is not None and util.is_writable_dir(cand):
            return cand
    raise util.CvnError(
        "找不到可写的目录来放 OCR 编译产物（试过 $TMPDIR、/tmp、out_dir/.cache）"
    )


def _safe_temp_root() -> bool:
    """pick_temp_root 会抛错，这里先探一下，避免把异常提前炸出来。"""
    try:
        util.pick_temp_root()
        return True
    except util.CvnError:
        return False


def _source_signature() -> str:
    """Swift 源码的指纹：源码改了就要重新编译。"""
    try:
        st = SWIFT_SOURCE.stat()
        return f"{st.st_size}-{int(st.st_mtime)}"
    except OSError:
        return "0-0"


def ensure_swift_binary(out_dir: Path, *, force: bool = False) -> Path:
    """确保 Vision 批量 OCR 小程序已编译，返回可执行文件路径。

    编译策略：
      - 产物放在 out_dir/.cache/vision_ocr_<源码指纹>，源码不变就复用；
      - swiftc 第一个 -O，另给 -module-cache-path 指到可写目录
        （默认的 module cache 也在 ~/Library/Caches，同样可能不可写）；
      - 编译失败只报一次可读错误，不打印整段 Swift 编译日志（太吵），
        但把前几行错误压进 message 里方便排查。
    """
    binary = _binary_cache_dir(out_dir) / f"vision_ocr_{_source_signature()}"
    if binary.exists() and not force:
        return binary

    if not SWIFT_SOURCE.exists():
        raise util.CvnError(f"缺少 Swift 源码：{SWIFT_SOURCE}（随 skill 一起分发，请勿删除）")
    swiftc = _swiftc_path()
    if not swiftc:
        raise util.CvnError(
            "找不到 swiftc，无法编译 Apple Vision OCR 小工具。\n"
            "解决办法：安装 Xcode Command Line Tools（xcode-select --install），"
            "或改用 --ocr-engine tesseract。"
        )

    util.ensure_dir(binary.parent)
    module_cache = binary.parent / "swift-module-cache"
    util.ensure_dir(module_cache)

    if binary.exists():
        binary.unlink()
    util.log("  · 首次使用，正在编译 Vision OCR 小工具（第一次约 1 分钟，之后秒开）…")
    args = [
        swiftc,
        "-O",
        "-module-cache-path", str(module_cache),
        "-o", str(binary),
        str(SWIFT_SOURCE),
    ]
    started = time.time()
    proc = subprocess.run(args, capture_output=True, text=True, errors="replace")
    if proc.returncode != 0 or not binary.exists():
        detail = "\n".join((proc.stderr or proc.stdout or "").strip().splitlines()[:8])
        raise util.CvnError(
            "编译 Apple Vision OCR 小工具失败（swiftc 返回 "
            f"{proc.returncode}）。\n原始错误前几行：\n{detail}\n"
            "排查建议：xcode-select --install 后重试；或改用 --ocr-engine tesseract。"
        )
    util.vlog(f"  · 编译完成，用时 {time.time() - started:.1f}s -> {binary}")
    return binary


# ==========================================================================
# 三种引擎的批量识别实现
# 统一返回：{图片路径(str): [{"text","conf","x","y","w","h"}, ...]}
# 坐标归一化到 0..1，原点左上（y 自上而下）；拿不到版面信息时坐标全 0。
# ==========================================================================


def _run_vision_swift(
    binary: Path, paths: Sequence[Path], *, fast: bool = False
) -> dict[str, list[dict[str, Any]]]:
    """用编译好的 Swift 程序批量识别。一次进程处理 `paths` 里的全部图片。"""
    if not paths:
        return {}
    stdin_text = "\n".join(str(p) for p in paths) + "\n"
    cmd = [str(binary), "--languages", ",".join(VISION_LANGUAGES)]
    if fast:
        cmd.append("--fast")

    # 单批超时按「每张最多 10 秒 + 60 秒启动」估算，避免某些病态图片卡死整条流水线
    timeout = 60 + 10 * len(paths)
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise util.CvnError(
            f"Apple Vision 识别超时（{timeout}s，{len(paths)} 张图）。"
            "如果是超大分辨率截图，可在 frames 阶段降低输出分辨率。"
        ) from exc

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        raise util.CvnError(f"Vision OCR 进程失败（exit {proc.returncode}）\n{tail}")

    results: dict[str, list[dict[str, Any]]] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            util.vlog(f"  · 跳过无法解析的 Vision 输出行：{line[:80]}")
            continue
        path = str(obj.get("file") or "")
        if not path:
            continue
        if obj.get("error"):
            util.warn(f"  图片无法识别（{obj['error']}）：{path}")
            results[path] = []
            continue
        results[path] = _normalise_boxes(obj.get("lines") or [])
    # 输入里有、输出里没有的（理论上不该发生）当空页处理，别让下游 KeyError
    for p in paths:
        results.setdefault(str(p), [])
    return results


def _run_vision_pyobjc(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
    """pyobjc 版 Vision：接口和 Swift 一样，只是从 Python 调。"""
    if not _HAS_PYOBJC:  # pragma: no cover - 防御
        raise util.CvnError("pyobjc 的 Vision 绑定不可用")
    results: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        lines: list[dict[str, Any]] = []
        try:
            url = _PyQuartz.NSURL.fileURLWithPath_(str(path))  # type: ignore[attr-defined]
            src = _PyQuartz.CGImageSourceCreateWithURL(url, None)
            cg = _PyQuartz.CGImageSourceCreateImageAtIndex(src, 0, None) if src else None
            if cg is None:
                util.warn(f"  图片无法读取：{path}")
                results[str(path)] = []
                continue
            req = _PyVision.VNRecognizeTextRequest.alloc().init()
            req.setRecognitionLevel_(0)  # 0 = accurate
            req.setRecognitionLanguages_(list(VISION_LANGUAGES))
            req.setUsesLanguageCorrection_(True)
            handler = _PyVision.VNImageRequestHandler.alloc().initWithCGImage_options_(cg, {})
            ok, _err = handler.performRequests_error_([req], None)
            if not ok:
                util.warn(f"  图片识别失败：{path}")
                results[str(path)] = []
                continue
            for obs in req.results() or []:
                cands = obs.topCandidates_(1)
                if not cands:
                    continue
                cand = cands[0]
                bb = obs.boundingBox()
                lines.append(
                    {
                        "text": str(cand.string()),
                        "conf": float(cand.confidence()),
                        "x": float(bb.origin.x),
                        "y": float(1.0 - bb.origin.y - bb.size.height),
                        "w": float(bb.size.width),
                        "h": float(bb.size.height),
                    }
                )
        except Exception as exc:  # 单张失败不影响整批
            util.warn(f"  图片识别异常（跳过）：{path}：{exc}")
            lines = []
        results[str(path)] = _normalise_boxes(lines)
    return results


def _tesseract_langs(tess: str, available: list[str] | None = None) -> str:
    """挑 tesseract 的语言参数：有中文包用 chi_sim+eng，否则退回 eng。"""
    langs = available
    if langs is None:
        langs = probe_ocr_engines()["tesseract"]["langs"]
    picked = [code for code in ("chi_sim", "eng") if code in (langs or [])]
    return "+".join(picked) if picked else "eng"


def _run_tesseract(paths: Sequence[Path], *, tess: str | None = None) -> dict[str, list[dict[str, Any]]]:
    """tesseract 批量识别。

    tesseract 没有「一次进程多张图」的模式，这里做的是「一个 Python 进程
    循环调用」，比一张一进程好在只导入一次、也只探测一次语言包。
    用 TSV 输出拿版面框（-c tessedit_create_tsv=1 的 stdout 版本）。
    """
    tess = tess or util.which("tesseract")
    if not tess:
        raise util.CvnError("找不到 tesseract 命令")
    langs = _tesseract_langs(tess)
    results: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        lines: list[dict[str, Any]] = []
        try:
            proc = subprocess.run(
                [tess, str(path), "stdout", "-l", langs, "--psm", "3", "tsv"],
                capture_output=True,
                text=True,
                timeout=120,
                errors="replace",
            )
            if proc.returncode != 0:
                util.warn(f"  tesseract 识别失败（跳过）：{path}")
                results[str(path)] = []
                continue
            lines = _parse_tesseract_tsv(proc.stdout or "")
        except subprocess.TimeoutExpired:
            util.warn(f"  tesseract 识别超时（跳过）：{path}")
        except Exception as exc:
            util.warn(f"  tesseract 识别异常（跳过）：{path}：{exc}")
        results[str(path)] = lines
    return results


def _parse_tesseract_tsv(tsv: str) -> list[dict[str, Any]]:
    """把 tesseract 的 TSV 输出按 (block, par, line) 聚成行，附归一化版面框。

    TSV 列：level page_num block_num par_num line_num word_num
            left top width height conf text
    """
    rows = [ln for ln in (tsv or "").splitlines() if ln.strip()]
    if not rows:
        return []
    header = rows[0].split("\t")
    try:
        idx = {name: header.index(name) for name in
               ("level", "block_num", "par_num", "line_num", "left", "top", "width", "height", "conf", "text")}
    except ValueError:
        return []

    words: list[dict[str, Any]] = []
    max_x, max_y = 0.0, 0.0
    for row in rows[1:]:
        cols = row.split("\t")
        if len(cols) <= max(idx.values()):
            continue
        try:
            level = int(cols[idx["level"]])
        except ValueError:
            continue
        if level != 5:  # 只要词级别
            continue
        text = cols[idx["text"]].strip()
        if not text:
            continue
        try:
            left = float(cols[idx["left"]]); top = float(cols[idx["top"]])
            width = float(cols[idx["width"]]); height = float(cols[idx["height"]])
            conf = float(cols[idx["conf"]])
        except ValueError:
            continue
        key = (cols[idx["block_num"]], cols[idx["par_num"]], cols[idx["line_num"]])
        words.append(
            {"key": key, "text": text, "conf": conf, "left": left, "top": top,
             "width": width, "height": height}
        )
        max_x = max(max_x, left + width)
        max_y = max(max_y, top + height)

    if not words:
        return []
    max_x = max_x or 1.0
    max_y = max_y or 1.0

    grouped: dict[tuple, list[dict[str, Any]]] = {}
    for w in words:
        grouped.setdefault(w["key"], []).append(w)

    out: list[dict[str, Any]] = []
    for key in sorted(grouped, key=lambda k: (int(k[0]), int(k[1]), int(k[2]))):
        ws = sorted(grouped[key], key=lambda w: w["left"])
        text = " ".join(w["text"] for w in ws)
        left = min(w["left"] for w in ws)
        top = min(w["top"] for w in ws)
        right = max(w["left"] + w["width"] for w in ws)
        bottom = max(w["top"] + w["height"] for w in ws)
        confs = [w["conf"] for w in ws if w["conf"] >= 0]
        out.append(
            {
                "text": text,
                "conf": (sum(confs) / len(confs) / 100.0) if confs else 0.0,
                "x": left / max_x,
                "y": top / max_y,
                "w": (right - left) / max_x,
                "h": (bottom - top) / max_y,
            }
        )
    return _normalise_boxes(out)


def _normalise_boxes(lines: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    """统一版面框：夹到 0..1，缺的补 0，去掉空文本。

    不同引擎对「没有版面信息」的表达不一样（缺省 / 0 / None），下游只认
    x/y/w/h 四个 float，所以在这里一次性抹平。
    """
    out: list[dict[str, Any]] = []
    for item in lines:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "")
        if not text.strip():
            continue
        def _f(name: str) -> float:
            try:
                v = float(item.get(name) or 0.0)
            except (TypeError, ValueError):
                return 0.0
            return min(1.0, max(0.0, v))
        try:
            conf = float(item.get("conf") or 0.0)
        except (TypeError, ValueError):
            conf = 0.0
        out.append(
            {
                "text": text,
                "conf": min(1.0, max(0.0, conf)),
                "x": _f("x"),
                "y": _f("y"),
                "w": _f("w"),
                "h": _f("h"),
            }
        )
    return out


# ==========================================================================
# 版面 -> 行序 -> 标题
# ==========================================================================


def _sort_boxes(lines: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """按阅读序（自上而下、同一行内自左而右）排序。

    Vision 返回的观测顺序并不保证是阅读序（多栏版面尤其乱），所以自己排。
    同一行判定：纵向中心差 < 行高的一半。
    """
    if not lines:
        return []
    def cy(b: dict[str, Any]) -> float:
        return b["y"] + b["h"] / 2.0

    ordered = sorted(lines, key=lambda b: (b["y"], b["x"]))
    rows: list[list[dict[str, Any]]] = []
    for box in ordered:
        if rows:
            last = rows[-1]
            ref = last[0]
            tol = max(0.012, (ref["h"] + box["h"]) / 3.0)  # 归一化坐标下约「半行高」
            if abs(cy(box) - cy(ref)) <= tol:
                last.append(box)
                continue
        rows.append([box])
    out: list[dict[str, Any]] = []
    for row in rows:
        out.extend(sorted(row, key=lambda b: b["x"]))
    return out


def _merge_row_text(row: Sequence[dict[str, Any]]) -> str:
    """把同一视觉行内的多个观测拼成一行文本。

    中文之间不加空格，拉丁词之间加空格——和 util.join_lines 保持一致的口径，
    但这里处理的是「同一行被切成好几段」的情况，所以间距用版面距离判断。
    """
    if not row:
        return ""
    parts: list[str] = []
    prev: dict[str, Any] | None = None
    for box in sorted(row, key=lambda b: b["x"]):
        text = str(box.get("text") or "").strip()
        if not text:
            continue
        if prev is not None:
            gap = box["x"] - (prev["x"] + prev["w"])
            # 归一化后 gap > 0.01 大致相当于一个空格；中文直接粘
            if gap > 0.01 and not (util.has_cjk(parts[-1][-1:]) and util.has_cjk(text[:1])):
                parts.append(" ")
        parts.append(text)
        prev = box
    return util.clean_line("".join(parts))


def _as_text(item: Any) -> str:
    """把「一行」的任意常见表示收窄成 str。

    OCR 结果里的行可能是 str，也可能是带 text 字段的 dict（Vision / 外部 VLM
    都可能直接给我们原始框）。util.clean_line 只吃 str，传 dict 会抛
    `TypeError: normalize() argument 2 must be str`——这里统一挡住。
    """
    if isinstance(item, str):
        return item
    if isinstance(item, dict):
        for key in ("text", "line", "content"):
            value = item.get(key)
            if isinstance(value, str):
                return value
        return ""
    if item is None:
        return ""
    return str(item)


# 页眉/页脚特征：日期时间戳、机房号、页码
_PAGE_CHROME_RE = re.compile(
    r"^\s*(?:\d{4}\s*年\s*(?:\d{1,2}\s*月)?\s*(?:\d{1,2}\s*日)?"
    r"|星期[一二三四五六日]"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|[a-zA-Z]{2,6}\s*[_\-]\s*[xX×]?\s*\d{2,4}"
    r"|\d{1,3}\s*$"
    r"|[a-zA-Z]{2,8}\d{1,4}[_\-][a-zA-Z]?\d{2,4})"
)

# 标题行的最低置信度：低于此值说明这行没认准（艺术字/水印干扰），
# 宁可让标题留空，也不要在目录里挂一串乱码。
_TITLE_MIN_CONF = 0.5

# 行内页眉片段：日期时间戳、机房号、水印
_CHROME_FRAG_RE = re.compile(
    r"(?:\d{4}\s*年\s*\d{1,2}\s*月\s*\d{1,2}\s*日"
    r"|星期[一二三四五六日]"
    r"|\d{1,2}:\d{2}(?::\d{2})?"
    r"|[xX]\s*ier\s*\d*"
    r"|\s*[|｜]\s*English\s*Version"
    r"|\s*[|｜]\s*首页\s*[|｜]?\s*English)"
)


def _strip_chrome(text: str) -> str:
    """去掉一行文字里的页眉片段（日期时间戳、校名、机房号）。"""
    cleaned = _CHROME_FRAG_RE.sub(" ", text)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip(" |｜\t")
    return cleaned


def build_title(lines: Sequence[Any], boxes: Sequence[dict[str, Any]] | None = None) -> str:
    """挑出最像标题的那一行。

    `lines` 可以混着 str 和 {"text": ...} 形式的 dict（内部用 _as_text 收窄）。

    启发式（分数越高越像标题，都不满足就返回 ""）：
      + 位置越靠上分越高（标题几乎总在第一屏上部）
      + 字号（bbox 高度）越大分越高
      + 越短越像标题（正文段落动辄几十字）
      + 含「第X章 / 第X节 / 一、 / 1.」这类章节编号额外加分
      - 以冒号/逗号结尾（更像正文的一截）
    """
    if not lines:
        return ""
    boxes = [b for b in (boxes or []) if isinstance(b, dict)]
    height_of = [float(b.get("h") or 0.0) for b in boxes]
    max_h = max(height_of) if height_of else 0.0

    best, best_score = "", -1e9
    for i, raw in enumerate(lines):
        text = util.clean_line(_as_text(raw))
        if not text:
            continue
        box = boxes[i] if i < len(boxes) else {}
        h = float(box.get("h") or 0.0)
        y = float(box.get("y") or 0.0)
        # 页眉/页脚（日期时间戳、机房号水印）永远不是标题
        if _PAGE_CHROME_RE.match(text):
            continue
        # 一行里混着页眉和正文（如「202m食閉怎科天?:0:31 某某大学」）时，
        # 把页眉那截去掉，剩下的才是标题内容
        stripped = _strip_chrome(text)
        if stripped and stripped != text:
            text = stripped
        # 置信度过低的行多半是艺术字/水印，识别成乱码，不适合当标题
        conf = float(box.get("conf") or 1.0)
        if conf < _TITLE_MIN_CONF:
            continue
        score = 0.0
        score += max(0.0, 1.0 - y * 2.0) * 3.0          # 靠上
        if max_h > 0:
            score += (h / max_h) * 3.0                   # 字号大
        n = len(text)
        if n <= 18:
            score += 1.2
        elif n <= 30:
            score += 0.4
        else:
            score -= 1.0                                 # 太长不像标题
        if re.search(r"(第\s*[0-9一二三四五六七八九十百]+\s*[章节讲部篇课])", text):
            score += 1.5
        if re.match(r"^\s*([一二三四五六七八九十]+[、.)]|[0-9]+[、.)]\s*\S)", text):
            score += 0.6
        if text.endswith((":", "：", ",", "，", ";", "；")):
            score -= 1.0
        if text.endswith(("?", "？")):
            score -= 0.3
        if util.has_cjk(text) and n >= 3:
            score += 0.3                                 # 中文短语优先于纯符号
        if score > best_score:
            best_score, best = score, text
    return util.truncate(best, 60) if best_score > 0 else (util.truncate(best, 60) if best else "")


# ==========================================================================
# has_figure：Pillow 启发式
# ==========================================================================

# 公式/图表里常见的非汉字符号：它们密集出现通常意味着这一页有图或公式
_FIGURE_SYMBOLS = set("=≈≠≤≥∑∫√±×÷→∞∂∆∇^_{}[]<>|/\\%‰°αβγδεθλμπρσφψωΩΔ")

# 「文字框之外的墨迹」占全图采样点的比例阈值。
# 实测（1280x720 课件图）：纯文字页 0.0050，带电机示意图的页 0.0178，
# 取 0.012 正好卡在中间（差值 3.5 倍，不敏感）。
_FIGURE_INK_OUTSIDE = 0.012

# 墨迹判定：灰度 >= 此值算纯背景，跳过。232 能吃掉 JPEG 噪点和浅灰底纹，
# 但保留中文抗锯齿的深色边缘。
_INK_BG_LEVEL = 232

# 归一化坐标下给 OCR 框加的外扩边距：中文的字身通常顶到 bbox 边缘，
# 外扩 0.004 让抗锯齿毛边算作「框内」，避免纯文字页被误判。
_BOX_PAD = 0.004


def detect_figure(
    path: str | os.PathLike[str],
    *,
    text_boxes: Sequence[dict[str, Any]] | None = None,
) -> bool:
    """判断一页幻灯片里是否有图表 / 公式区域（启发式，宁可少报不可乱报）。

    核心判据：**OCR 文字框覆盖不到的墨迹占比**。

    为什么用这个而不是「全图彩色占比」：中文抗锯齿会在文字边缘产生大量中间灰度
    像素，纯文字页的「非背景像素」往往并不少（实测 4.3%），和真图页（5.8%）
    几乎不可分。但只要把「已被 OCR 文字框覆盖的区域」减掉，差异立刻拉开 3.5 倍：
    文字页的墨迹基本全被框吸收（0.005），示意图的线条没有对应文字框（0.018）。

    具体信号：
      1. 框外墨迹 > 1.2%  -> 有图/公式/示意图
      2. 公式符号密度 > 12% -> 有公式（OCR 文本里 = ∑ ∫ √ 等占比）
         第 2 条独立成立，因为一行公式的墨迹仍可能落在它自己的文字框里。

    没有 Pillow 时只看第 2 条（用 OCR 文本），仍能给出保守结论；
    没有 boxes 时也不做图像判据（否则等于全图墨迹率，正是踩过的坑）。

    参数：
        path:       幻灯片图片路径
        text_boxes: 该页 OCR 返回的框（含 x/y/w/h，归一化 0..1）。
                    必须传真实 OCR 结果，不要自己猜。
    """
    boxes = [b for b in (text_boxes or []) if isinstance(b, dict)]

    # ---- 信号 2：公式符号密度（不依赖 Pillow）----
    text = "".join(str(b.get("text") or "") for b in boxes)
    compact = re.sub(r"\s+", "", text)
    if compact:
        symbol_hits = sum(1 for ch in compact if ch in _FIGURE_SYMBOLS)
        if symbol_hits / len(compact) > 0.12:
            return True

    if not _HAS_PIL or not boxes:
        return False

    # ---- 信号 1：框外墨迹占比 ----
    try:
        return _ink_outside_ratio(path, boxes) > _FIGURE_INK_OUTSIDE
    except Exception as exc:  # 图片坏了不要紧，has_figure 只是加分信息
        util.vlog(f"  · has_figure 读图失败（当作无图）：{path}：{exc}")
        return False


def _ink_outside_ratio(path: str | os.PathLike[str], boxes: Sequence[dict[str, Any]]) -> float:
    """计算「不在任何 OCR 文字框内的墨迹像素」占全图采样点的比例。

    实现细节：
      - 转灰度，按 step=4 降采样（1280x720 只要 5.8 万次采样，约 20ms）；
      - 灰度 >= 232 视为背景直接跳过；
      - 把采样点坐标归一化后依次比对 OCR 框（框已外扩 _BOX_PAD）。
    框数量通常 < 20，线性比对完全够快，不引入 numpy。
    """
    step = 4
    with Image.open(path) as im:  # type: ignore[union-attr]
        gray = im.convert("L")
        width, height = gray.size
        if width <= 0 or height <= 0:
            return 0.0
        px = gray.load()
        # 预先算好每个框的实际边界，循环里只做比较
        rects = [
            (
                float(b.get("x") or 0.0) - _BOX_PAD,
                float(b.get("y") or 0.0) - _BOX_PAD,
                float(b.get("x") or 0.0) + float(b.get("w") or 0.0) + _BOX_PAD,
                float(b.get("y") or 0.0) + float(b.get("h") or 0.0) + _BOX_PAD,
            )
            for b in boxes
        ]
        sampled = 0
        outside = 0
        for y in range(0, height, step):
            ny = y / height
            for x in range(0, width, step):
                sampled += 1
                if px[x, y] >= _INK_BG_LEVEL:
                    continue
                nx = x / width
                for (x0, y0, x1, y1) in rects:
                    if x0 <= nx <= x1 and y0 <= ny <= y1:
                        break
                else:
                    outside += 1
    return (outside / sampled) if sampled else 0.0


# ==========================================================================
# 单页结果组装
# ==========================================================================


def _clean_page_boxes(boxes: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """清掉录屏软件压在课件顶部的页眉（日期/时间/机房号/校名）。

    录屏类素材的顶部常有一条半透明时间戳，OCR 会把它和标题并成一行：
        「2026年0核心问提是什么?:10:31 某某大学」
    这不是课件内容，而且会把标题挤成乱码。处理分三步：
      1. 只对页面顶部 14% 的框动手（正文里的时间不碰）；
      2. 如果一行以 20xx 开头、并且日期片段后面还有 2 个以上连续汉字，
         就从那串汉字开始截断（"2026年0核心问题是什么" -> "核心问题是什么"）；
      3. 截完只剩日期/星期/机房的，整框丢掉。
    """
    out: list[dict[str, Any]] = []
    for box in boxes:
        text = util.clean_line(str(box.get("text") or ""))
        if not text:
            continue
        y = float(box.get("y") or 0.0)
        if y < 0.14:
            # 顶部残留的日期/星期/校名碎片：只剩 1 个字符、或「数字+月日」
            # 但几乎没有正文的，直接丢。
            if len(text) <= 1:
                continue
            if re.match(r"^20\d{2}.*[月日]", text) and len(_cjk_only(text)) <= 6:
                continue
            if re.match(r"^星[一二三四五六日天]", text):
                continue
            if re.fullmatch(r"[A-Z'’\s]{2,}", text):
                continue
            m = re.search(r"[\u4e00-\u9fff]{2,}", text)
            if m and m.start() > 0 and re.match(r"^20\d{2}", text):
                prefix = text[: m.start()]
                if re.search(r"[年月日]", prefix):
                    text = text[m.start() :]
            text = _strip_chrome(text).strip()
        if not text or (
            _PAGE_CHROME_RE.match(text) and (y < 0.14 or len(text) <= 12)
        ):
            continue
        useful = sum(1 for ch in text if ch.isalnum() or "\u4e00" <= ch <= "\u9fff")
        if useful < 2:
            continue
        item = dict(box)
        item["text"] = text
        out.append(item)
    return out


def _assemble_slide(
    index: int,
    boxes: Sequence[dict[str, Any]],
    image_path: Path,
    *,
    want_figure: bool = True,
) -> dict[str, Any]:
    """把一页的原始 OCR 框整理成契约里的一条 slide 记录。

    顺序：版面排序 -> 同一视觉行合并 -> util.clean_ocr_lines（去水印/去重）-> join_lines。
    注意 clean_ocr_lines 会丢行，所以标题要在「清洗后的行」+「对应框」上算，
    这里用一个简单的对齐：按清洗后的文本顺序回填它第一次出现时的框。
    """
    ordered = _sort_boxes(_clean_page_boxes(boxes))
    # 先按视觉行合并成「行 + 该行的代表框」
    rows: list[tuple[str, dict[str, Any]]] = []
    buf: list[dict[str, Any]] = []
    for box in ordered:
        if buf:
            ref = buf[0]
            tol = max(0.012, (ref["h"] + box["h"]) / 3.0)
            gap = box["x"] - max(b["x"] + b["w"] for b in buf)
            if (
                abs((box["y"] + box["h"] / 2) - (ref["y"] + ref["h"] / 2)) <= tol
                and gap <= 0.05
            ):
                buf.append(box)
                continue
            rows.append((_merge_row_text(buf), _union_box(buf)))
            buf = [box]
        else:
            buf = [box]
    if buf:
        rows.append((_merge_row_text(buf), _union_box(buf)))

    raw_lines = [text for text, _ in rows]
    box_by_text: dict[str, dict[str, Any]] = {}
    for text, box in rows:
        box_by_text.setdefault(util.clean_line(text), box)

    cleaned = util.clean_ocr_lines(raw_lines)
    key_boxes = [box_by_text.get(line, {}) for line in cleaned]
    text = util.join_lines(cleaned)
    title = build_title(cleaned, key_boxes)

    confs = [float(b.get("conf") or 0.0) for b in ordered if float(b.get("conf") or 0.0) > 0]
    ocr_conf = round(sum(confs) / len(confs), 4) if confs else 0.0

    has_figure = detect_figure(image_path, text_boxes=ordered) if want_figure else False

    # 判断这一帧是「真课件」还是桌面/浏览器/编辑器等界面画面。
    # 录屏类视频里这些非课件画面占很大比重，不区分会污染笔记。
    kind_info = classify.classify_page(cleaned)

    # 按版面把「视觉行」重建成「带角色的正文块」：
    # 去页眉页脚、接回被排版切断的句子、区分标题/小标题/正文/引文/出处。
    # 这是「把 PPT 扒下来」的核心——用户要的是课件上写了什么，
    # 而不是一堆夹杂时间戳和水印的行。
    try:
        blocks = layout.rebuild([(t, b) for t, b in rows])
    except Exception as exc:  # noqa: BLE001 - 版面重建失败不该拖垮整页
        util.vlog(f"  · 第 {index} 页版面重建失败，退回纯行文本：{exc}")
        blocks = [{"role": "body", "text": t} for t in cleaned]

    return {
        "index": index,
        "lines": cleaned,
        "blocks": blocks,
        "text": text,
        "title": title,
        "has_figure": bool(has_figure),
        "figure_note": "",
        "ocr_conf": ocr_conf,
        "kind": kind_info.get("kind", "unknown"),
        "kind_reason": kind_info.get("reason", ""),
    }


def _union_box(boxes: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """同一视觉行多个观测的并集框（用于标题字号估计）。

    同时带上该行的平均置信度——下游做跨页纠错要靠**逐行**置信度判断
    哪个版本的写法更可信（用整页平均会把错版当权威，纠错方向会反）。
    """
    if not boxes:
        return {}
    x0 = min(b["x"] for b in boxes)
    y0 = min(b["y"] for b in boxes)
    x1 = max(b["x"] + b["w"] for b in boxes)
    y1 = max(b["y"] + b["h"] for b in boxes)
    confs = [float(b.get("conf") or 0.0) for b in boxes if float(b.get("conf") or 0.0) > 0]
    return {
        "x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0,
        "conf": (sum(confs) / len(confs)) if confs else 0.0,
    }


# ==========================================================================
# 幻灯片清单
# ==========================================================================


_SLIDE_NAME_RE = re.compile(r"slide[_-]?(\d+)\.(png|jpe?g|webp)$", re.I)


def collect_slides(out_dir: Path) -> list[dict[str, Any]]:
    """汇总要识别的幻灯片列表：[{"index", "file"(绝对路径), "rel"(写入契约的相对路径)}]。

    优先用 frames.json 的顺序（契约规定的权威 index），其次扫 slides 目录。
    图片缺失时不是直接报错——单独记下来，由 run_ocr 决定是跳过还是报错。
    """
    slides_dir = out_dir / "slides"
    frames = util.read_json(out_dir / "frames.json", default=None)

    entries: list[dict[str, Any]] = []
    if isinstance(frames, dict) and isinstance(frames.get("slides"), list):
        for pos, item in enumerate(frames["slides"], start=1):
            if not isinstance(item, dict):
                continue
            index = item.get("index")
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = pos
            raw_file = str(item.get("file") or "")
            if not raw_file:
                raw_file = f"slides/slide_{index:04d}.png"
            path = (out_dir / raw_file) if not os.path.isabs(raw_file) else Path(raw_file)
            entries.append({"index": index, "file": path, "rel": raw_file})
    else:
        util.vlog("  · 没有 frames.json，改为扫描 slides/ 目录")
        if not slides_dir.exists():
            return []
        found: list[tuple[int, Path]] = []
        for p in slides_dir.iterdir():
            m = _SLIDE_NAME_RE.fullmatch(p.name)
            if m:
                found.append((int(m.group(1)), p))
        for index, p in sorted(found):
            entries.append({"index": index, "file": p, "rel": f"slides/{p.name}"})
    return entries


def _file_signature(path: Path) -> str:
    """增量判断用的指纹：大小 + mtime。文件不存在返回 ""。"""
    try:
        st = path.stat()
        return f"{st.st_size}:{int(st.st_mtime)}"
    except OSError:
        return ""


# ==========================================================================
# 主入口
# ==========================================================================


def run_ocr(
    out_dir: str | os.PathLike[str],
    *,
    engine: str = "auto",
    vlm: bool = False,
    force: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> dict:
    """识别 out_dir/slides/ 下所有幻灯片图片，写出 out_dir/slides.json。

    参数：
        out_dir: 课程输出目录（含 frames.json / slides/）
        engine:  "auto" | "vision" | "tesseract" | "none"
                 auto 依次尝试 Swift Vision -> pyobjc Vision -> tesseract -> none
        vlm:     True 时尝试用视觉大模型补充 has_figure / figure_note。
                 **绝不改写 OCR 原文**（契约要求 lines/text/title 来自 OCR 引擎）。
                 当前版本没有内置 VLM 客户端，会打一条提示后原样跳过；
                 外部可通过环境变量 CVN_VLM_CMD 注入一个「读图返回说明」的命令。
        force:   True = 忽略缓存全部重做；False = 只补缺页 / 图片变了的页
        limit:   只处理前 N 页（调试用）
        dry_run: 只打印将要做什么，不写文件

    返回：写出的 slides.json 内容（dict）。
    异常：util.CvnError —— 用户可见的中文错误。
    """
    root = Path(out_dir)
    if not root.exists():
        raise util.CvnError(f"输出目录不存在：{root}")
    util.ensure_dir(root)

    engine = (engine or "auto").strip().lower()
    if engine not in ("auto", "rapidocr", "hybrid", "vision", "pyobjc", "tesseract", "none"):
        raise util.CvnError(
            f"未知的 OCR 引擎：{engine}"
            "（可选 auto/rapidocr/hybrid/vision/tesseract/none）"
        )

    entries = collect_slides(root)
    if not entries:
        raise util.CvnError(
            f"在 {root} 里找不到任何幻灯片。\n"
            "请先跑抽帧阶段（frames）生成 frames.json 与 slides/slide_0001.png 等文件。"
        )

    missing = [e for e in entries if not e["file"].exists()]
    if missing and len(missing) == len(entries):
        preview = "、".join(str(e["file"].name) for e in missing[:3])
        raise util.CvnError(
            f"{len(missing)} 张幻灯片图片全部缺失（例如 {preview}）。\n"
            "frames.json 说有这些页，但 slides/ 目录里没有对应文件，请重新抽帧。"
        )
    for e in missing:
        util.warn(f"  图片缺失，跳过该页（保留空页占位）：{e['file']}")
    present = [e for e in entries if e["file"].exists()]
    if limit is not None and limit > 0:
        present = present[:limit]

    util.banner(f"OCR 识别  共 {len(present)} 页")

    # ---- 选引擎 ----
    selected, runner = _select_engine(root, engine)
    util.log(f"  · 引擎：{selected}")

    # ---- 读旧结果做增量 ----
    old = util.read_json(root / "slides.json", default=None)
    old_by_index: dict[int, dict[str, Any]] = {}
    if isinstance(old, dict) and isinstance(old.get("slides"), list) and not force:
        for item in old["slides"]:
            if isinstance(item, dict) and item.get("index") is not None:
                try:
                    old_by_index[int(item["index"])] = item
                except (TypeError, ValueError):
                    continue

    todo: list[dict[str, Any]] = []
    reused: dict[int, dict[str, Any]] = {}
    for e in present:
        sig = _file_signature(e["file"])
        prev = old_by_index.get(e["index"])
        if (
            prev is not None
            and not force
            and prev.get("_sig") == sig
            and isinstance(prev.get("lines"), list)
        ):
            reused[e["index"]] = prev
        else:
            todo.append(e)

    if reused:
        util.log(f"  · 跳过已识别的 {len(reused)} 页（--force 可强制重算）")
    util.log(f"  · 本次需要识别 {len(todo)} 页")

    if dry_run:
        util.log("  · --dry-run：不执行识别、不写文件")
        return {"engine": selected, "slides": [], "dry_run": True, "todo": len(todo)}

    # ---- 执行 ----
    new_results: dict[int, dict[str, Any]] = {}
    if todo and selected != "none":
        boxes_by_page = _recognise_multipass(
            runner,
            [e["file"] for e in todo],
            root,
            use_enhance=selected not in ("rapidocr", "hybrid"),
        )
        pages_with_text = 0
        for e in todo:
            boxes = boxes_by_page.get(str(e["file"]), [])
            slide = _assemble_slide(e["index"], boxes, e["file"])
            slide["_sig"] = _file_signature(e["file"])
            new_results[e["index"]] = slide
            if slide["lines"]:
                pages_with_text += 1
            else:
                util.vlog(f"  · 第 {e['index']} 页无文字（可能是纯图/空白页）")
        util.log(f"  · 识别完成：{pages_with_text}/{len(todo)} 页有文字")
    elif todo:
        # engine=none：仍然产出结构完整的空页，保证下游能跑
        util.warn("  没有可用的 OCR 引擎，所有页面按「无文字」处理（下游仍可生成讲解笔记）")
        for e in todo:
            slide = _assemble_slide(e["index"], [], e["file"], want_figure=True)
            slide["_sig"] = _file_signature(e["file"])
            new_results[e["index"]] = slide

    # ---- 合并（保留 reused 的顺序，按 index 升序输出保证契约）----
    merged: list[dict[str, Any]] = []
    for e in entries:
        idx = e["index"]
        slide = new_results.get(idx) or reused.get(idx)
        if slide is None:
            # 缺图页：占位，保证 index 连续（契约要求 index 与页对应）
            slide = _empty_slide(idx)
        merged.append(slide)
    merged.sort(key=lambda s: int(s.get("index", 0)))

    # ---- 跨页校正标题 ----
    # 同一条标题常在不同页重复出现（例如「核心问题是什么？」先以正文出现，
    # 再作为章节页的大字标题）。大字/彩色标题容易认错一个字，但它在别的页
    # 有一份干净的版本。用全课词频做先验，把明显更合理的写法换回来。
    _repair_titles(merged, allow_insert=True, trusted_only=False)
    # 跑第二遍：只用第一遍修好的「可信标题」去对齐同章节其它页（只补字/换字，不缩字）。
    _repair_titles(merged, allow_insert=True, trusted_only=True)

    # ---- vlm 补充（禁止改写原文）----
    if vlm:
        _apply_vlm(merged, present, force=force)

    # 形近字纠错：用同一课程里置信度更高的写法，纠正个别认错的形近字
    # （「核心问提是什么」→「核心问题是什么」）。只改 1 个字，风险很小。
    try:
        fixed = crossfix.fix_glyph_by_confidence(merged)
        if fixed:
            util.log(f"  · 形近字纠错：{fixed} 处（可用 --no-fix 关闭）")
    except Exception as exc:  # noqa: BLE001
        util.vlog(f"  · 形近字纠错跳过：{exc}")

    payload = {
        "engine": selected,
        "slides": merged,
    }
    util.write_json(root / "slides.json", payload)
    with_text = sum(1 for s in merged if s.get("lines"))
    util.log(f"  · 已写入 {root / 'slides.json'}（{len(merged)} 页，{with_text} 页含文字）")

    # ---- 回写 manifest 的 ocr 阶段标记（有 manifest 才写，没有就算了）----
    _mark_manifest(root, selected, params={"engine": engine, "vlm": bool(vlm), "force": bool(force)})

    return payload


def _empty_slide(index: int) -> dict[str, Any]:
    """缺图页的占位记录，结构与正常页完全一致。"""
    return {
        "index": index,
        "lines": [],
        "text": "",
        "title": "",
        "has_figure": False,
        "figure_note": "",
        "ocr_conf": 0.0,
        "_sig": "",
    }


def _select_engine(root: Path, engine: str) -> tuple[str, Callable[..., dict[str, list[dict[str, Any]]]] | None]:
    """按用户选择/可用性定下引擎，返回 (引擎名, 批量识别函数)。

    优先级（auto）：
        rapidocr > hybrid > vision > tesseract > none

    为什么 RapidOCR 排第一：中文课件的标题/正文准确率明显高于 Apple Vision
    （实测同一页 0.98 vs 0.30，Vision 会把「本课程总学分」读成「某门课程名称」）。
    Vision 仍然保留，用来补 RapidOCR 漏掉的英文/小字，所以 auto 默认走 hybrid。
    """
    if engine == "none":
        return "none", None

    info = probe_ocr_engines()
    rapid_ok = bool(info["rapidocr"]["available"])

    def rapid_runner(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
        try:
            return ocr_rapid.run(paths)
        except Exception as exc:  # noqa: BLE001 - 转成用户可见错误
            raise util.CvnError(f"RapidOCR 识别失败：{exc}") from exc

    def vision_runner() -> Callable[..., dict[str, list[dict[str, Any]]]] | None:
        if sys.platform == "darwin" and _swiftc_path() and SWIFT_SOURCE.exists():
            try:
                binary = ensure_swift_binary(root)
                return lambda paths: _run_vision_swift(binary, paths)
            except util.CvnError as exc:
                if engine == "vision":
                    raise
                util.warn(f"  Swift Vision 不可用，尝试其它引擎：{exc}")
        if _HAS_PYOBJC:
            return _run_vision_pyobjc
        return None

    if engine == "rapidocr":
        if not rapid_ok:
            raise util.CvnError(
                "指定的 OCR 引擎 rapidocr 不可用：\n  · "
                + (info["rapidocr"]["reason"] or "没有找到 rapidocr-onnxruntime")
            )
        return "rapidocr", rapid_runner

    if engine == "hybrid":
        if not rapid_ok:
            raise util.CvnError(
                "指定的 OCR 引擎 hybrid 需要 RapidOCR，但当前不可用：\n  · "
                + (info["rapidocr"]["reason"] or "没有找到 rapidocr-onnxruntime")
            )
        vrunner = vision_runner()
        if vrunner is None:
            util.warn("  hybrid 模式下 Vision 不可用，退化为纯 RapidOCR")
            return "rapidocr", rapid_runner

        def hybrid_runner(paths: Sequence[Path]) -> dict[str, list[dict[str, Any]]]:
            try:
                primary = ocr_rapid.run(paths)
            except Exception as exc:  # noqa: BLE001
                util.warn(f"  RapidOCR 失败，本批改用 Vision：{exc}")
                return vrunner(paths)
            try:
                supplement = vrunner(paths)
            except Exception as exc:  # noqa: BLE001
                util.warn(f"  Vision 补充失败，本批只用 RapidOCR：{exc}")
                return primary
            return {
                str(p): _fuse_primary_supplement(
                    primary.get(str(p), []), supplement.get(str(p), [])
                )
                for p in paths
            }

        return "hybrid", hybrid_runner

    if engine in ("auto", "vision"):
        vrunner = vision_runner()
        if engine == "vision":
            if vrunner is None:
                raise util.CvnError(
                    "指定的 OCR 引擎 vision 不可用：\n"
                    f"  · {info['vision']['reason'] or '没有 swiftc，也没有 pyobjc 的 Vision 绑定'}\n"
                    "解决办法：xcode-select --install（装 swiftc）；"
                    "或在有 pip 权限的环境里 pip install pyobjc-framework-Vision。"
                )
            return "vision", vrunner
        # auto：RapidOCR 优先；它 + Vision 都有时走 hybrid
        if rapid_ok and vrunner is not None:
            return _select_engine(root, "hybrid")
        if rapid_ok:
            return "rapidocr", rapid_runner
        if vrunner is not None:
            return "vision", vrunner

    if engine in ("auto", "tesseract"):
        tess = util.which("tesseract")
        if tess:
            if "chi_sim" not in (info["tesseract"].get("langs") or []):
                util.warn(
                    "  tesseract 没装中文语言包（chi_sim），中文识别效果会很差。\n"
                    "    建议：brew install tesseract tesseract-lang；或改用 RapidOCR。"
                )
            return "tesseract", lambda paths: _run_tesseract(paths, tess=tess)
        if engine == "tesseract":
            raise util.CvnError(
                "指定的 OCR 引擎 tesseract 不可用：找不到 tesseract 命令。\n"
                "解决办法：brew install tesseract tesseract-lang"
            )

    return "none", None


def _fuse_primary_supplement(
    primary: Sequence[dict[str, Any]],
    supplement: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """以主引擎（RapidOCR）为准，只吸收补充引擎（Vision）里**没有重叠**的行。

    不能简单按 IoU 合并：两个引擎的行切分方式不同（Vision 会把页眉时间戳和
    标题并成一行，RapidOCR 分成两行），按框合并会留下重复文本。
    所以策略是「主引擎的框全部保留；补充引擎的框只有在主引擎没有覆盖时才收」。
    """
    out = [dict(b) for b in primary]
    if not supplement:
        return out

    def covered(box: dict[str, Any]) -> bool:
        bx0, by0 = float(box.get("x") or 0), float(box.get("y") or 0)
        bx1, by1 = bx0 + float(box.get("w") or 0), by0 + float(box.get("h") or 0)
        bcy = (by0 + by1) / 2
        for p in out:
            px0, py0 = float(p.get("x") or 0), float(p.get("y") or 0)
            px1, py1 = px0 + float(p.get("w") or 0), py0 + float(p.get("h") or 0)
            ph = max(1e-6, py1 - py0)
            pcy = (py0 + py1) / 2
            # 同一行：垂直中心落在主框高度范围内
            if abs(bcy - pcy) > max(0.012, ph * 0.6):
                continue
            overlap = max(0.0, min(bx1, px1) - max(bx0, px0))
            if overlap / max(1e-6, min(bx1 - bx0, px1 - px0)) >= 0.25:
                return True
        return False

    for box in supplement:
        text = str(box.get("text") or "").strip()
        conf = float(box.get("conf") or 0.0)
        if not text or conf < 0.5:
            continue
        if not any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text):
            continue
        if covered(box):
            continue
        item = dict(box)
        item["source"] = "vision"
        out.append(item)
    return out


def _recognise_multipass(
    runner: Callable[..., dict[str, list[dict[str, Any]]]],
    paths: Sequence[Path],
    root: Path,
    *,
    use_enhance: bool = True,
) -> dict[str, list[dict[str, Any]]]:
    """多路识别：每页跑若干个增强版本，逐框取置信度更高的一版。

    为什么需要：课件模板常见「深色标题栏 + 深色艺术字」「半透明水印压字」
    这类版面，单路识别会整片读不出来（返回空或乱码）。
    增强版本各自解决一类问题（去深底、锐化、放大、灰度），
    再按 IoU 把不同路的框对齐，逐框择优，结果明显更稳。

    代价是识别耗时约等于路数倍，所以默认只用 2 路（原图 + 去深底）。
    """
    if not paths:
        return {}

    variants = ("base", "flat")
    if not enhance.available():
        variants = ("base",)

    tmp_dir = root / ".ocr-variants"
    util.ensure_dir(tmp_dir)

    # 第 1 路：原图
    base_raw = _recognise_batch(runner, paths)

    if not use_enhance:
        return base_raw

    # 第 2 路：增强版
    extra_raw: dict[str, list[dict[str, Any]]] = {}
    try:
        for p in paths:
            made = enhance.render_variants(p, tmp_dir, variants)
            alt = made.get("flat")
            if alt is None or alt == p:
                continue
            extra_raw[str(p)] = [alt]
        if extra_raw:
            util.log(f"  · 多路识别：原图 + 去深底增强（{len(extra_raw)} 页）")
            # 注意：extra_raw 的值是「该页增强图路径」的列表，
            # 这里要展平成路径列表喂给识别器，再按页还原。
            alt_paths = [p for lst in extra_raw.values() for p in lst]
            flat_raw = _recognise_batch(runner, alt_paths)
            extra_raw = {
                key: flat_raw.get(str(lst[0]), []) if lst else []
                for key, lst in extra_raw.items()
            }
    except Exception as exc:  # noqa: BLE001 - 增强失败就退回单路
        util.vlog(f"  · 多路识别的增强部分跳过：{exc}")
        extra_raw = {}

    merged: dict[str, list[dict[str, Any]]] = {}
    for p in paths:
        key = str(p)
        base_boxes = base_raw.get(key, [])
        alt_boxes = extra_raw.get(key)
        if not alt_boxes:
            merged[key] = base_boxes
            continue
        merged[key] = enhance.merge_by_confidence(
            [("base", base_boxes), ("flat", alt_boxes)]
        )
    return merged


def _recognise_batch(
    runner: Callable[..., dict[str, list[dict[str, Any]]]],
    paths: Sequence[Path],
) -> dict[str, list[dict[str, Any]]]:
    """按 VISION_BATCH 分批调用识别函数，边跑边打进度。"""
    results: dict[str, list[dict[str, Any]]] = {}
    total = len(paths)
    for start in range(0, total, VISION_BATCH):
        chunk = paths[start : start + VISION_BATCH]
        util.log(f"  · 识别第 {start + 1}-{start + len(chunk)} 页 / 共 {total} 页")
        part = runner(chunk)
        results.update(part)
    return results


def _cjk_only(text: str) -> str:
    return "".join(ch for ch in text if "\u4e00" <= ch <= "\u9fff")


def _edit_distance(a: str, b: str) -> int:
    """Levenshtein 距离；a、b 都是短标题，朴素 DP 足够。"""
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > 2:
        return 99
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[lb]


def _repair_titles(
    slides: list[dict[str, Any]],
    *,
    allow_insert: bool = True,
    trusted_only: bool = False,
) -> None:
    """用全课语料校正「只错一个常用字」的标题。

    典型场景：章节页的蓝色艺术字「核心问题是什么？」被认成「核心问提是什么？」，
    而同一句话在另一页的正文里是干净的。校正必须同时满足：
      1) 候选行与本页标题等长（±1），编辑距离 1–2；
      2) 候选行在别的页出现过，或它在全课语料里的字频显著更高；
      3) 改动的字都是汉字，且候选字的字频至少是本页字的 3 倍。
    只改标题，不动 lines/text（避免污染 OCR 原文）。
    """
    # 语料：只采信「课件页」的文字。桌面/浏览器/编辑器画面的 OCR 也会产出
    # 像「成为您样的人?」这种错字，混进语料会把正确标题投票投歪。
    course_kinds = {"slide", "blank", "unknown"}
    corpus: list[str] = []
    for s in slides:
        if s.get("kind") not in course_kinds:
            continue
        raw_title = _cjk_only(str(s.get("title") or ""))
        for line in list(s.get("lines") or []) + [s.get("title") or ""]:
            t = util.clean_line(str(line))
            if not t or _PAGE_CHROME_RE.match(t):
                continue
            if re.search(r"\d{1,2}:\d{2}", t):
                continue
            core = _cjk_only(t)
            if 3 <= len(core) <= 60:
                # 标题行本身不进语料：否则同一个错字会被反复投票成「常用字」。
                # 例外：如果这行就是「已经校正过的当前标题」，它是可信语料，要留下。
                keep_current = bool(s.get("title_repaired_from")) and core == raw_title
                near_raw = bool(
                    raw_title
                    and abs(len(core) - len(raw_title)) <= 1
                    and _edit_distance(core, raw_title) <= 1
                )
                if near_raw and not keep_current:
                    continue
                corpus.append(t)
    if len(corpus) < 3:
        return

    freq: dict[str, int] = {}
    for line in corpus:
        for ch in _cjk_only(line):
            freq[ch] = freq.get(ch, 0) + 1

    def score(text: str) -> float:
        # 用对数频率衡量「这串字在课程语境里有多常见」
        return sum(math.log1p(freq.get(ch, 0)) for ch in _cjk_only(text))

    # 已经跨页校正过的标题是「可信语料」，匹配时给它加权，
    # 避免又被同一页的原始乱码标题（也进了语料）拉回去。
    trusted = {str(s.get("title") or "") for s in slides if s.get("title_repaired_from")}

    for slide in slides:
        if slide.get("kind") not in course_kinds:
            continue
        title = util.clean_line(str(slide.get("title") or ""))
        core = _cjk_only(title)
        if len(core) < 4 or len(core) > 24:
            continue
        best: tuple[float, str] | None = None
        for line in corpus:
            cand = _cjk_only(line)
            if cand == core or len(cand) < len(core):
                continue
            # 等长窗口 + 允许插入一个字（标题首字被录屏时间戳盖住时很常见）
            if trusted_only and line not in trusted:
                continue
            if trusted_only and len(cand) < len(core):
                continue
            # 1) 可信标题：整条比对，不许截取子串（否则会多出「是」之类的常用字）
            if line in trusted and len(cand) in (len(core), len(core) + 1):
                if len(cand) > len(core) and not allow_insert:
                    continue
                dist = _edit_distance(core, cand)
                if 0 < dist <= 2:
                    gain = score(cand) - score(core) + 3.0
                    if gain >= 1.2 and (best is None or gain > best[0]):
                        best = (gain, cand)
                continue
            # 2) 普通语料：等长窗口滑动（±1 字用于补齐被时间戳盖住的标题首字）
            for width in (len(core), len(core) + 1):
                if width > len(core) and not allow_insert:
                    continue
                if len(cand) < width:
                    continue
                for start in range(0, len(cand) - width + 1):
                    window = cand[start : start + width]
                    dist = _edit_distance(core, window)
                    if dist == 0:
                        continue
                    if dist > (2 if width == len(core) else 1):
                        continue
                    gain = score(window) - score(core)
                    if gain < 1.2:
                        continue
                    if best is None or gain > best[0]:
                        best = (gain, window)
        if best is not None:
            new_title = best[1]
            if new_title != core:
                slide.setdefault("title_repaired_from", title)
                slide["title"] = util.truncate(new_title, 40)
                util.vlog(f"  · 第 {slide.get('index')} 页标题跨页校正：{title} → {new_title}")


def _title_plausibility(text: str) -> float:
    """用 jieba 词频给标题打「像不像人话」的分（越高越通顺）。

    只用于在 OCR 标题和 VLM 标题之间做选择，不参与正文，也不会改写任何文字。
    """
    if not _HAS_JIEBA or not text:
        return 0.0
    tokens = [w for w in jieba.cut(text) if _cjk_only(w)]
    total = sum(len(w) for w in tokens)
    if not total:
        return 0.0
    score = 0.0
    for w in tokens:
        score += len(w) * math.log1p(int(jieba.dt.FREQ.get(w, 0)))
    return score / total


# 标题里出现这些字，基本可以断定 OCR 认错了（非常用字或字形近似的误认）。
# 「仟」应为「什」/「要」、「這」是繁体、「曰」被当成「日」等。
_SUSPECT_CHARS = set("仟這歷麼們發現後裡為與從對開關時間問題學習寫讀")

# 常用汉字表（GB2312 一级字库的近似集合）——不在此表里的字，
# 出现在课件标题里大概率是 OCR 认错，而不是老师真的用生僻字。
_COMMON_HANZI: set[str] | None = None


def _load_common_hanzi() -> set[str]:
    """加载常用汉字集；取不到就返回空集（此时该判据自动失效）。"""
    global _COMMON_HANZI
    if _COMMON_HANZI is not None:
        return _COMMON_HANZI
    chars: set[str] = set()
    # GB2312 一级汉字区：B0A1-F7FE
    for b1 in range(0xB0, 0xF8):
        for b2 in range(0xA1, 0xFF):
            try:
                ch = bytes([b1, b2]).decode("gb2312")
            except UnicodeDecodeError:
                continue
            if "\u4e00" <= ch <= "\u9fff":
                chars.add(ch)
    _COMMON_HANZI = chars
    return chars


def _char_overlap(a: str, b: str) -> float:
    """两段文字的字重合度（多重集 Dice 系数），0~1。"""
    if not a or not b:
        return 0.0
    from collections import Counter

    ca, cb = Counter(a), Counter(b)
    inter = sum((ca & cb).values())
    return 2.0 * inter / (len(a) + len(b))


def _rare_char_ratio(text: str) -> float:
    """汉字里有多少是"不常用"的（OCR 认错的强信号）。"""
    common = _load_common_hanzi()
    if not common:
        return 0.0
    han = [c for c in _cjk_only(text)]
    if not han:
        return 0.0
    rare = sum(1 for c in han if c not in common)
    return rare / len(han)


# 标题开头的日期/时间戳残留：录屏软件的时间戳压在标题上，会混进来
_TITLE_LEADING_DATE = re.compile(
    r"^\s*(?:\d{4}|[xX]\s*ier|\d{1,2}\s*[:：]\s*\d{2})"
)


def _looks_misrecognized(text: str) -> bool:
    """这个标题看起来是不是 OCR 认错了。"""
    t = util.clean_line(text or "")
    if not t:
        return False
    # 以年份/时间戳开头 —— 没有任何正经标题会这样，必然是录屏水印混入
    if _TITLE_LEADING_DATE.match(t):
        return True
    # 含明显的错字/繁体字
    if any(c in _SUSPECT_CHARS for c in t):
        return True
    # 生僻字占比过高（正常课件标题不会这样）
    if _rare_char_ratio(t) >= 0.15:
        return True
    return False


def _has_unknown_word(text: str) -> bool:
    """标题里有没有「词典里不存在的二字词」——错字标题的典型特征。"""
    if not _HAS_JIEBA or not text:
        return False
    for w in jieba.cut(text):
        if len(w) >= 2 and _cjk_only(w) == w and int(jieba.dt.FREQ.get(w, 0)) == 0:
            return True
    return False


def _accept_vlm_title(slide: dict[str, Any], cand: str) -> bool:
    """判断 VLM 读出的标题是否比 OCR 标题更可信。

    不是「VLM 一读就换」：2B 模型在无标题页会退化成「（一）」，
    也会给好标题硬加编号（「大学生成长的三个层次」→「…（一）」）。
    规则：
      1. OCR 标题太短 / 只是编号 → 用 VLM；
      2. OCR 标题其实是正文长句 → 用 VLM；
      3. VLM 只是给 OCR 加编号或标点 → 保留 OCR；
      4. VLM 明显更通顺（词频分高 0.8 以上）或 OCR 含词典里没有的错词 → 用 VLM。
    """
    ocr = util.clean_line(str(slide.get("title") or ""))
    ocr_core = _cjk_only(ocr)
    cand_core = _cjk_only(cand)
    if len(cand_core) < 2:
        return False

    # 1) OCR 标题太短，基本可以确定没读全
    if len(ocr_core) < 3:
        return True

    # 2) OCR 标题是从正文里错抓的长句（标题行本身也在 lines 里，
    #    但页面还有明显更短的行，说明它不是标题）
    lines = [str(l).strip() for l in (slide.get("lines") or []) if str(l).strip()]
    same = [l for l in lines if _cjk_only(l) == ocr_core]
    if len(ocr_core) >= 15 and same and any(
        _cjk_only(l) and len(_cjk_only(l)) < len(ocr_core) for l in lines
    ):
        return True

    # 3) VLM 只是给 OCR 加了编号/标点，或两边核心字完全一致 → 保留 OCR
    if cand_core == ocr_core or cand_core in ocr_core or ocr_core in cand_core:
        return False

    # 4) OCR 标题明显认错了（含错字/生僻字），而 VLM 的看起来正常 → 用 VLM
    #    这条不依赖 jieba，实测是修复「人为仟要学习」「(如荷学以成大」的关键。
    if _looks_misrecognized(ocr) and not _looks_misrecognized(cand):
        return True

    # 4b) 两边长度接近但"逐字对不上"→ OCR 那段整体认错了，用 VLM。
    #     例：「天写动物最天的这别是什么」vs「人与动物最大的区别是什么」
    #     字都是常用字（生僻字判据抓不到），但重合度只有一半左右，
    #     显然不是同一句话被"稍微"认错。
    # 门槛放到 4：短标题（「日天"是什么」5 字）错得更离谱，更该让 VLM 接管
    if len(ocr_core) >= 4 and len(cand_core) >= 4:
        overlap = _char_overlap(ocr_core, cand_core)
        # 0.75 是实测出来的分界：
        #   「某某课程与素养」vs「…（一）」这类只是加编号  → 0.95+
        #   「天写动物最天的这别是什么」vs「人与动物最大的区别是什么」→ 0.67
        if overlap < 0.75:
            return True

    # 5) 词频判断
    if _has_unknown_word(ocr) and not _has_unknown_word(cand):
        return True
    return _title_plausibility(cand) >= _title_plausibility(ocr) + 0.8


def _apply_vlm(slides: list[dict[str, Any]], present: list[dict[str, Any]], *, force: bool) -> None:
    """用视觉大模型读标题 / 补图表说明（只增强，不改写正文 OCR）。

    两条路，优先级从高到低：
      1. 外部命令：CVN_VLM_CMD 指向「读图输出 JSON」的命令（老接口，保留兼容）
      2. 本地 Ollama：qwen3-vl:2b-instruct（或 CVN_VLM_MODEL 指定的模型）

    为什么需要：录屏时间戳常压在标题上，OCR 会把「（一）人为什么要学习」
    读成「人为仟要学习」。2B 视觉模型能稳定读对（实测），代价是每页 2–5 秒。
    """
    cmd = os.environ.get("CVN_VLM_CMD", "").strip()
    local = vlm.probe()
    by_index = {int(s["index"]): s for s in slides if s.get("index") is not None}
    course_kinds = {"slide", "blank", "unknown"}
    targets = [
        e for e in present
        if e["index"] in by_index
        and by_index[e["index"]].get("kind") in course_kinds
        and (force or not by_index[e["index"]].get("vlm_model"))
    ]
    if not targets:
        return

    if not cmd and not local.get("available"):
        util.warn(
            "  --vlm 已开启，但没有可用的视觉模型，已跳过（OCR 原文不受影响）。\n"
            "    · 本地模型：`ollama pull qwen3-vl:2b-instruct`\n"
            "    · 或设置 CVN_VLM_CMD 指向一个「读图输出 JSON」的命令"
        )
        return

    label = cmd if cmd else f"{local.get('model')} @ {local.get('url')}"
    util.log(f"  · VLM 读标题/图表：{len(targets)} 页（{label}）")
    fixed = 0
    for e in targets:
        slide = by_index.get(e["index"])
        if slide is None:
            continue
        try:
            if cmd:
                proc = subprocess.run(
                    [*cmd.split(), str(e["file"])],
                    capture_output=True, text=True, timeout=180, errors="replace",
                )
                if proc.returncode != 0:
                    util.vlog(f"  · VLM 命令失败（第 {e['index']} 页），跳过")
                    continue
                data = json.loads((proc.stdout or "").strip().splitlines()[-1] or "{}")
                title = vlm._clean_title(data.get("title"))
                if title and title != slide.get("title") and _accept_vlm_title(slide, title):
                    slide["title_ocr"] = slide.get("title", "")
                    slide["title"] = title
                    slide["title_source"] = "VLM 读图"
                    fixed += 1
                note = str(data.get("figure_note") or "").strip()
                if note:
                    slide["figure_note"] = note
                    slide["has_figure"] = True
                slide["vlm_model"] = "external"
            else:
                data = vlm.analyse(e["file"])
                if data.get("error"):
                    util.vlog(f"  · 第 {e['index']} 页 VLM 失败：{data['error']}")
                    continue
                title = vlm._clean_title(data.get("title"))
                if title and title != slide.get("title") and _accept_vlm_title(slide, title):
                    slide["title_ocr"] = slide.get("title", "")
                    slide["title"] = title
                    slide["title_source"] = "VLM 读图"
                    fixed += 1
                note = str(data.get("figure_note") or "").strip()
                if note:
                    slide["figure_note"] = note
                    slide["has_figure"] = True
                slide["vlm_model"] = local.get("model", "")
        except Exception as exc:  # noqa: BLE001 - VLM 失败不能拖垮 OCR
            util.vlog(f"  · VLM 调用异常（第 {e['index']} 页），跳过：{exc}")
    util.log(f"  · VLM 完成：{fixed} 页标题被纠正")


def _mark_manifest(root: Path, engine: str, *, params: dict[str, Any]) -> None:
    """把 ocr 阶段标记写进 manifest.json（存在才写；写失败不影响主流程）。"""
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = util.read_json(manifest_path, default=None)
        if not isinstance(manifest, dict):
            return
        stages = manifest.setdefault("stages", {})
        stages["ocr"] = {"done": True, "at": util.stamp(), "params": {"engine": engine, **params}}
        util.write_json(manifest_path, manifest)
    except Exception as exc:  # manifest 是辅助产物，坏了不能拖垮 OCR
        util.vlog(f"  · 写 manifest.json 失败（忽略）：{exc}")


# ==========================================================================
# 命令行入口（方便单独调试；正式入口在 course_kit.py）
# ==========================================================================


def _main(argv: Sequence[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        prog="cvn ocr",
        description="识别课件截图的文字，生成 slides.json",
    )
    ap.add_argument("out_dir", help="课程输出目录（含 frames.json 与 slides/）")
    ap.add_argument("--engine", default="auto", choices=["auto", "rapidocr", "hybrid", "vision", "tesseract", "none"])
    ap.add_argument("--vlm", action="store_true", help="用视觉大模型补充图片说明（不改 OCR 原文）")
    ap.add_argument("--force", action="store_true", help="忽略缓存，全部重做")
    ap.add_argument("--limit", type=int, default=None, help="只处理前 N 页（调试用）")
    ap.add_argument("--dry-run", action="store_true", help="只报告将要做什么")
    ap.add_argument("--probe", action="store_true", help="只打印引擎探测结果")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    util.set_verbose(args.verbose)
    try:
        if args.probe:
            util.log(json.dumps(probe_ocr_engines(), ensure_ascii=False, indent=2))
            return 0
        result = run_ocr(
            args.out_dir,
            engine=args.engine,
            vlm=args.vlm,
            force=args.force,
            limit=args.limit,
            dry_run=args.dry_run,
        )
        util.log(f"✔ OCR 完成，engine={result.get('engine')}，页数={len(result.get('slides', []))}")
        return 0
    except util.CvnError as exc:
        util.log(f"✘ {exc}")
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_main())
