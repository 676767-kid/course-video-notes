"""ocr_rapid.py —— 用 RapidOCR（PP-OCRv4/v5 的 ONNX 版）识别课件。

为什么加这一路
--------------
Apple Vision 对中文印刷体正文很强，但对「深色标题栏上的彩色标题」
「字体较特殊的标题」会给出**看起来像中文但完全错的字**，而且置信度只有 0.3。
实测同一张 1920×1080 课件：

    Vision   : 20苯颗椁思分苏为z苏分，总苏时为32颗时      conf 0.30
    RapidOCR : 本课程总学分为2学分，总学时为32课时          conf 0.98

RapidOCR 是 PaddleOCR 的 PP-OCR 模型转成 ONNX，纯 CPU、约 15MB，
中文识别准确率明显更高，单页 0.4–0.7 秒（M2）。所以把它作为
中文课件的**首选引擎**，Vision 退为补充（补 RapidOCR 漏掉的英文/细字）。

依赖
----
RapidOCR 需要 Python ≤ 3.13（onnxruntime 尚未支持 3.14+）。主程序可能跑在
更新的 Python 上，所以这里**不直接 import**，而是把识别放到一个子进程里，
用 `CVN_OCR_PYTHON`（或自动探测到的解释器）执行 `ocr_rapid_worker.py`。

探测顺序（找到第一个能 `import rapidocr_onnxruntime` 的解释器）：
    1. 环境变量 CVN_OCR_PYTHON
    2. 当前解释器
    3. ~/.venvs/cvn/bin/python
    4. ~/.cache/cvn/venv/bin/python
    5. PATH 里的 python3.12 / python3.13 / python3.11

对上层只暴露：
    available() -> bool
    python_path() -> str | None
    run(paths) -> {path: [box, ...]}
    probe() -> {"available": bool, "python": str, "reason": str}
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

WORKER = Path(__file__).resolve().parent / "ocr_rapid_worker.py"

# 探测结果缓存，避免每次 OCR 都去起子进程 import 一遍
_PROBE_CACHE: dict[str, Any] | None = None

_CANDIDATE_NAMES = ("python3.12", "python3.13", "python3.11")


def _candidate_pythons() -> list[str]:
    """按优先级列出可能的解释器（去重、只保留存在的）。"""
    out: list[str] = []
    seen: set[str] = set()

    def add(value: str | os.PathLike[str] | None) -> None:
        if not value:
            return
        text = str(value)
        if not text or text in seen:
            return
        if not os.path.isabs(text):
            found = shutil.which(text)
            if not found:
                return
            text = found
        if not Path(text).exists():
            return
        seen.add(text)
        out.append(text)

    add(os.environ.get("CVN_OCR_PYTHON"))
    add(sys.executable)
    add(Path.home() / ".venvs" / "cvn" / "bin" / "python")
    add(Path.home() / ".cache" / "cvn" / "venv" / "bin" / "python")
    add(Path(__file__).resolve().parents[1] / ".venv" / "bin" / "python")
    for name in _CANDIDATE_NAMES:
        add(name)
    return out


def _can_import(python: str) -> bool:
    """子进程里试 import；超时/失败都算不可用。"""
    try:
        proc = subprocess.run(
            [python, "-c", "import rapidocr_onnxruntime, onnxruntime"],
            capture_output=True,
            text=True,
            timeout=30,
            errors="replace",
        )
        return proc.returncode == 0
    except (OSError, subprocess.TimeoutExpired):
        return False


def probe(*, refresh: bool = False) -> dict[str, Any]:
    """探测 RapidOCR 是否可用，返回 {available, python, reason}。"""
    global _PROBE_CACHE
    if _PROBE_CACHE is not None and not refresh:
        return dict(_PROBE_CACHE)

    if not WORKER.exists():
        _PROBE_CACHE = {
            "available": False,
            "python": "",
            "reason": f"缺少 worker 脚本：{WORKER}",
        }
        return dict(_PROBE_CACHE)

    tried: list[str] = []
    for python in _candidate_pythons():
        tried.append(python)
        if _can_import(python):
            _PROBE_CACHE = {"available": True, "python": python, "reason": ""}
            return dict(_PROBE_CACHE)

    _PROBE_CACHE = {
        "available": False,
        "python": "",
        "reason": (
            "找不到装了 rapidocr-onnxruntime 的 Python（试过："
            + "、".join(tried[:4])
            + ("…" if len(tried) > 4 else "")
            + "）。安装：uv venv ~/.venvs/cvn --python 3.12 && "
            "uv pip install --python ~/.venvs/cvn/bin/python rapidocr-onnxruntime"
        ),
    }
    return dict(_PROBE_CACHE)


def available() -> bool:
    return bool(probe().get("available"))


def python_path() -> str | None:
    info = probe()
    return str(info.get("python") or "") or None


def _normalise(box: Any, width: int, height: int) -> dict[str, Any] | None:
    """把 RapidOCR 的 4 点框转成契约里的归一化 x/y/w/h（原点左上）。"""
    try:
        pts = [(float(p[0]), float(p[1])) for p in box]
    except (TypeError, ValueError, IndexError):
        return None
    if not pts or width <= 0 or height <= 0:
        return None
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = max(0.0, min(xs)), min(float(width), max(xs))
    y0, y1 = max(0.0, min(ys)), min(float(height), max(ys))
    if x1 <= x0 or y1 <= y0:
        return None
    return {
        "x": round(x0 / width, 6),
        "y": round(y0 / height, 6),
        "w": round((x1 - x0) / width, 6),
        "h": round((y1 - y0) / height, 6),
    }


def _keep(text: str, conf: float) -> bool:
    """过滤装饰图形产生的符号垃圾；保留正常中英文/数字行。"""
    t = (text or "").strip()
    if not t or conf < 0.45:
        return False
    # 至少要有一个汉字 / 拉丁字母 / 数字；纯标点、纯符号丢掉
    return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in t)


def run(paths: Sequence[Path], *, timeout: float | None = None) -> dict[str, list[dict[str, Any]]]:
    """批量识别。一次子进程吃下全部路径，返回 {路径: [框]}。

    失败时抛 RuntimeError；上层会转成中文 CvnError 或降级到 Vision。
    """
    if not paths:
        return {}
    python = python_path()
    if not python:
        raise RuntimeError(probe().get("reason") or "RapidOCR 不可用")

    stdin_text = "\n".join(str(p) for p in paths) + "\n"
    if timeout is None:
        timeout = 120 + 20 * len(paths)
    try:
        proc = subprocess.run(
            [python, str(WORKER)],
            input=stdin_text,
            capture_output=True,
            text=True,
            timeout=timeout,
            errors="replace",
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"RapidOCR 识别超时（{timeout:.0f}s，{len(paths)} 张图）") from exc
    except OSError as exc:
        raise RuntimeError(f"无法启动 RapidOCR 子进程：{exc}") from exc

    if proc.returncode != 0:
        tail = "\n".join((proc.stderr or "").strip().splitlines()[-6:])
        raise RuntimeError(f"RapidOCR 进程失败（exit {proc.returncode}）\n{tail}")

    results: dict[str, list[dict[str, Any]]] = {}
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        path = str(obj.get("file") or "")
        if not path:
            continue
        width = int(obj.get("width") or 0)
        height = int(obj.get("height") or 0)
        boxes: list[dict[str, Any]] = []
        for item in obj.get("lines") or []:
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            conf = float(item.get("conf") or 0.0)
            if not _keep(text, conf):
                continue
            geom = _normalise(item.get("box"), width, height)
            if not geom:
                continue
            boxes.append({"text": text, "conf": conf, **geom})
        results[path] = boxes

    for p in paths:
        results.setdefault(str(p), [])
    return results


def main(argv: Sequence[str] | None = None) -> int:
    """命令行自检：python -m cvn.ocr_rapid probe|图片..."""
    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] == "probe":
        info = probe(refresh=True)
        print("RapidOCR:", "可用" if info["available"] else "不可用")
        if info.get("python"):
            print("解释器：", info["python"])
        if info.get("reason"):
            print("原因：", info["reason"])
        return 0 if info["available"] else 1
    out = run([Path(p) for p in args])
    for path, boxes in out.items():
        print("=" * 20, path)
        for b in boxes:
            print(f"  {b['conf']:.2f} {b['text']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
