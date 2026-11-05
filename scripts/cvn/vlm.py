"""vlm.py —— 本地视觉大模型（Ollama）读课件标题 / 图表说明。

为什么需要这一层
----------------
录屏软件的时间戳常常正好压在课件标题上：
    「（一）人为什么要学习」被盖住一半，OCR 读成「人为仟要学习」。
RapidOCR 对正文很强，但这类「被遮挡的彩色标题」是它的能力边界。
本地 2B 视觉模型（qwen3-vl:2b-instruct）能稳定读对，实测：

    OCR  : 人为仟要学习
    VLM  : （一）人为什么要学习

本模块只做两件事，**绝不改写正文 OCR**：
    1. 读最上方的标题；
    2. 判断有没有图表/照片，并用一句话说明。

依赖：本机 Ollama（默认 http://localhost:11434）与一个视觉模型。
默认模型 qwen3-vl:2b-instruct（约 2GB，8GB 内存的 Mac 能跑）。
可用环境变量覆盖：
    CVN_VLM_MODEL  模型名
    CVN_VLM_URL    Ollama 地址
    CVN_VLM_TIMEOUT 单页超时秒数（默认 180）
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

DEFAULT_MODEL = "qwen3-vl:2b-instruct"
DEFAULT_URL = "http://localhost:11434"

_PROMPT = """请阅读这张课件截图，只输出一个 JSON，不要解释：
{"title": "最上方的标题文字（忽略时间戳/日期/星期/校名；保留（一）这样的编号）",
 "has_figure": true或false（是否有图表、示意图、照片）,
 "figure_note": "若有图表，用一句话说明它画了什么；没有则空字符串"}"""

# 明显不是标题的输出：时间戳、校名/机构名、纯标点
_BAD_TITLE = re.compile(
    r"(20\d{2}\s*年|\d{1,2}:\d{2}|星期[一二三四五六日]"
    r"|大学|大學|学院|學院|UNIVERSITY|COLLEGE|INSTITUTE)"
)

_CACHE: dict[str, Any] | None = None


def _url() -> str:
    return (os.environ.get("CVN_VLM_URL") or DEFAULT_URL).rstrip("/")


def _model() -> str:
    return (os.environ.get("CVN_VLM_MODEL") or DEFAULT_MODEL).strip()


def _timeout() -> float:
    try:
        return float(os.environ.get("CVN_VLM_TIMEOUT") or 180)
    except ValueError:
        return 180.0


def _request(path: str, payload: dict[str, Any] | None = None, *, timeout: float = 20.0) -> Any:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        _url() + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if payload is None else "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def probe(*, refresh: bool = False) -> dict[str, Any]:
    """检查 Ollama 是否在跑、目标模型是否已安装。"""
    global _CACHE
    if _CACHE is not None and not refresh:
        return dict(_CACHE)

    model = _model()
    try:
        tags = _request("/api/tags", timeout=3.0)
    except (urllib.error.URLError, OSError, TimeoutError) as exc:
        # 失败结果**不缓存**：Ollama 可能只是这一下忙/冷启动慢，
        # 缓存住会让整个进程再也用不了 VLM。下次调用重新探测。
        return {
            "available": False,
            "model": model,
            "url": _url(),
            "reason": f"连不上 Ollama（{_url()}）：{exc}。先运行 `ollama serve`。",
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "model": model,
            "url": _url(),
            "reason": f"Ollama 返回异常：{exc}",
        }

    names = [str(m.get("name") or "") for m in tags.get("models") or []]
    # 允许写 qwen3-vl:2b-instruct 或 qwen3-vl:2b（Ollama 的 latest 别名）
    hit = model in names or any(n == model or n.startswith(model + "-") for n in names)
    if not hit:
        return {
            "available": False,
            "model": model,
            "url": _url(),
            "reason": (
                f"Ollama 在跑，但没装模型 {model}。"
                f"安装：`ollama pull {model}`；或设置 CVN_VLM_MODEL 指向已有视觉模型。"
            ),
            "installed": names,
        }

    _CACHE = {"available": True, "model": model, "url": _url(), "reason": "", "installed": names}
    return dict(_CACHE)


def available() -> bool:
    return bool(probe().get("available"))


def _parse_json(text: str) -> dict[str, Any]:
    """从模型输出里抠出第一个 JSON 对象（常带 ```json 围栏）。"""
    cleaned = re.sub(r"```(?:json)?", "", text or "").strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end > start:
        cleaned = cleaned[start : end + 1]
    try:
        data = json.loads(cleaned)
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _clean_title(text: Any) -> str:
    title = str(text or "").strip().strip("`\"'“”‘’")
    title = title.replace("\u200b", "").replace("\ufeff", "")
    title = re.sub(r"\s+", " ", title)
    if not title or len(title) > 40:
        return ""
    if _BAD_TITLE.search(title):
        return ""
    cjk = [ch for ch in title if "\u4e00" <= ch <= "\u9fff"]
    # 至少两个字，且不能只有「（一）」这种编号：VLM 在无标题页会退化成编号。
    if len(cjk) < 2:
        return ""
    if re.fullmatch(r"[\s（(]?[一二三四五六七八九十]+[）)]?[\s、.．:：]*", title):
        return ""
    if not any("\u4e00" <= ch <= "\u9fff" or ch.isalnum() for ch in title):
        return ""
    return title


def analyse(path: str | os.PathLike[str], *, timeout: float | None = None) -> dict[str, Any]:
    """读一页课件：返回 {title, has_figure, figure_note, raw}。

    失败时返回空字段并带 `error`，不抛异常——VLM 只是增强，不能拖垮主流程。
    """
    info = probe()
    if not info.get("available"):
        return {"title": "", "has_figure": False, "figure_note": "", "error": info.get("reason")}

    try:
        raw = Path(path).read_bytes()
    except OSError as exc:
        return {"title": "", "has_figure": False, "figure_note": "", "error": str(exc)}

    payload = {
        "model": _model(),
        "prompt": _PROMPT,
        "images": [base64.b64encode(raw).decode("ascii")],
        "stream": False,
        "format": "json",
        "options": {"temperature": 0},
    }
    try:
        data = _request("/api/generate", payload, timeout=timeout or _timeout())
    except Exception as exc:  # noqa: BLE001
        return {"title": "", "has_figure": False, "figure_note": "", "error": str(exc)}

    body = _parse_json(str(data.get("response") or ""))
    return {
        "title": _clean_title(body.get("title")),
        "has_figure": bool(body.get("has_figure")),
        "figure_note": str(body.get("figure_note") or "").strip()[:200],
        "raw": body,
    }


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(argv if argv is not None else sys.argv[1:])
    if not args or args[0] == "probe":
        info = probe(refresh=True)
        print("VLM:", "可用" if info["available"] else "不可用")
        print("模型：", info.get("model"))
        print("地址：", info.get("url"))
        if info.get("reason"):
            print("原因：", info["reason"])
        return 0 if info["available"] else 1
    for p in args:
        print(p, "=>", analyse(p))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
