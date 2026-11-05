"""OCR 前的图像增强：把「艺术字 / 低对比 / 水印干扰」的课件变得更好认。

Vision 对印刷体正文识别很好（置信度 1.0），但对艺术字体、竖排标题、
带半透明水印的标题会给出低置信度甚至乱码。单跑一路不够稳，
所以这里准备若干个增强版本，让上层做**多路识别 + 逐框择优**。

每个版本只做一件事，避免互相干扰：

    sharp    轻度锐化 —— 边缘更实，艺术字笔画粘连被切开
    gray     纯灰度     —— 去掉主题色/彩色装饰干扰
    scale    放大 1.5x —— 小字号（页脚、页码）看得更清
    contrast 局部对比拉伸 —— 水印把字压灰时，把笔画拉回深色

所有版本都保持与原图相同的宽高比，这样不同路的 bbox 坐标可以直接对齐择优。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

try:
    from PIL import Image, ImageFilter, ImageOps  # type: ignore

    _HAS_PIL = True
except Exception:  # pragma: no cover
    Image = None  # type: ignore[assignment]
    ImageFilter = None  # type: ignore[assignment]
    ImageOps = None  # type: ignore[assignment]
    _HAS_PIL = False


# 变体名 -> (说明, 是否会改变坐标)
VARIANTS: tuple[tuple[str, str], ...] = (
    ("base", "原图"),
    ("flat", "深色底转白底"),
    ("sharp", "锐化"),
    ("gray", "灰度"),
    ("scale", "放大 1.5x"),
)

# 一次最多做几路（路数越多越慢）
MAX_VARIANTS = 3


def available() -> bool:
    return _HAS_PIL


def _fix_colored_background(img: Any) -> Any:
    """把「深色/彩色底 + 深色字」的版面拉成「白底 + 黑字」。

    很多课件模板顶部有一条深蓝色标题栏，上面压着深色艺术字。
    这种组合对比度极低（蓝底 RGB≈57,97,186，文字≈19,19,19），
    Vision 会整片读不出来（返回空或乱码）。

    做法：按颜色把版面分成两类——
      - 背景：偏蓝/偏暗的大色块 → 置白
      - 前景：其余像素保留，并额外做一次对比拉伸
    不依赖 numpy 时退回纯 Pillow 实现。
    """
    try:
        import numpy as np  # type: ignore

        arr = np.array(img.convert("RGB")).astype("int16")
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        mx = arr.max(axis=2)
        mn = arr.min(axis=2)
        # 偏蓝的色块（b 明显高于 r）且不是纯黑文字
        bluish = (b > r + 22) & (b > 95)
        out = np.full_like(arr, 255)
        keep = ~bluish
        out[keep] = arr[keep]
        dark = out.max(axis=2) < 130
        out[dark] = 0
        return Image.fromarray(out.astype("uint8"))
    except Exception:  # noqa: BLE001 - 没 numpy 就跳过这一路
        return img


def _make_variant(img: Any, name: str) -> Any:
    """按名字生成增强版本；失败时返回原图。"""
    if name == "base":
        return img
    try:
        if name == "sharp":
            return img.filter(ImageFilter.UnsharpMask(radius=2, percent=140, threshold=2))
        if name == "gray":
            return img.convert("L").convert("RGB")
        if name == "scale":
            w, h = img.size
            return img.resize((int(w * 1.5), int(h * 1.5)), Image.LANCZOS)
        if name == "flat":
            # 深色底 + 深色字 → 拉成白底黑字
            return _fix_colored_background(img)
    except Exception:  # noqa: BLE001
        return img
    return img


def render_variants(
    path: str | Path,
    out_dir: str | Path,
    names: tuple[str, ...] = ("base",),
) -> dict[str, Path]:
    """把一个页面渲染成多个增强版本，返回 {变体名: 临时文件路径}。

    没有 Pillow 时只返回原图。
    """
    src = Path(path)
    out: dict[str, Path] = {}
    if not _HAS_PIL:
        out["base"] = src
        return out

    out_dir = Path(out_dir)
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return {"base": src}

    try:
        with Image.open(src) as im:
            rgb = im.convert("RGB")
            for name in names:
                if name == "base":
                    out["base"] = src
                    continue
                dest = out_dir / f"{src.stem}__{name}.png"
                try:
                    _make_variant(rgb, name).save(dest)
                    out[name] = dest
                except Exception as exc:  # noqa: BLE001
                    # 单个增强版本失败不该拖垮整页，退回用原图那一路
                    import sys as _sys
                    print(f"[cvn] 增强版本 {name} 生成失败：{exc}", file=_sys.stderr)
    except Exception:  # noqa: BLE001 - 图片本身坏了就退回原图
        out["base"] = src
    return out


def merge_by_confidence(
    runs: list[tuple[str, list[dict[str, Any]]]],
    *,
    iou_threshold: float = 0.45,
) -> list[dict[str, Any]]:
    """把多路识别的框合并成一组，逐框取置信度更高的那一版。

    runs: [(变体名, 该路的框列表), ...]

    合并策略：
      - 同一路内部的框保持原顺序（阅读顺序由上层排）
      - 对每个路两两算 IoU，重叠超过阈值的视为"同一个框"
      - 同一个框在不同路的文本若**完全一致**，保留置信度更高的那个，
        并记录 seen_in（有几路认同），这本身是可信度信号
      - 同一个框在不同路的文本**不一致**时，取置信度更高的；
        若置信度接近（差 < 0.15），优先取 base 路（最忠实于原图）
    """
    if not runs:
        return []
    if len(runs) == 1:
        out = []
        for box in runs[0][1]:
            b = dict(box)
            b["seen_in"] = 1
            b["variants"] = [runs[0][0]]
            out.append(b)
        return out

    base_name = runs[0][0]
    base_boxes = runs[0][1]

    def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
        ax0, ay0 = a.get("x", 0.0), a.get("y", 0.0)
        ax1, ay1 = ax0 + a.get("w", 0.0), ay0 + a.get("h", 0.0)
        bx0, by0 = b.get("x", 0.0), b.get("y", 0.0)
        bx1, by1 = bx0 + b.get("w", 0.0), by0 + b.get("h", 0.0)
        ix = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        iy = max(0.0, min(ay1, by1) - max(ay0, by0))
        inter = ix * iy
        if inter <= 0:
            return 0.0
        area_a = max(1e-9, (ax1 - ax0) * (ay1 - ay0))
        area_b = max(1e-9, (bx1 - bx0) * (by1 - by0))
        return inter / (area_a + area_b - inter)

    def _same_box(a: dict[str, Any], b: dict[str, Any]) -> bool:
        """判断两路的框是否指同一块文字。

        只用 IoU 是不够的：不同增强版本下 Vision 给出的框会有几像素的
        位置漂移，同一行的框甚至可能完全不相交（框高度差很多）。
        所以再加一条「同一行」判据：垂直方向大幅重叠（IoU > 0.3）
        且水平方向也有重叠，即认为是同一行文字。
        """
        if _iou(a, b) >= iou_threshold:
            return True
        ax0, ay0 = a.get("x", 0.0), a.get("y", 0.0)
        ax1, ay1 = ax0 + a.get("w", 0.0), ay0 + a.get("h", 0.0)
        bx0, by0 = b.get("x", 0.0), b.get("y", 0.0)
        bx1, by1 = bx0 + b.get("w", 0.0), by0 + b.get("h", 0.0)
        # 垂直重叠（同一行）
        vy = max(0.0, min(ay1, by1) - max(ay0, by0))
        ha = max(1e-9, ay1 - ay0)
        hb = max(1e-9, by1 - by0)
        v_ratio = vy / min(ha, hb)
        # 水平重叠（同一行内左右有交叠，或其中一方很窄）
        hx = max(0.0, min(ax1, bx1) - max(ax0, bx0))
        wa = max(1e-9, ax1 - ax0)
        wb = max(1e-9, bx1 - bx0)
        h_ratio = hx / min(wa, wb)
        return v_ratio >= 0.3 and h_ratio >= 0.3

    merged: list[dict[str, Any]] = []
    for box in base_boxes:
        best = dict(box)
        best["seen_in"] = 1
        best["variants"] = [base_name]
        for name, boxes in runs[1:]:
            partner = None
            for other in boxes:
                if _same_box(box, other):
                    partner = other
                    break
            if partner is None:
                continue
            best["seen_in"] = int(best.get("seen_in", 1)) + 1
            if name not in best["variants"]:
                best["variants"].append(name)
            bc = float(box.get("conf") or 0.0)
            pc = float(partner.get("conf") or 0.0)
            same_text = str(partner.get("text") or "").strip() == str(box.get("text") or "").strip()
            if same_text:
                if pc > bc:
                    best["text"] = partner.get("text")
                    best["conf"] = pc
            elif pc > bc + 0.15:
                best["text"] = partner.get("text")
                best["conf"] = pc
            elif abs(pc - bc) <= 0.15 and name == base_name:
                # 置信度接近时保留 base（最忠实）
                pass
        merged.append(best)

    # base 路没框、但别的路有框的 → 也收进来（可能 base 把浅色字漏了）
    for name, boxes in runs[1:]:
        for other in boxes:
            if not any(_same_box(box, other) for box in base_boxes):
                b = dict(other)
                b["seen_in"] = 1
                b["variants"] = [name]
                merged.append(b)
    return merged


def page_confidence(boxes: list[dict[str, Any]]) -> float:
    """整页的平均置信度（综合各路认同度）。"""
    if not boxes:
        return 0.0
    total = 0.0
    for b in boxes:
        c = float(b.get("conf") or 0.0)
        agree = int(b.get("seen_in") or 1)
        # 多路都认出来的框更可信
        total += min(1.0, c * (1.0 + 0.15 * (agree - 1)))
    return round(total / len(boxes), 4)
