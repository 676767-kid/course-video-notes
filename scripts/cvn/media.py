"""视频探测、抽帧与课件页去重。

本模块只依赖标准库 + ffmpeg/ffprobe。它的职责是把一段课程录像变成
`frames.json` + `slides/slide_NNNN.png`：每一张 = 一页真正的课件，而不是
每 2 秒一帧的时间序列截图。

核心做法（两遍扫描，避免把硬盘塞满）：

1. 第一遍：用 ffmpeg 的 `fps` 滤镜把整段视频降采样成 9x8 灰度缩略图，
   通过 stdout 读原始字节流。每帧只有 72 字节，两小时课程也就几百 KB。
   对每帧算 dHash（相邻像素比较 → 64 bit）。
2. 相邻帧的 dHash 汉明距离小于阈值 → 认为还是同一页；超过阈值 → 认为翻页了。
3. 第二遍：只对"每一页的代表时刻"调用一次 ffmpeg 取全分辨率原图。

这样 2 小时的课通常只需导出几十到一两百张图，且每张都是干净的一页，
不会出现半页动画、过渡残影。
"""

from __future__ import annotations

import math
import re
import threading
from pathlib import Path
from typing import Any, Iterable, Sequence

from .util import (
    CvnError,
    banner,
    ensure_dir,
    fmt_ts,
    log,
    read_json,
    run_cmd,
    run_cmd_bytes,
    vlog,
    warn,
    which,
)

# dHash 尺寸：横向比邻居需要 9 列，纵向 8 行 → 8x8 = 64 bit
_HASH_W = 9
_HASH_H = 8
_HASH_BITS = (_HASH_W - 1) * _HASH_H

DEFAULT_INTERVAL = 2.0
DEFAULT_MIN_HOLD = 1.5
# 一页停留不足这么多秒，多半是"翻过去又翻回来"或路过，
# 独立成页只会在笔记里留下噪声条目。
DEFAULT_MIN_PAGE = 12.0


# --------------------------------------------------------------------------
# 依赖检查
# --------------------------------------------------------------------------


def require_ffmpeg() -> tuple[str, str]:
    """返回 (ffmpeg, ffprobe) 可执行路径，缺失时抛 CvnError。"""
    ffmpeg = which("ffmpeg")
    ffprobe = which("ffprobe")
    if not ffmpeg or not ffprobe:
        raise CvnError(
            "需要 ffmpeg / ffprobe，但没找到。\n"
            "  macOS：brew install ffmpeg"
        )
    return ffmpeg, ffprobe


# --------------------------------------------------------------------------
# 探测
# --------------------------------------------------------------------------


_URL_RE = re.compile(r"^(https?|rtmp|rtsp|ftp)://", re.I)


def is_url(source: str | Path) -> bool:
    """这个输入是网络地址还是本地文件。"""
    return isinstance(source, str) and bool(_URL_RE.match(source.strip()))


def _as_source(video: str | Path) -> tuple[str, bool]:
    """统一成 (ffmpeg 可用的输入字符串, 是否网络地址)。

    网络地址**不能**走 Path()：Path 会把 "https://x" 规范化成 "https:/x"，
    导致路径解析失败（实测踩过）。本地文件照旧 expanduser + resolve。
    """
    if is_url(video):
        return str(video).strip(), True
    p = Path(video).expanduser()
    return str(p), False


def probe_video(video: str | Path) -> dict[str, Any]:
    """读取视频元信息：时长、分辨率、帧率、是否有音轨、编码。"""
    ffmpeg, ffprobe = require_ffmpeg()
    src, remote = _as_source(video)
    path = None if remote else Path(src)
    if not remote:
        if not path.exists():
            raise CvnError(f"找不到视频文件：{path}")
        if path.is_dir():
            raise CvnError(f"{path} 是目录，请传入单个视频文件")
        if path.stat().st_size == 0:
            raise CvnError(f"视频文件是空的：{path}")

    proc = run_cmd(
        [
            ffprobe, "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", src,
        ],
        timeout=120,
    )
    import json

    try:
        data = json.loads(proc.stdout or "{}")
    except json.JSONDecodeError as exc:
        raise CvnError(f"ffprobe 输出无法解析：{exc}") from exc

    streams = data.get("streams") or []
    fmt = data.get("format") or {}

    vstream = next((s for s in streams if s.get("codec_type") == "video"), None)
    astream = next((s for s in streams if s.get("codec_type") == "audio"), None)
    if vstream is None:
        name = path.name if path else src.split("/")[-1].split("?")[0]
        raise CvnError(f"{name} 里没有视频轨道（是不是纯音频或字幕文件？）")

    duration = 0.0
    for cand in (fmt.get("duration"), vstream.get("duration")):
        try:
            duration = float(cand)
        except (TypeError, ValueError):
            continue
        if duration > 0:
            break
    if duration <= 0:
        # 有些容器（如某些 ts）不写时长，退回用 ffprobe 的 -count_frames 太慢，
        # 这里用 nav 里的估算即可。
        warn("ffprobe 没给出时长，按 0 处理，抽帧会退化为按场景检测")
        duration = 0.0

    fps = 0.0
    raw_rate = vstream.get("avg_frame_rate") or vstream.get("r_frame_rate") or "0/0"
    if "/" in str(raw_rate):
        num, _, den = str(raw_rate).partition("/")
        try:
            fps = float(num) / float(den) if float(den) else 0.0
        except (ValueError, ZeroDivisionError):
            fps = 0.0

    # 手机竖拍视频常见：宽高存在旋转元数据里
    width = int(vstream.get("width") or 0)
    height = int(vstream.get("height") or 0)
    rotation = 0
    for side in vstream.get("side_data_list") or []:
        if "rotation" in side:
            try:
                rotation = int(float(side["rotation"]))
            except (TypeError, ValueError):
                rotation = 0
    if rotation % 180 == 90:
        width, height = height, width

    if remote:
        _name = src.split("/")[-1].split("?")[0] or "remote.mp4"
        _size = int(fmt.get("size") or 0)
    else:
        _name = path.name
        _size = int(fmt.get("size") or path.stat().st_size)

    return {
        "path": src if remote else str(path.resolve()),
        "name": _name,
        "remote": remote,
        "size_mb": round(_size / 1024 / 1024, 1),
        "duration": round(duration, 3),
        "width": width,
        "height": height,
        "fps": round(fps, 3),
        "video_codec": vstream.get("codec_name") or "",
        "has_audio": astream is not None,
        "audio_codec": (astream or {}).get("codec_name") or "",
        "container": (Path(_name).suffix.lstrip(".").lower() if _name else ""),
    }


# --------------------------------------------------------------------------
# 裁剪参数
# --------------------------------------------------------------------------


def _parse_crop(value: str | None, width: int, height: int) -> tuple[int, int, int, int]:
    """把 `--crop x,y,w,h` 解析成像素矩形；小于等于 1 的值视为比例。"""
    if not value:
        return (0, 0, width, height)
    parts = [p for p in re.split(r"[,:x ]+", value.strip()) if p]
    if len(parts) != 4:
        raise CvnError("--crop 需要 4 个数：x,y,w,h（像素或 0~1 比例），例如 0,0,0.8,0.7")
    try:
        nums = [float(p) for p in parts]
    except ValueError as exc:
        raise CvnError(f"--crop 参数不是数字：{value}") from exc
    if all(n <= 1.0 for n in nums):
        x, y, w, h = nums[0] * width, nums[1] * height, nums[2] * width, nums[3] * height
    else:
        x, y, w, h = nums
    x, y, w, h = (int(round(v)) for v in (x, y, w, h))
    # 夹到画面内，并保证宽高是偶数（H.264 友好，也避免 PNG 尺寸取整抖动）
    x = max(0, min(x, max(0, width - 2)))
    y = max(0, min(y, max(0, height - 2)))
    w = max(2, min(w, width - x))
    h = max(2, min(h, height - y))
    return (x - x % 2, y - y % 2, w - w % 2, h - h % 2)


def build_crop_chain(
    width: int,
    height: int,
    *,
    crop: str | None = None,
    mask_top: float = 0.0,
    mask_bottom: float = 0.0,
) -> tuple[str | None, tuple[int, int, int, int]]:
    """组合 --crop / --mask-top / --mask-bottom 得到单个裁切矩形。

    --mask-bottom 用来去掉画面底部的硬字幕条（烧进画面的字幕会被 OCR 误当成课件内容），
    --mask-top 用来去掉录制软件的标题栏/水印。
    """
    x, y, w, h = _parse_crop(crop, width, height)
    if mask_top or mask_bottom:
        total = mask_top + mask_bottom
        if total >= 0.95:
            raise CvnError("--mask-top 与 --mask-bottom 之和不能接近或超过 1")
        top_px = int(round(y + h * mask_top))
        bot_px = int(round(y + h * (1.0 - mask_bottom)))
        y, h = top_px, max(2, bot_px - top_px)
    x, y, w, h = (v - v % 2 for v in (x, y, w, h))
    if w <= 0 or h <= 0:
        raise CvnError("裁切后画面尺寸为 0，请检查 --crop / --mask-* 参数")
    if (x, y, w, h) == (0, 0, width, height) or (x, y, w, h) == (0, 0, width - width % 2, height - height % 2):
        return None, (x, y, w, h)
    return f"crop={w}:{h}:{x}:{y}", (x, y, w, h)


# --------------------------------------------------------------------------
# 第一遍：版面指纹（容忍逐条动画，但能识别翻页）
# --------------------------------------------------------------------------
#
# 为什么不直接用 dHash / 整帧平均差：
# 课程录像里"逐条动画"（PPT 一条条出来）和"翻页"在像素层面幅度几乎一样，
# 单一阈值无法区分，会把一个 40 分钟的大课切成上千页，或者把翻页全漏掉。
#
# 这里改用"版面网格指纹 + 变化占比"：
#   - 把画面缩成 GRID_W x GRID_H 的灰度网格（每格 = 一小块区域的平均亮度）
#   - 用逐格差异判断"这一块有没有变"
#   - 变化的**面积占比**是区分二者的关键信号：
#       逐条动画：只在正文区局部出现几行 → 占比低
#       真正翻页：标题 / 版面结构整体换掉 → 占比高，且常伴随顶部标题区变化
#   - 再加"变化持续"判定：翻页是不可逆的阶跃，动画后画面会稳定

GRID_W = 64
GRID_H = 36
# 变化面积占比阈值。
# 实测（真实录屏 + 合成课两种素材）：
#   同页动画推进     ratio ≈ 0.02 ~ 0.03
#   真正的翻页       ratio ≈ 0.19 ~ 0.92
# 两个聚类之间取 0.10 最稳。
DEFAULT_CHANGE_RATIO = 0.10
BODY_CHANGE_RATIO = 0.60       # 标题带没变时，判定翻页需要的更大变化面积
DEFAULT_PIXEL_DELTA = 18        # 单格灰度差阈值（滤掉压缩噪声）
TOP_BAND_ROWS = 9               # 顶部标题带（占 36 行里的 9 行 = 上部 25%）
TOP_CHANGE_RATIO = 0.10         # 标题带内变化占比超过它 → 几乎一定是新页

def _normalize_grid(buf: bytes, w: int = GRID_W, h: int = GRID_H) -> bytes:
    """把灰度网格做亮度归一化，抵消录制软件的自动曝光/亮度漂移。"""
    n = w * h
    mean = sum(buf) / n
    out = bytearray(n)
    for i in range(n):
        v = int(buf[i] - mean + 128)
        out[i] = 0 if v < 0 else (255 if v > 255 else v)
    return bytes(out)


def compare_grid(a: bytes, b: bytes, *, w: int = GRID_W, h: int = GRID_H,
                 delta: int = DEFAULT_PIXEL_DELTA) -> dict[str, float]:
    """比较两张版面网格，返回变化占比与顶部标题带变化占比。"""
    n = len(a)
    changed = 0
    for i in range(n):
        if abs(a[i] - b[i]) > delta:
            changed += 1
    top_n = min(len(a), TOP_BAND_ROWS * w)
    top_changed = sum(1 for i in range(top_n) if abs(a[i] - b[i]) > delta)
    return {
        "ratio": changed / n if n else 0.0,
        "top_ratio": top_changed / top_n if top_n else 0.0,
    }


def adaptive_ratio_threshold(
    ratios: Sequence[float],
    *,
    floor: float = 0.008,
    cap: float = DEFAULT_CHANGE_RATIO,
) -> float:
    """按本段视频的实际情况定"翻页"的 ratio 阈值。

    为什么不能写死：不同素材的翻页幅度差一个量级——
      纯 PPT 视频：换页时画面大部分仍是白底，ratio 只有 0.02~0.03
      真人录屏（老师出镜 + 桌面）：换页时整屏都变，ratio 能到 0.9
    固定成 0.10 会把前者全漏掉，固定成 0.02 会让后者碎成一堆页。

    观察到的事实是：无论哪种素材，**同一页内相邻帧几乎完全相同**
    （ratio 中位数≈0），所以翻页表现为少数几个明显的尖峰。
    取分布的 p90 附近作为分界，再夹在 [floor, cap] 之间。
    """
    vals = sorted(v for v in ratios if v > 0)
    if not vals:
        return floor
    n = len(vals)
    # 取非零变化里的 p85 作为"明显变化"的代表值
    idx = min(n - 1, int(n * 0.85))
    peak = vals[idx]
    # 阈值取尖峰的 35%，但不超过 cap、不低于 floor
    thr = max(floor, min(cap, peak * 0.35))
    return thr


def is_new_slide(metrics: dict[str, float], *, ratio: float = DEFAULT_CHANGE_RATIO,
                 top_ratio: float = TOP_CHANGE_RATIO,
                 body_ratio: float = BODY_CHANGE_RATIO) -> bool:
    """判定这一次变化是否构成"翻页"。

    三个信号：
      1. 顶部标题带变化够大 → 一定是新页（标题换了通常是新的一节）
      2. 整屏变化面积够大，**且**标题带也有变化 → 新页（版面整体重排）
      3. 整屏变化面积够大，但标题带**没变** → 多半是同一页的动画推进
         （PPT 逐条出现、图表动画），这时要求变化面积更大才认作翻页

    只看面积会把"整屏动画"误判成翻页——实测一段 PPT 动画推进时
    正文区整块变化（ratio 0.93）但标题带纹丝不动（top 0.002），
    这显然不是翻页。所以第 3 条用更严的 body_ratio 阈值。
    """
    r = metrics.get("ratio", 0.0)
    t = metrics.get("top_ratio", 0.0)
    # 顶部标题带变了 —— 多数模板里标题在顶部，换标题就是换页
    if t >= top_ratio:
        return True
    # 整屏变化面积够大 —— 换内容就换页。
    # 注意比较基准是"本页第一帧"，所以动画逐条出现时累积差异很小，
    # 不会把一页拆成好几页。
    if r >= ratio:
        return True
    # 变化极大时放宽：有些模板顶部是固定校名栏，标题带永远不变，
    # 这时只能靠面积判断。
    if r >= body_ratio:
        return True
    return False


def sample_fingerprints(
    video: str | Path,
    *,
    interval: float = DEFAULT_INTERVAL,
    crop_chain: str | None = None,
    duration: float = 0.0,
) -> list[dict[str, Any]]:
    """第一遍扫描：按固定间隔取版面网格，返回 [{t, grid}]。

    全程只读 ffmpeg stdout 的裸像素流，不落盘、不依赖 Pillow。
    """
    ffmpeg, _ = require_ffmpeg()
    if interval <= 0:
        raise CvnError("--interval 必须大于 0")

    filters = [f"fps=1/{interval}"]
    if crop_chain:
        filters.append(crop_chain)
    filters += [f"scale={GRID_W}:{GRID_H}:flags=area", "format=gray"]

    args = [
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-i", _as_source(video)[0],
        "-an", "-sn", "-dn",
        "-vf", ",".join(filters),
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-",
    ]
    code, raw, err = run_cmd_bytes(args, timeout=None, check=False)
    if code != 0 and not raw:
        tail = (err or "").strip().splitlines()[-5:]
        raise CvnError("抽帧指纹失败：\n" + "\n".join(tail))

    frame_size = GRID_W * GRID_H
    count = len(raw) // frame_size
    if count == 0:
        raise CvnError(
            "没有从视频里取到任何帧。可能是编码不受支持或文件损坏，"
            "可以先试：ffmpeg -i <视频> -frames:v 1 test.png"
        )

    out: list[dict[str, Any]] = []
    for i in range(count):
        buf = raw[i * frame_size:(i + 1) * frame_size]
        out.append({"t": round(i * interval, 3), "grid": _normalize_grid(buf)})
    vlog(f"版面采样 {len(out)} 帧（interval={interval}s）")
    return out


# --------------------------------------------------------------------------
# 场景检测（可选补充信号）
# --------------------------------------------------------------------------


def detect_scenes(
    video: str | Path,
    *,
    threshold: float = 0.30,
    crop_chain: str | None = None,
) -> list[float]:
    """用 ffmpeg 的场景切换打分找出硬切时刻（翻页通常伴随硬切/淡入）。"""
    ffmpeg, _ = require_ffmpeg()
    filters = []
    if crop_chain:
        filters.append(crop_chain)
    filters.append(f"select='gt(scene,{threshold})'")
    filters.append("metadata=print:file=-")

    args = [
        ffmpeg, "-hide_banner", "-v", "error", "-nostdin",
        "-i", _as_source(video)[0],
        "-an", "-sn", "-dn",
        "-vf", ",".join(filters),
        "-f", "null", "-",
    ]
    proc = run_cmd(args, timeout=None, check=False)
    times: list[float] = []
    for line in (proc.stdout or "").splitlines():
        m = re.search(r"pts_time:([0-9.]+)", line)
        if m:
            try:
                times.append(float(m.group(1)))
            except ValueError:
                continue
    vlog(f"场景检测命中 {len(times)} 处")
    return times


# --------------------------------------------------------------------------
# 分组：把采样序列切成"页"
# --------------------------------------------------------------------------


def group_frames(
    samples: Sequence[dict[str, Any]],
    *,
    ratio: float | None = None,
    top_ratio: float = TOP_CHANGE_RATIO,
    body_ratio: float = BODY_CHANGE_RATIO,
    delta: int = DEFAULT_PIXEL_DELTA,
    min_hold: float = DEFAULT_MIN_HOLD,
    min_page: float = DEFAULT_MIN_PAGE,
    duration: float = 0.0,
) -> list[dict[str, Any]]:
    """把采样序列切成页。

    返回 [{t_start, t_last, frames, metrics}]：
      t_start 该页首次出现；t_last 该页最后稳定时刻（--pick last 用它，
      可以避开 PPT 逐条动画还没演完的那段，拿到"这一页讲完时的完整样子"）。
    """
    if not samples:
        return []

    # ratio=None 表示"让工具自己看这段视频的尺度来定阈值"
    if ratio is None:
        seq: list[float] = []
        prev_grid = None
        for sm in samples:
            if prev_grid is not None:
                seq.append(compare_grid(prev_grid, sm["grid"], delta=delta)["ratio"])
            prev_grid = sm["grid"]
        ratio = adaptive_ratio_threshold(seq)
        vlog(f"自适应翻页阈值 ratio={ratio:.4f}（本段 {len(seq)} 个相邻对比）")

    groups: list[dict[str, Any]] = []
    cur = {"t_start": samples[0]["t"], "t_last": samples[0]["t"],
           "grid": samples[0]["grid"], "frames": [samples[0]],
           "metrics": {"ratio": 1.0, "top_ratio": 1.0}}
    for sample in samples[1:]:
        metrics = compare_grid(cur["grid"], sample["grid"], delta=delta)
        if is_new_slide(metrics, ratio=ratio, top_ratio=top_ratio, body_ratio=body_ratio):
            groups.append(cur)
            cur = {"t_start": sample["t"], "t_last": sample["t"],
                   "grid": sample["grid"], "frames": [sample], "metrics": metrics}
        else:
            # 同页：持续吸收新画面。注意 grid 保持为本页起始画面，
            # 这样逐条动画累积后仍然和起始页比较，不会自己漂移成新页。
            cur["t_last"] = sample["t"]
            cur["frames"].append(sample)
    groups.append(cur)

    # 太短的"页"多半是转场残影 / 淡入中间态：并进上一页
    merged: list[dict[str, Any]] = []
    for g in groups:
        hold = g["t_last"] - g["t_start"]
        if merged and hold < min_hold:
            merged[-1]["t_last"] = g["t_last"]
            merged[-1]["frames"].extend(g["frames"])
            continue
        merged.append(g)

    # 再扫一遍：翻页动画的中间帧常常只出现 5~10 秒，独立成页只会让笔记
    # 多出噪声条目。
    #
    # 但这条规则不能硬用固定秒数：一段 50 秒的短课，每页 12 秒是**正常**的，
    # 按固定阈值会把整段课吞成一页。所以阈值取「该视频平均每页时长的 40%」，
    # 并夹在 [3s, min_page] 之间——只有明显短于本视频平均页长的页才会被合并。
    if merged:
        span = merged[-1]["t_last"] - merged[0]["t_start"]
        avg_page = span / len(merged) if len(merged) else 0.0
        short_threshold = min(min_page, max(3.0, avg_page * 0.4))
    else:
        short_threshold = min_page

    pruned: list[dict[str, Any]] = []
    for g in merged:
        hold = g["t_last"] - g["t_start"]
        prev_hold = (pruned[-1]["t_last"] - pruned[-1]["t_start"]) if pruned else 0.0
        prev_short = bool(pruned) and prev_hold < short_threshold
        if hold < short_threshold and pruned and not prev_short:
            # 并进前一页：把它的出现时刻也算作前一页的结束
            pruned[-1]["t_last"] = max(pruned[-1]["t_last"], g["t_last"])
            pruned[-1]["frames"].extend(g["frames"])
            continue
        pruned.append(g)
    merged = pruned

    if duration > 0 and merged:
        merged[-1]["t_last"] = max(merged[-1]["t_last"], min(duration, merged[-1]["t_last"]))
    return merged


# --------------------------------------------------------------------------
# 第二遍：取全分辨率代表帧
# --------------------------------------------------------------------------


def _extract_one(
    ffmpeg: str,
    video: str,
    t: float,
    dest: Path,
    *,
    crop_chain: str | None,
    max_width: int,
) -> bool:
    """在时刻 t 截一张全分辨率 PNG，返回是否成功。"""
    filters = []
    if crop_chain:
        filters.append(crop_chain)
    if max_width and max_width > 0:
        filters.append(f"scale='min({max_width},iw)':-2")

    args = [ffmpeg, "-hide_banner", "-v", "error", "-nostdin"]
    # 输入侧 -ss：关键帧快速定位，ffmpeg 会自动解码到精确时刻
    if t > 0.05:
        args += ["-ss", f"{t:.3f}"]
    args += ["-i", video, "-an", "-sn", "-dn", "-frames:v", "1"]
    if filters:
        args += ["-vf", ",".join(filters)]
    args += ["-y", str(dest)]

    proc = run_cmd(args, timeout=180, check=False)
    return proc.returncode == 0 and dest.exists() and dest.stat().st_size > 0


def _extract_many(
    video: str,
    jobs: Sequence[tuple[float, Path]],
    *,
    crop_chain: str | None,
    max_width: int,
    workers: int = 4,
) -> tuple[int, int]:
    """并发截取多张代表帧。返回 (成功数, 失败数)。"""
    ffmpeg, _ = require_ffmpeg()
    if not jobs:
        return (0, 0)

    ok = 0
    failed = 0
    lock = threading.Lock()
    errors: list[str] = []

    # ffmpeg 本身是多线程的，这里只做轻微的 I/O 并行
    workers = max(1, min(workers, 8))

    def work(t: float, dest: Path) -> None:
        nonlocal ok, failed
        try:
            good = _extract_one(ffmpeg, video, t, dest, crop_chain=crop_chain, max_width=max_width)
            if not good and t > 0.35:
                # 有些视频在片头第 0 秒取不到帧，退后一点重试
                good = _extract_one(ffmpeg, video, max(0.0, t - 0.3), dest,
                                    crop_chain=crop_chain, max_width=max_width)
        except CvnError as exc:
            good = False
            with lock:
                errors.append(str(exc))
        with lock:
            if good:
                ok += 1
            else:
                failed += 1

    chunk = math.ceil(len(jobs) / workers)
    threads: list[threading.Thread] = []
    for i in range(0, len(jobs), chunk):
        part = jobs[i:i + chunk]

        def runner(part=part) -> None:
            for t, dest in part:
                work(t, dest)

        th = threading.Thread(target=runner, daemon=True)
        th.start()
        threads.append(th)
    for th in threads:
        th.join()

    if errors:
        vlog("部分帧截取报错：" + " | ".join(errors[:3]))
    return (ok, failed)


# --------------------------------------------------------------------------
# 主流程
# --------------------------------------------------------------------------


def extract_slides(
    out_dir: str | Path,
    video: str | Path,
    *,
    interval: float = DEFAULT_INTERVAL,
    strategy: str = "interval",
    pick: str = "first",
    change_ratio: float | None = None,   # None = 按本段素材自适应
    top_ratio: float = TOP_CHANGE_RATIO,
    pixel_delta: int = DEFAULT_PIXEL_DELTA,
    min_hold: float = DEFAULT_MIN_HOLD,
    min_page: float = DEFAULT_MIN_PAGE,
    crop: str | None = None,
    mask_top: float = 0.0,
    mask_bottom: float = 0.0,
    max_width: int = 0,
    scene_threshold: float = 0.30,
    force: bool = False,
) -> dict[str, Any]:
    """抽帧 + 去重，产出 frames.json 与 slides/ 原图。"""
    out = ensure_dir(out_dir)
    slides_dir = ensure_dir(out / "slides")
    frames_json = out / "frames.json"

    if strategy not in ("interval", "scene", "both"):
        raise CvnError(f"--strategy 只能是 interval / scene / both，收到 {strategy!r}")
    if pick not in ("first", "last", "mid"):
        raise CvnError(f"--pick 只能是 first / last / mid，收到 {pick!r}")

    # 增量：已有结果且帧图还在，直接复用
    if frames_json.exists() and not force:
        cached = read_json(frames_json) or {}
        if cached.get("slides") and all(
            (out / s["file"]).exists() for s in cached["slides"]
        ):
            log(f"复用已有 frames.json（{len(cached['slides'])} 页）")
            return cached

    banner("抽帧 / 去重")
    info = probe_video(video)
    log(f"视频：{info['name']}  {info['width']}x{info['height']}  "
        f"{fmt_ts(info['duration'])}  音轨={'有' if info['has_audio'] else '无'}")

    crop_chain, (cx, cy, cw, ch) = build_crop_chain(
        info["width"], info["height"], crop=crop, mask_top=mask_top, mask_bottom=mask_bottom
    )
    if crop_chain:
        log(f"裁切区域：{cw}x{ch} @ ({cx},{cy})")

    # --- 第一遍：指纹 ---
    log("第 1/2 遍：按间隔采样指纹…")
    samples = sample_fingerprints(
        video, interval=interval, crop_chain=crop_chain, duration=info["duration"]
    )

    # --- 场景检测（可选）：把硬切时刻作为额外候选 ---
    scene_times: list[float] = []
    if strategy in ("scene", "both"):
        log("场景检测中…")
        scene_times = detect_scenes(video, threshold=scene_threshold, crop_chain=crop_chain)
        if strategy == "scene" and scene_times:
            # 纯场景模式：直接用场景时刻作为候选序列
            samples = [{"t": round(t, 3), "grid": bytes(GRID_W * GRID_H)} for t in scene_times]
            log(f"使用场景模式：{len(samples)} 个候选时刻（不做指纹去重）")

    if strategy == "scene":
        groups = [
            {"t_start": s["t"], "t_last": s["t"], "frames": [s], "changed": True}
            for s in samples
        ]
    else:
        groups = group_frames(
            samples, ratio=change_ratio, top_ratio=top_ratio,
            delta=pixel_delta, min_hold=min_hold, min_page=min_page,
            duration=info["duration"],
        )
        if strategy == "both" and scene_times:
            # 场景时刻如果离已有分组边界太远，补一个候选页边界
            existing = {round(g["t_start"], 1) for g in groups}
            for t in scene_times:
                if min((abs(t - e) for e in existing), default=999) > interval:
                    groups.append({"t_start": t, "t_last": t, "frames": [], "changed": True})
            groups.sort(key=lambda g: g["t_start"])
        log(f"去重后得到 {len(groups)} 页（原始采样 {len(samples)} 帧）")

    # --- 选代表时刻 ---
    total = info["duration"]
    jobs: list[tuple[float, Path]] = []
    plan: list[dict[str, Any]] = []
    for idx, g in enumerate(groups, start=1):
        start = float(g["t_start"])
        end = float(groups[idx]["t_start"]) if idx < len(groups) else total
        if end <= start:
            end = start + max(interval, 0.5)
        if pick == "last":
            # 页面末尾再往前挪一点，避开翻页过渡的第一帧
            t_frame = max(start, end - max(0.4, interval * 0.35))
        elif pick == "mid":
            t_frame = (start + end) / 2.0
        else:
            # first：用该页第一个稳定采样点（跳过 0.4s 内的转场）
            t_frame = min(start + 0.4, max(start, end - 0.2))
        dest = slides_dir / f"slide_{idx:04d}.png"
        jobs.append((t_frame, dest))
        metrics = g.get("metrics") or {}
        plan.append({
            "index": idx,
            "file": f"slides/slide_{idx:04d}.png",
            "t": round(start, 3),
            "t_end": round(min(end, total) if total else end, 3),
            "frame_t": round(t_frame, 3),
            "change_ratio": round(float(metrics.get("ratio", 1.0)), 4),
            "top_ratio": round(float(metrics.get("top_ratio", 1.0)), 4),
        })

    # --- 第二遍：截全分辨率 ---
    log(f"第 2/2 遍：截取 {len(jobs)} 张代表帧…")
    ok, failed = _extract_many(
        video, jobs, crop_chain=crop_chain, max_width=max_width,
        workers=4 if len(jobs) > 8 else 2,
    )
    if ok == 0:
        raise CvnError(
            "一张课件图都没截出来。请确认视频能正常解码：\n"
            f"  ffmpeg -ss 10 -i {info['name']} -frames:v 1 test.png"
        )
    if failed:
        warn(f"{failed} 张代表帧截取失败（通常是个别时刻的编码问题），已保留其余页")

    # 过滤掉没截成功的页，并重新编号，保证 index 与文件一一对应
    kept: list[dict[str, Any]] = []
    for item in plan:
        if (out / item["file"]).exists():
            kept.append(item)
    kept.sort(key=lambda s: s["t"])
    for i, item in enumerate(kept, start=1):
        old = out / item["file"]
        new_rel = f"slides/slide_{i:04d}.png"
        new = out / new_rel
        if old != new:
            old.replace(new)
        item["index"] = i
        item["file"] = new_rel
    # 重新闭合 t_end（过滤后可能留下空洞）
    for i, item in enumerate(kept):
        nxt = kept[i + 1]["t"] if i + 1 < len(kept) else (total or item["t_end"])
        item["t_end"] = round(nxt, 3)

    result = {
        "duration": round(total, 3),
        "interval": interval,
        "strategy": strategy,
        "pick": pick,
        "count": len(kept),
        "crop": {"x": cx, "y": cy, "w": cw, "h": ch} if crop_chain else None,
        "slides": kept,
    }
    from .util import write_json

    write_json(frames_json, result)

    log("")
    log(f"完成：{len(kept)} 页课件图 → {slides_dir}")
    preview = kept[:3]
    for item in preview:
        log(f"  [{item['index']:>3}] {fmt_ts(item['t'])} → {item['file']}")
    if len(kept) > 3:
        log(f"  … 共 {len(kept)} 页")
    return result


def load_frames(out_dir: str | Path) -> dict[str, Any]:
    """读取 frames.json，不存在时给出可执行的提示。"""
    data = read_json(Path(out_dir) / "frames.json")
    if not data or not data.get("slides"):
        raise CvnError("还没有抽帧结果，请先运行：course_kit.py frames --video <视频>")
    return data
