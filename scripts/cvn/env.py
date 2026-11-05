"""环境自检：判断这台机器上能跑哪些环节，并给出可执行的修复建议。

`course_kit.py probe` 调这里。设计原则：**永远不抛异常**，缺什么就老老实实
报告缺什么，让用户先看到一张清楚的体检表，而不是跑一半炸掉。
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from .util import CvnError, is_writable_dir, pick_temp_root, which

# 这些目录在部分沙箱/受限环境下不可写，需要显式指到别处
_RISKY_CACHES = (
    Path.home() / ".cache",
    Path.home() / ".dsh",
)


def _can_write(path: Path) -> bool:
    try:
        return is_writable_dir(path)
    except Exception:
        return False


def _try_import(name: str) -> tuple[bool, str]:
    try:
        mod = __import__(name)
        ver = getattr(mod, "__version__", "")
        return True, str(ver)
    except Exception as exc:  # noqa: BLE001 - 探测用途，任何异常都算"不可用"
        return False, type(exc).__name__


def probe_platform() -> dict[str, Any]:
    return {
        "os": platform.system(),
        "macos_version": platform.mac_ver()[0] or "",
        "python": sys.version.split()[0],
        "python_exe": sys.executable,
        "arch": platform.machine(),
    }


def probe_ffmpeg() -> dict[str, Any]:
    ffmpeg = which("ffmpeg")
    ffprobe = which("ffprobe")
    info: dict[str, Any] = {"ffmpeg": ffmpeg, "ffprobe": ffprobe, "ok": bool(ffmpeg and ffprobe)}
    if ffmpeg:
        try:
            out = subprocess.run([ffmpeg, "-version"], capture_output=True, text=True, timeout=20)
            first = (out.stdout or "").splitlines()[0] if out.stdout else ""
            info["version"] = first.replace("ffmpeg version ", "").split(" Copyright")[0]
        except Exception:
            info["version"] = ""
    return info


def probe_swift() -> dict[str, Any]:
    """Apple Vision OCR 走 Swift 编译版（最快、中文效果好）。"""
    swiftc = which("swiftc")
    result: dict[str, Any] = {"swiftc": swiftc, "ok": False}
    if not swiftc:
        return result
    try:
        proc = subprocess.run([swiftc, "--version"], capture_output=True, text=True, timeout=30)
        txt = (proc.stdout or "") + (proc.stderr or "")
        ver = ""
        for line in txt.splitlines():
            if "Swift version" in line or line.startswith("Apple Swift"):
                ver = line.strip()
                break
        result["version"] = ver
        result["ok"] = proc.returncode == 0
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    return result


def probe_pyobjc_vision() -> dict[str, Any]:
    ok_vision, _ = _try_import("Vision")
    ok_quartz, _ = _try_import("Quartz")
    return {"ok": bool(ok_vision and ok_quartz), "Vision": ok_vision, "Quartz": ok_quartz}


def probe_asr() -> dict[str, Any]:
    ok, ver = _try_import("faster_whisper")
    info: dict[str, Any] = {"faster_whisper": ok, "version": ver}
    if ok:
        # 模型缓存能不能写，直接决定第一次跑会不会失败
        hf_home = os.environ.get("HF_HOME") or str(Path.home() / ".cache" / "huggingface")
        info["hf_home"] = hf_home
        info["hf_home_writable"] = _can_write(Path(hf_home))
    try:
        info["temp_root"] = str(pick_temp_root())
    except CvnError:
        info["temp_root"] = ""
    info["ok"] = ok and bool(info.get("temp_root"))
    return info


def probe_pillow() -> dict[str, Any]:
    ok, ver = _try_import("PIL")
    return {"ok": ok, "version": ver}


def env_hints(report: dict[str, Any]) -> list[str]:
    """把探测结果翻译成"下一步该干什么"的中文提示。"""
    hints: list[str] = []
    if not report["ffmpeg"]["ok"]:
        hints.append("缺少 ffmpeg/ffprobe：macOS 执行 `brew install ffmpeg`，"
                     "其它系统用各自的包管理器安装。这是**必需**依赖。")

    ocr = report["ocr"]
    if not ocr["any_ok"]:
        hints.append("没有任何可用的 OCR 引擎：装 Xcode Command Line Tools（`xcode-select --install`）"
                     "可用 Apple Vision；或 `pip install pyobjc-framework-Vision`；"
                     "或 `brew install tesseract tesseract-lang`。")
    elif ocr.get("swift_vision") and not ocr.get("rapidocr"):
        hints.append("只有 Apple Vision：中文标题/艺术字容易认错（实测把「本课程总学分」读成"
                     "「苯颗椁思分苏」这类乱码）。建议装 RapidOCR 做主力引擎：\n"
                     "      uv venv ~/.venvs/cvn --python 3.12\n"
                     "      uv pip install --python ~/.venvs/cvn/bin/python rapidocr-onnxruntime")
    elif ocr.get("rapidocr") and ocr.get("swift_vision"):
        hints.append("RapidOCR + Apple Vision 都在：默认走 hybrid（RapidOCR 主识别，"
                     "Vision 补漏），中文课件准确率最好。")

    asr = report["asr"]
    if not asr["faster_whisper"]:
        hints.append("没有 faster-whisper：语音转写不可用。可以 `pip install faster-whisper`；"
                     "或者把已有字幕用 `--from-srt 字幕.srt` 导入。")
    elif not asr.get("hf_home_writable", True):
        hints.append(f"模型缓存目录 {asr.get('hf_home')} 不可写，"
                     "请设置 `export HF_HOME=~/.cvn-models` 后再跑转写。")
    if asr["faster_whisper"] and not report.get("hf_reachable", True):
        hints.append("连不上 HuggingFace 官方站点，请先 `export HF_ENDPOINT=https://hf-mirror.com`。")

    if not report["pillow"]["ok"]:
        hints.append("没有 Pillow：不影响主流程，但缩略图和图表检测会退化到纯 OCR。")
    return hints


def collect_report(*, check_network: bool = False) -> dict[str, Any]:
    from . import ocr_rapid

    report: dict[str, Any] = {
        "platform": probe_platform(),
        "ffmpeg": probe_ffmpeg(),
        "swift": probe_swift(),
        "pyobjc": probe_pyobjc_vision(),
        "asr": probe_asr(),
        "pillow": probe_pillow(),
        "rapidocr": ocr_rapid.probe(),
    }
    tesseract = which("tesseract")
    report["tesseract"] = {"path": tesseract, "ok": bool(tesseract)}

    report["ocr"] = {
        "swift_vision": report["swift"]["ok"],
        "pyobjc_vision": report["pyobjc"]["ok"],
        "tesseract": report["tesseract"]["ok"],
        "rapidocr": bool(report["rapidocr"].get("available")),
        "any_ok": any([
            report["swift"]["ok"],
            report["pyobjc"]["ok"],
            report["tesseract"]["ok"],
            bool(report["rapidocr"].get("available")),
        ]),
    }
    if report["ocr"]["rapidocr"]:
        report["ocr"]["preferred"] = "rapidocr"
    elif report["swift"]["ok"]:
        report["ocr"]["preferred"] = "swift-vision"
    elif report["pyobjc"]["ok"]:
        report["ocr"]["preferred"] = "pyobjc-vision"
    elif report["tesseract"]["ok"]:
        report["ocr"]["preferred"] = "tesseract"
    else:
        report["ocr"]["preferred"] = "none"

    if check_network:
        report["hf_reachable"] = _check_url("https://hf-mirror.com")
    else:
        report["hf_reachable"] = True

    report["hints"] = env_hints(report)
    report["ready"] = {
        "frames": report["ffmpeg"]["ok"],
        "ocr": report["ocr"]["any_ok"],
        "asr": report["asr"]["ok"],
        "full": report["ffmpeg"]["ok"] and report["ocr"]["any_ok"] and report["asr"]["ok"],
    }
    return report


def _check_url(url: str, timeout: float = 8.0) -> bool:
    try:
        import urllib.request

        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout):
            return True
    except Exception:  # noqa: BLE001
        return False


def format_report(report: dict[str, Any]) -> str:
    """人看的体检表。"""
    def mark(ok: bool) -> str:
        return "✔" if ok else "✘"

    p = report["platform"]
    lines: list[str] = []
    lines.append("课程录像 → 笔记  环境自检")
    lines.append("=" * 46)
    lines.append(f"系统      {p['os']} {p['macos_version']} ({p['arch']})")
    lines.append(f"Python    {p['python']}  {p['python_exe']}")
    lines.append("")

    ff = report["ffmpeg"]
    lines.append(f"{mark(ff['ok'])} ffmpeg    {ff.get('version') or '未安装'}  [抽帧必需]")

    ocr = report["ocr"]
    lines.append(f"{mark(ocr['any_ok'])} OCR 引擎  首选: {ocr['preferred']}")
    lines.append(f"    RapidOCR (PP-OCR)     {mark(ocr.get('rapidocr', False))}"
                 + (f"  {report.get('rapidocr', {}).get('python', '')}" if ocr.get("rapidocr") else ""))
    lines.append(f"    Apple Vision (Swift)  {mark(ocr['swift_vision'])}"
                 + (f"  {report['swift'].get('version', '')}" if ocr["swift_vision"] else ""))
    lines.append(f"    Apple Vision (PyObjC) {mark(ocr['pyobjc_vision'])}")
    lines.append(f"    tesseract             {mark(ocr['tesseract'])}"
                 + (f"  {report['tesseract']['path']}" if ocr["tesseract"] else ""))

    asr = report["asr"]
    if asr.get("ok") and not asr["faster_whisper"]:
        # 当前解释器没有，但 skill 的独立环境里有（cmd_probe 会回填 venv 字段）
        lines.append(f"{mark(True)} 语音转写  faster-whisper（独立环境）")
    else:
        lines.append(f"{mark(asr['ok'])} 语音转写  faster-whisper "
                     + (f"{asr.get('version', '')}" if asr["faster_whisper"] else "未安装"))
    if asr["faster_whisper"]:
        lines.append(f"    模型缓存  {asr.get('hf_home')} "
                     f"{mark(asr.get('hf_home_writable', False))}")

    pil = report["pillow"]
    lines.append(f"{mark(pil['ok'])} Pillow    {pil.get('version', '未安装')}  [可选，图像启发式]")
    lines.append("")

    ready = report["ready"]
    lines.append("可用能力：")
    lines.append(f"  抽帧去重   {mark(ready['frames'])}")
    lines.append(f"  课件识字   {mark(ready['ocr'])}")
    lines.append(f"  语音转写   {mark(ready['asr'])}")
    lines.append(f"  一条命令跑完 all  {mark(ready['full'])}")

    if report["hints"]:
        lines.append("")
        lines.append("建议：")
        for h in report["hints"]:
            lines.append(f"  • {h}")
    return "\n".join(lines)
