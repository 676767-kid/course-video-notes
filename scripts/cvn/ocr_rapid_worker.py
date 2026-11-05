"""ocr_rapid_worker.py —— 在装了 rapidocr 的解释器里跑，不和主程序共享依赖。

协议：
    标准输入：每行一个图片绝对路径
    标准输出：每行一个 JSON
        {"file": "...", "width": 1920, "height": 1080,
         "lines": [{"box": [[x,y],...], "text": "...", "conf": 0.98}, ...]}

RapidOCR 自己会往 stdout 打日志，所以识别期间把 sys.stdout 重定向到 stderr，
结果统一写到保存下来的真实 stdout。这样主程序只看到 JSON 行。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    real_stdout = sys.stdout
    sys.stdout = sys.stderr  # 挡住 RapidOCR 的 loading 日志

    try:
        from PIL import Image
        from rapidocr_onnxruntime import RapidOCR
    except Exception as exc:  # pragma: no cover - 环境问题
        print(f"import failed: {exc}", file=sys.stderr)
        return 2

    try:
        engine = RapidOCR()
    except Exception as exc:  # pragma: no cover
        print(f"init failed: {exc}", file=sys.stderr)
        return 3

    for raw in sys.stdin:
        line = raw.strip()
        if not line:
            continue
        path = Path(line)
        try:
            with Image.open(path) as im:
                width, height = im.size
        except Exception as exc:
            print(json.dumps({"file": str(path), "error": str(exc)}, ensure_ascii=False),
                  file=real_stdout, flush=True)
            continue
        try:
            result, _elapse = engine(str(path))
        except Exception as exc:  # 单张失败不影响整批
            print(json.dumps({"file": str(path), "error": str(exc)}, ensure_ascii=False),
                  file=real_stdout, flush=True)
            continue
        lines = []
        for item in result or []:
            try:
                box, text, conf = item[0], item[1], item[2]
            except (TypeError, ValueError, IndexError):
                continue
            lines.append({"box": box, "text": text, "conf": float(conf)})
        print(
            json.dumps(
                {"file": str(path), "width": width, "height": height, "lines": lines},
                ensure_ascii=False,
            ),
            file=real_stdout,
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
