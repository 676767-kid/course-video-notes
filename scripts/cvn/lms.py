"""从在线教学平台（LMS）取课程录像直链。

为什么需要这个模块
------------------
很多学校的录播系统要经过多层鉴权才能拿到视频，手工点很繁琐：

    1. 统一身份认证（CAS / OAuth）登录，拿到浏览器 cookie
    2. 前端调若干接口，最终返回一个带 token 的播放页
    3. 播放页内部再调接口，返回对象存储的**预签名直链**

预签名直链有效期通常是几天，**拿到后就是普通 HTTPS mp4**，可以交给
ffmpeg 流式读取（不必先下载整个文件）。这是「给个地址直接出笔记」的基础。

设计：站点适配器
----------------
不同学校的系统千差万别，所以这里不写死某个学校，而是把差异抽成**适配器**：

    SiteAdapter
      ├── base_url            站点根地址
      ├── list_endpoint       课程内容列表接口（返回 activities）
      ├── lesson_url()        某节课的页面地址
      └── pick_screen()       从多路录像里挑出带课件的那一路

各校系统千差万别，本模块**不内置任何学校的适配器**。请按本校实际情况在
`ADAPTERS` 里注册一个 `SiteAdapter` 实例——通常只需要改三处：接口路径、
字段名、以及哪一路是电脑屏幕画面。

本模块只做「取链接」，不做下载。下载/流式处理交给 media.py。

用法
----
    python scripts/course_kit.py lms login --site my-school --course 12345
    python scripts/course_kit.py lms list  --site my-school --course 12345
    python scripts/course_kit.py lms urls  --site my-school --course 12345 -o urls.json

登录态存在 ~/.cache/course-video-notes/<site>_state.json（Playwright
storageState），过期重新 login 即可。
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

CACHE_DIR = Path.home() / ".cache" / "course-video-notes"


# ==========================================================================
# 站点适配器
# ==========================================================================


@dataclass
class SiteAdapter:
    """描述一个录播站点的抓取方式。

    新增学校时，只需实现这个类的字段，必要时覆盖 pick_screen。
    """

    name: str
    base_url: str
    # 课程内容列表接口，{course} 会被替换成课程 id
    list_endpoint: str
    # 「电脑屏幕」画面的识别正则（用于在多路录像中挑出带课件的那一路）
    screen_pattern: str = r"/2-[^/]*\.mp4"
    # 「教室摄像头」画面的识别正则
    camera_pattern: str = r"/0-[^/]*\.mp4"
    # 抓播放信息时，响应 URL 里包含这个关键字即认为命中
    capture_url_keyword: str = "/captures/"
    # 课程页面路径模板
    lesson_page: str = "/course/{course}/lesson"
    # 从列表接口响应中提取 activities
    parse_list: Optional[Callable[[Dict[str, Any]], List[Dict[str, Any]]]] = field(default=None)

    def __post_init__(self) -> None:
        if self.parse_list is None:
            self.parse_list = self._default_parse_list

    @staticmethod
    def _default_parse_list(data: Dict[str, Any]) -> List[Dict[str, Any]]:
        return [
            {"activity_id": a.get("id"), "title": a.get("title")}
            for a in (data.get("activities") or [])
        ]

    @property
    def state_path(self) -> Path:
        return CACHE_DIR / f"{self.name}_state.json"

    def lesson_url(self, course: Any) -> str:
        return self.base_url + self.lesson_page.format(course=course)

    def pick_screen(self, videos: List[Dict[str, Any]]) -> Optional[str]:
        """在多路录像里挑出电脑屏幕那一路的直链。"""
        for v in videos:
            url = str(v.get("download_url") or v.get("url") or "")
            if re.search(self.screen_pattern, url):
                return url
        return None

    def pick_camera(self, videos: List[Dict[str, Any]]) -> Optional[str]:
        """挑出教室摄像头那一路（用户明确要时才用）。"""
        for v in videos:
            url = str(v.get("download_url") or v.get("url") or "")
            if re.search(self.camera_pattern, url):
                return url
        return None


# --------------------------------------------------------------------------
# 站点适配器注册表
# --------------------------------------------------------------------------
# 本工具不内置任何特定学校的配置。请按本校录播系统的实际情况添加，例如：
#
#   MY_SCHOOL = SiteAdapter(
#       name="my-school",
#       base_url="https://lms.example.edu.cn",
#       list_endpoint="/api/course/{course}/coursewares?page=1&page_size=100",
#       screen_pattern=r"/screen[^/]*\.mp4",   # 哪一路是电脑屏幕
#       camera_pattern=r"/camera[^/]*\.mp4",
#   )
#   ADAPTERS[MY_SCHOOL.name] = MY_SCHOOL
#
# 常见差异就是三处：接口路径、返回字段名、以及哪一路是电脑屏幕画面。

ADAPTERS: Dict[str, SiteAdapter] = {}


def get_adapter(site: str) -> SiteAdapter:
    if site not in ADAPTERS:
        known = "、".join(sorted(ADAPTERS)) or "（空）"
        raise RuntimeError(
            f"没有名为 {site!r} 的站点适配器。当前已注册：{known}\n"
            "请在 scripts/cvn/lms.py 里按本校情况写一个 SiteAdapter，"
            "并注册到 ADAPTERS（通常只需改接口路径和字段名）。"
        )
    return ADAPTERS[site]


# ==========================================================================
# 浏览器后端：优先 Python playwright，没有就用 Node 版 playwright-core 桥接
# ==========================================================================
#
# 有些环境里 Python 侧装不了 playwright（或版本不匹配），但 node_modules
# 里有 playwright-core。为不强制用户装东西，这里允许走 Node 子进程：
# Python 负责组织与落盘，Node 负责开浏览器抓数据。

_NODE_PLAYWRIGHT_CANDIDATES = (
    Path.cwd() / "node_modules",
    Path.home() / ".dsh" / "skills" / "huashu-design" / "node_modules",
    Path(__file__).resolve().parent.parent.parent / "node_modules",
)


def _find_node_playwright() -> Optional[Path]:
    """找一个能 require 的 playwright-core 目录。"""
    for base in _NODE_PLAYWRIGHT_CANDIDATES:
        if base.name == "node_modules":
            cand = base / "playwright-core"
        else:
            cand = base / "node_modules" / "playwright-core"
        if cand.exists():
            return cand
    return None


def _node_bin() -> Optional[str]:
    import shutil

    return shutil.which("node")


def backend() -> str:
    """当前用哪种浏览器后端：python / node / none。"""
    try:
        import playwright  # noqa: F401

        return "python"
    except ImportError:
        pass
    if _node_bin() and _find_node_playwright():
        return "node"
    return "none"


def _run_node(script: str, *, timeout: float = 900.0) -> str:
    """把一段 Node 代码交给 node 执行，返回 stdout。"""
    import subprocess
    import tempfile

    node = _node_bin()
    nm = _find_node_playwright()
    if not node or not nm:
        raise RuntimeError(
            "没有可用的浏览器自动化后端。任选其一：\n"
            "  1) pip install playwright && playwright install chromium\n"
            "  2) npm install playwright-core（让 node_modules 里能找到）\n"
            "  3) 直接用 --video 传已拿到的 mp4 直链（不需要浏览器）"
        )
    code = f"const {{chromium}} = require({json.dumps(str(nm))});\n" + script
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(code)
        tmp = fh.name
    try:
        proc = subprocess.run([node, tmp], capture_output=True, text=True, timeout=timeout)
        if proc.returncode != 0:
            tail = (proc.stderr or "").strip().splitlines()[-4:]
            raise RuntimeError("浏览器抓取失败：\n" + "\n".join(tail))
        return proc.stdout
    finally:
        Path(tmp).unlink(missing_ok=True)


# --------------------------------------------------------------------------
# Node 端脚本模板（用 __XXX__ 占位符，避免与 JS 大括号冲突）
# --------------------------------------------------------------------------

_NODE_LIST_JS = """
(async () => {
  const b = await chromium.launch({ headless: true });
  const ctx = await b.newContext({ storageState: __STATE__ });
  const p = await ctx.newPage();
  await p.goto(__PAGEURL__, { waitUntil: 'networkidle', timeout: 90000 });
  await p.waitForTimeout(4000);
  const data = await p.evaluate(async (endpoint) => {
    const r = await fetch(endpoint, { credentials: 'include' });
    if (!r.ok) return { error: 'HTTP ' + r.status };
    return await r.json();
  }, __ENDPOINT__);
  await b.close();
  console.log(JSON.stringify(data));
})();
"""

_NODE_HARVEST_JS = """
(async () => {
  const titles = __TITLES__;
  const b = await chromium.launch({ headless: true });
  const ctx = await b.newContext({ storageState: __STATE__, viewport: {width:1500,height:950} });
  const out = [];
  for (const title of titles) {
    let cap = null;
    const p = await ctx.newPage();
    p.on('response', async r => {
      if (!cap && r.url().includes(__KEYWORD__)) {
        try { cap = await r.json(); } catch (e) {}
      }
    });
    await p.goto(__PAGEURL__, { waitUntil:'networkidle', timeout:90000 }).catch(()=>{});
    await p.waitForTimeout(4000);
    await p.evaluate((t) => {
      const nodes = [...document.querySelectorAll('a,div,span,td')];
      const hit = nodes.find(n => n.textContent && n.textContent.trim() === t)
               || nodes.find(n => n.textContent && n.textContent.includes(t));
      if (hit) { hit.scrollIntoView(); hit.click(); return true; }
      return false;
    }, title).catch(()=>{});
    for (let k=0;k<26 && !cap;k++) await p.waitForTimeout(1500);
    if (!cap) {
      await p.goto(__PAGEURL__, { waitUntil:'networkidle', timeout:90000 }).catch(()=>{});
      await p.waitForTimeout(4000);
      await p.evaluate((t) => {
        const nodes = [...document.querySelectorAll('a,div,span,td')];
        const hit = nodes.find(n => n.textContent && n.textContent.includes(t));
        if (hit) { hit.scrollIntoView(); hit.click(); return true; }
        return false;
      }, title).catch(()=>{});
      for (let k=0;k<20 && !cap;k++) await p.waitForTimeout(1500);
    }
    const d = cap && (cap.data || cap);
    out.push({ title, code: d ? d.code : null,
               videos: (d && (d.lesson_videos || d.videos)) || [] });
    await p.close();
  }
  await b.close();
  console.log(JSON.stringify(out));
})();
"""

_NODE_LOGIN_JS = """
(async () => {
  const b = await chromium.launch({ headless: false, args: ['--no-first-run'] });
  const ctx = await b.newContext({ viewport: {width:1280,height:860} });
  const p = await ctx.newPage();
  await p.goto(__URL__, { timeout: 90000 });
  console.log('请在浏览器里完成登录…');
  for (let i=0;i<180;i++) {
    await p.waitForTimeout(3000);
    const u = p.url();
    if (u.includes('/course/') || u.replace(/\\/$/,'') === __BASE__) {
      await p.waitForTimeout(3000);
      await ctx.storageState({ path: __STATE__ });
      console.log('LOGIN_OK');
      await b.close();
      process.exit(0);
    }
  }
  await b.close();
  process.exit(1);
})();
"""


# ==========================================================================
# 对外接口
# ==========================================================================


def login(site: str, course: Any = None) -> Path:
    """弹一个有头浏览器，让用户登录一次，保存会话。"""
    ad = get_adapter(site)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    url = ad.lesson_url(course) if course else ad.base_url
    be = backend()

    if be == "python":
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=False, args=["--no-first-run"])
            ctx = browser.new_context(viewport={"width": 1280, "height": 860})
            page = ctx.new_page()
            page.goto(url, timeout=90000)
            print(f"已打开 {ad.base_url}，请在浏览器里完成登录…")
            for _ in range(180):
                page.wait_for_timeout(3000)
                try:
                    cur = page.url
                    if "/course/" in cur or cur.rstrip("/") == ad.base_url:
                        page.wait_for_timeout(3000)
                        ctx.storage_state(path=str(ad.state_path))
                        browser.close()
                        print(f"✔ 登录成功，会话已保存：{ad.state_path}")
                        return ad.state_path
                except Exception:
                    break
            browser.close()
        raise RuntimeError("登录超时。请重试，或直接用预签名 mp4 链接。")

    if be == "node":
        script = (
            _NODE_LOGIN_JS
            .replace("__URL__", json.dumps(url))
            .replace("__BASE__", json.dumps(ad.base_url))
            .replace("__STATE__", json.dumps(str(ad.state_path)))
        )
        out = _run_node(script, timeout=600)
        if "LOGIN_OK" in out:
            print(f"✔ 登录成功，会话已保存：{ad.state_path}")
            return ad.state_path
        raise RuntimeError("登录超时。请重试。")

    raise RuntimeError(
        "没有可用的浏览器后端。pip install playwright && playwright install chromium"
    )


def list_lessons(site: str, course: Any = None) -> List[Dict[str, Any]]:
    """列出课程里的所有课时。"""
    ad = get_adapter(site)
    if not ad.state_path.exists():
        raise RuntimeError(
            f"还没有 {ad.name} 的登录态。先运行："
            f"python scripts/course_kit.py lms login --site {ad.name} --course <id>"
        )
    if course is None:
        raise RuntimeError("需要 --course <课程 id>")

    endpoint = ad.list_endpoint.format(course=course)
    page_url = ad.lesson_url(course)
    be = backend()

    if be == "python":
        from playwright.sync_api import sync_playwright

        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(storage_state=str(ad.state_path))
            page = ctx.new_page()
            page.goto(page_url, wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(4000)
            data = page.evaluate(
                """async (endpoint) => {
                    const r = await fetch(endpoint, {credentials: 'include'});
                    if (!r.ok) return {error: 'HTTP ' + r.status};
                    return await r.json();
                }""",
                endpoint,
            )
            browser.close()
    else:
        script = (
            _NODE_LIST_JS
            .replace("__STATE__", json.dumps(str(ad.state_path)))
            .replace("__PAGEURL__", json.dumps(page_url))
            .replace("__ENDPOINT__", json.dumps(endpoint))
        )
        data = json.loads(_run_node(script, timeout=180).strip().splitlines()[-1])

    if isinstance(data, dict) and data.get("error"):
        raise RuntimeError(f"取课程列表失败：{data['error']}（检查课程 id 与登录态）")
    return ad.parse_list(data)


def harvest_urls(
    site: str,
    course: Any = None,
    *,
    only: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """抓每节课的录像直链。

    参数
    ----
    site   : 站点适配器名
    course : 课程 id
    only   : 只处理标题含这些关键字的课

    返回 [{activity_id, title, capture_code, streams, pc_url, room_url}]
    """
    ad = get_adapter(site)
    if not ad.state_path.exists():
        raise RuntimeError(
            f"还没有 {ad.name} 的登录态。先运行："
            f"python scripts/course_kit.py lms login --site {ad.name} --course <id>"
        )
    if course is None:
        raise RuntimeError("需要 --course <课程 id>")

    lessons = list_lessons(site, course)
    if only:
        lessons = [x for x in lessons if any(k in str(x.get("title") or "") for k in only)]
    if not lessons:
        raise RuntimeError("课程列表为空，或 --only 过滤后没有匹配的课。")

    titles = [str(x.get("title") or "") for x in lessons]
    page_url = ad.lesson_url(course)

    if backend() == "python":
        from playwright.sync_api import sync_playwright

        results: List[Dict[str, Any]] = []
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True)
            ctx = browser.new_context(
                storage_state=str(ad.state_path),
                viewport={"width": 1500, "height": 950},
            )
            for i, item in enumerate(lessons, 1):
                got = _grab_one_py(ctx, ad, titles[i - 1], page_url)
                results.append(_make_record(ad, item, got))
                _log_progress(i, len(lessons), results[-1])
            browser.close()
        return results

    script = (
        _NODE_HARVEST_JS
        .replace("__TITLES__", json.dumps(titles, ensure_ascii=False))
        .replace("__STATE__", json.dumps(str(ad.state_path)))
        .replace("__PAGEURL__", json.dumps(page_url))
        .replace("__KEYWORD__", json.dumps(ad.capture_url_keyword))
    )
    graw = json.loads(_run_node(script, timeout=600 + 90 * len(titles)).strip().splitlines()[-1])

    results = []
    for i, (item, got) in enumerate(zip(lessons, graw), 1):
        results.append(_make_record(ad, item, got))
        _log_progress(i, len(lessons), results[-1])
    return results


def _make_record(ad: SiteAdapter, item: Dict[str, Any], got: Dict[str, Any]) -> Dict[str, Any]:
    videos = got.get("videos") or []
    return {
        "activity_id": item.get("activity_id"),
        "title": item.get("title"),
        "capture_code": got.get("code"),
        "streams": len(videos),
        "pc_url": ad.pick_screen(videos),
        "room_url": ad.pick_camera(videos),
    }


def _log_progress(i: int, total: int, rec: Dict[str, Any]) -> None:
    mark = "✔" if rec["pc_url"] else "✘"
    print(f"[{i}/{total}] {str(rec['title'])[-24:]} 路={rec['streams']} 屏幕={mark}", flush=True)


def _grab_one_py(ctx, ad: SiteAdapter, title: str, page_url: str, *, wait_rounds: int = 16) -> Dict[str, Any]:
    """Python 后端：打开一节课，抓 capture 响应。"""
    page = ctx.new_page()
    captured: Dict[str, Any] = {}

    def on_response(resp):
        if ad.capture_url_keyword in resp.url and not captured:
            try:
                captured.update(resp.json())
            except Exception:
                pass

    page.on("response", on_response)
    try:
        page.goto(page_url, wait_until="networkidle", timeout=90000)
        page.wait_for_timeout(4000)
        page.evaluate(
            """(t) => {
                const nodes = [...document.querySelectorAll('a,div,span,td')];
                const hit = nodes.find(n => n.textContent && n.textContent.trim() === t)
                         || nodes.find(n => n.textContent && n.textContent.includes(t));
                if (hit) { hit.scrollIntoView(); hit.click(); return true; }
                return false;
            }""",
            title,
        )
        for _ in range(wait_rounds):
            if captured:
                break
            page.wait_for_timeout(1500)
        if not captured:
            page.goto(page_url, wait_until="networkidle", timeout=90000)
            page.wait_for_timeout(4000)
            page.evaluate(
                """(t) => {
                    const nodes = [...document.querySelectorAll('a,div,span,td')];
                    const hit = nodes.find(n => n.textContent && n.textContent.includes(t));
                    if (hit) { hit.scrollIntoView(); hit.click(); return true; }
                    return false;
                }""",
                title,
            )
            for _ in range(20):
                if captured:
                    break
                page.wait_for_timeout(1500)
    finally:
        page.close()

    data = captured.get("data") or captured or {}
    return {
        "code": data.get("code"),
        "videos": data.get("lesson_videos") or data.get("videos") or [],
    }


# ==========================================================================
# 命令行（自测 / 手动补抓）
# ==========================================================================


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(prog="lms", description="抓在线教学平台的课程录像直链")
    sub = p.add_subparsers(dest="cmd", required=True)

    for name, help_text in (
        ("login", "弹窗口登录一次"),
        ("list", "列出所有课时"),
        ("urls", "抓录像直链"),
    ):
        sp = sub.add_parser(name, help=help_text)
        sp.add_argument("--site", required=True, help="站点适配器名（在 ADAPTERS 中注册的名字）")
        sp.add_argument("--course", required=True, help="课程 id（URL 里 /course/<id>/ 那段）")
        if name == "urls":
            sp.add_argument("--only", nargs="*", help="只处理标题含这些关键字的课")
            sp.add_argument("-o", "--out", help="保存为 JSON")

    args = p.parse_args(argv)
    try:
        if args.cmd == "login":
            login(args.site, args.course)
        elif args.cmd == "list":
            for x in list_lessons(args.site, args.course):
                print(f"{x['activity_id']}\t{x['title']}")
        else:
            rows = harvest_urls(args.site, args.course, only=args.only)
            ok = sum(1 for r in rows if r.get("pc_url"))
            print(f"\n拿到屏幕画面：{ok}/{len(rows)}")
            if args.out:
                Path(args.out).write_text(
                    json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8"
                )
                print(f"已保存：{args.out}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"\n✘ {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
