"""PDF model: saves a lecture's report as a PDF file (report.pdf in its lesson folder).

The report page is rendered once more as a self-contained print version (style sheet, script, logo and chart
images embedded, every tab one after another) and printed to PDF by a browser already on the computer (Microsoft
Edge or Google Chrome, run hidden). Nothing is downloaded and nothing leaves the computer; the charts in it are the
same blurred pictures the page shows. The PDF is kept and only made again when the report has changed.
"""
import base64
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from web.models import print_charts, stat_charts
from web.models.charts import chart_path

STATIC = Path(__file__).resolve().parents[1] / "views" / "static"
PRINT_SECONDS = 90
REPLACE_ATTEMPTS = 5
LAYOUT_FILES = [STATIC.parent / "templates" / "report.html", STATIC.parent / "templates" / "base.html",
                STATIC / "style.css", STATIC / "report.js", Path(__file__).parent / "print_charts.py", Path(__file__)]
JOIN_SECONDS = 10         # in the PDF, moments of one kind and seat closer than this are one row
BROWSER_PATHS = [   # Windows first (this project's laptops), then the usual names on other systems
    Path(os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Microsoft/Edge/Application/msedge.exe",
    Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "Google/Chrome/Application/chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google/Chrome/Application/chrome.exe",
]
BROWSER_NAMES = ["msedge", "microsoft-edge", "google-chrome", "chrome", "chromium", "chromium-browser"]


class PdfUnavailable(Exception):
    """The PDF could not be made (no browser found, or printing failed); the message says why."""


def find_browser(configured=None):
    """The browser used for printing: TEACHLENS_PDF_BROWSER if set, else Edge or Chrome where they usually are."""
    if configured:
        return Path(configured) if Path(configured).exists() else None
    for path in BROWSER_PATHS:
        if path.exists():
            return path
    for name in BROWSER_NAMES:
        found = shutil.which(name)
        if found:
            return Path(found)
    return None


def data_uri(path, mime):
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode("ascii")


def moment_summary(moments):
    """One row per kind of moment: how many, and on which seats (most first), for the PDF's short Moments section."""
    rows = {}
    for m in moments:
        row = rows.setdefault(m["behaviour"], {"behaviour": m["behaviour"], "count": 0, "seats": {}})
        row["count"] += 1
        row["seats"][m["seat"]] = row["seats"].get(m["seat"], 0) + 1
    for row in rows.values():
        row["seats"] = ", ".join(f"{seat} ({n})" for seat, n in sorted(row["seats"].items(), key=lambda s: -s[1]))
    return sorted(rows.values(), key=lambda r: -r["count"])


def bytes_uri(data, mime="image/png"):
    return f"data:{mime};base64," + base64.b64encode(data).decode("ascii") if data else None


def joined_moments(moments, gap=JOIN_SECONDS):
    """Every moment, with moments of the same kind on the same seat less than `gap` seconds apart joined into one row
    (start of the first to end of the last; "Phone seen (7 times)"), so the PDF can list them all in a few pages."""
    rows = []
    for m in sorted(moments, key=lambda m: m["seconds"]):
        end = m["seconds"] + m["duration"]
        last = next((r for r in reversed(rows) if r["kind"] == m["kind"] and r["seat"] == m["seat"]), None)
        if last and m["seconds"] - last["_end"] < gap:
            last["_end"], last["_count"] = max(last["_end"], end), last["_count"] + 1
        else:
            rows.append({**m, "_end": end, "_count": 1, "_label": m["behaviour"]})
    for row in rows:
        row["end"] = clock(row["_end"])
        row["duration"] = int(round(row["_end"] - row["seconds"]))
        row["behaviour"] = row["_label"] + (f" ({row['_count']} times)" if row["_count"] > 1 else "")
    return sorted(rows, key=lambda r: r["seconds"])


def clock(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def print_assets(lecture, report):
    """What report.html needs for the PDF: the CSS and JS text, the logo and each heatmap as data URIs, the two
    statistical charts as static figures for paper (web/models/print_charts.py), and the short Moments section (a
    summary, then every moment with close ones joined). `report` is build_report()'s result."""
    charts, moments = report["charts"], report["moments"]
    rows, _window = stat_charts.comparison_values(lecture)
    return {
        "print_charts": {"timeline": bytes_uri(print_charts.timeline_png((report.get("timeline") or {}).get("data"))),
                         "comparison": bytes_uri(print_charts.comparison_png(rows))},
        "pdf": True,
        "pdf_moment_rows": joined_moments(moments),
        "join_seconds": JOIN_SECONDS,
        "moment_summary": moment_summary(moments),
        "inline_css": (STATIC / "style.css").read_text(encoding="utf-8"),
        "inline_js": (STATIC / "report.js").read_text(encoding="utf-8"),
        "logo_src": data_uri(STATIC / "uaeu-logo.svg", "image/svg+xml"),
        "chart_src": {c["file"]: data_uri(chart_path(lecture, c["file"]), "image/png") for c in charts
                      if chart_path(lecture, c["file"]).exists()},
    }


def pdf_path(lecture):
    return lecture.lesson_dir / "report.pdf"


def is_current(lecture):
    """True when report.pdf exists and is newer than everything the report is made from."""
    path = pdf_path(lecture)
    if not path.exists():
        return False
    sources = [lecture.lesson_dir / name for name in ("final_feedback_report.txt", "facts.json", "report_status.json")]
    sources += [lecture.input_dir / "context.json", lecture.charts_dir / "style.txt"]
    sources += LAYOUT_FILES   # a new layout (templates, styles, figure code) makes a new PDF too
    newest = max((p.stat().st_mtime for p in sources if p.exists()), default=0)
    return path.stat().st_mtime >= newest


def print_to_pdf(html, out_path, browser, runner=subprocess.run):
    """Prints one self-contained HTML page to out_path with a hidden browser; raises PdfUnavailable on failure."""
    out_path = Path(out_path)
    with tempfile.TemporaryDirectory(prefix="teachlens_pdf_") as work:
        page = Path(work) / "report.html"
        page.write_text(html, encoding="utf-8")
        partial = out_path.with_suffix(".tmp.pdf")
        partial.unlink(missing_ok=True)
        command = [str(browser), "--headless=new", "--disable-gpu", "--no-first-run", "--disable-extensions",
                   f"--user-data-dir={Path(work) / 'profile'}", "--no-pdf-header-footer",
                   "--virtual-time-budget=10000",   # lets the page's script draw the charts before printing
                   f"--print-to-pdf={partial}", page.as_uri()]
        try:
            runner(command, capture_output=True, timeout=PRINT_SECONDS, check=False)
        except subprocess.TimeoutExpired as error:
            raise PdfUnavailable("The browser took too long to print the report.") from error
        except OSError as error:
            raise PdfUnavailable(f"The browser could not be started ({error}).") from error
        if not partial.exists() or partial.stat().st_size == 0 or not partial.read_bytes().startswith(b"%PDF"):
            partial.unlink(missing_ok=True)
            raise PdfUnavailable("The browser did not produce a PDF.")
        for attempt in range(REPLACE_ATTEMPTS):
            try:
                os.replace(partial, out_path)   # a reader never sees a half-written file
                break
            except PermissionError:             # Windows: the old PDF is still open (being downloaded or viewed)
                time.sleep(0.2 * (attempt + 1))
        else:
            partial.unlink(missing_ok=True)
            if not out_path.exists():
                raise PdfUnavailable("The PDF file could not be saved (another program has it open).")
    return out_path


def download_name(lecture, context):
    title = "".join(ch if ch.isalnum() or ch in " -_()" else " " for ch in (context.get("name") or lecture.id))
    return f"TeachLens report - {' '.join(title.split())[:80]}.pdf"
