"""Convert a markdown file to PDF using markdown + Playwright (headless Chromium).

Usage:
    python md_to_pdf.py <input.md> [<output.pdf>]
"""

import asyncio
import base64
import mimetypes
import re
import sys
from pathlib import Path

import markdown

CSS = """
@page {
    size: A4 landscape;
    margin: 16mm 14mm;
    @bottom-right { content: counter(page) " / " counter(pages); font-size: 9pt; color: #666; }
}
body {
    font-family: -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    font-size: 10pt;
    line-height: 1.45;
    color: #222;
    max-width: 100%;
}
h1 { font-size: 16pt; color: #1e40af; border-bottom: 1px solid #1e40af; padding-bottom: 2px; margin: 0 0 4px 0; }
h2 { font-size: 14pt; color: #1e40af; border-bottom: 1px solid #c4d4ed; padding-bottom: 2px; margin-top: 22px; page-break-after: avoid; }
h3 { font-size: 12pt; color: #3b82f6; margin-top: 16px; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 14px 0; font-size: 9pt; }
th { background: #1e40af; color: white; text-align: left; padding: 6px 8px; font-weight: 600; }
td { padding: 5px 8px; border-bottom: 1px solid #e2e8f0; vertical-align: top; }
tr:nth-child(even) td { background: #f8fafc; }

/* Section headers (full-width merged rows in the two big tables) */
table.bigtable tr.section td {
  background: #1e40af;
  color: #ffffff;
  font-size: 11pt;
  font-weight: 700;
  text-align: left;
  padding: 7px 10px;
  border-bottom: 1px solid #1e3a8a;
  text-transform: uppercase;
  letter-spacing: 0.3px;
}
table.bigtable tr.section .hint {
  font-size: 9pt;
  font-weight: 400;
  color: #c4d4ed;
  text-transform: none;
  letter-spacing: 0;
  margin-left: 8px;
}
/* Sub-section headers (nested within a section) */
table.bigtable tr.subsection td {
  background: #dbeafe;
  color: #1e3a8a;
  font-size: 10pt;
  font-weight: 700;
  text-align: left;
  padding: 5px 10px;
  border-bottom: 1px solid #93c5fd;
}
/* Disable the zebra striping for section/subsection rows */
table.bigtable tr.section td, table.bigtable tr.subsection td { background: inherit !important; }
table.bigtable tr.section td { background: #1e40af !important; }
table.bigtable tr.subsection td { background: #dbeafe !important; }

/* Embedded diagram image at the top — sized to fit on page 1 alongside title */
img {
  max-width: 100%;
  max-height: 155mm;       /* fits page 1 alongside the H1 title (A4 landscape ~178mm content area) */
  height: auto;
  display: block;
  margin: 2px auto 0 auto;
  border: 1px solid #cbd5e1;
  page-break-after: always; /* force next content (Executables) onto page 2 */
  page-break-before: avoid; /* keep with the title above */
}

/* Page-break utility for marked elements */
.pagebreak {
  page-break-before: always;
  break-before: page;
}
tr.pagebreak td, tr.pagebreak {
  page-break-before: always;
  break-before: page;
}

/* Side-by-side columns (used in the Quick start guide) */
.two-col {
  display: flex;
  gap: 18px;
  margin-top: 4px;
  page-break-inside: avoid;   /* keep the two-col block on one page */
  break-inside: avoid;
}
.two-col > div {
  flex: 1;
  min-width: 0;
}
.two-col h3 { margin-top: 0; }
.two-col pre { font-size: 8pt; padding: 5px 8px; margin: 4px 0; }
.two-col p { font-size: 9.5pt; margin: 3px 0; }

/* Compact styling for the Quick start guide (everything following h2.pagebreak) */
h2.pagebreak { font-size: 13pt; margin: 0 0 4px 0; padding-bottom: 1px; }
h2.pagebreak ~ h3 { font-size: 11pt; margin: 6px 0 3px 0; }
h2.pagebreak ~ p { font-size: 9.5pt; margin: 3px 0; }
h2.pagebreak ~ pre { font-size: 8pt; padding: 5px 8px; margin: 4px 0; }
code { font-family: "Consolas", "Courier New", monospace; background: #f1f5f9; color: #1e293b;
       padding: 1px 4px; border-radius: 3px; font-size: 8.5pt; }
pre { background: #1e293b; color: #e2e8f0; padding: 8px 12px; border-radius: 4px;
      font-family: "Consolas", monospace; font-size: 8.5pt; overflow-x: auto; }
pre code { background: transparent; color: inherit; padding: 0; }
hr { border: 0; border-top: 1px solid #d1d5db; margin: 20px 0; }
p { margin: 6px 0; }
ul, ol { margin: 6px 0 6px 22px; padding: 0; }
li { margin: 2px 0; }
"""


async def render_pdf(html_str: str, out_pdf: Path) -> None:
    from playwright.async_api import async_playwright
    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html_str, wait_until="load")
        await page.pdf(
            path=str(out_pdf),
            format="A4",
            landscape=True,
            margin={"top": "16mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
            print_background=True,
            display_header_footer=False,
        )
        await browser.close()


def inline_images(html: str, base_dir: Path) -> str:
    """Replace each <img ... src="..."> with a base64 data URI so Playwright's
    set_content() can show images without a base URL. Handles attributes
    appearing in any order before `src`."""
    def repl(m):
        full = m.group(0)
        src = m.group(1)
        if src.startswith(("data:", "http://", "https://", "file://")):
            return full
        path = (base_dir / src).resolve()
        if not path.exists():
            print(f"  WARN: image not found at {path}", file=sys.stderr)
            return full
        mime, _ = mimetypes.guess_type(str(path))
        mime = mime or "image/png"
        data = base64.b64encode(path.read_bytes()).decode("ascii")
        return full.replace(f'src="{src}"', f'src="data:{mime};base64,{data}"')
    return re.sub(r'<img\b[^>]*?\bsrc="([^"]+)"[^>]*>', repl, html)


def main():
    if len(sys.argv) < 2:
        print(__doc__); sys.exit(1)
    in_md = Path(sys.argv[1])
    out_pdf = Path(sys.argv[2]) if len(sys.argv) >= 3 else in_md.with_suffix(".pdf")
    md_text = in_md.read_text(encoding="utf-8")
    html_body = markdown.markdown(
        md_text,
        extensions=["tables", "fenced_code", "sane_lists", "attr_list", "md_in_html"],
        output_format="html5",
    )
    # Inline any local images as data URIs so Playwright can render them.
    html_body = inline_images(html_body, in_md.parent)
    html_full = f"""<!doctype html><html><head><meta charset="utf-8">
<style>{CSS}</style></head><body>{html_body}</body></html>"""
    asyncio.run(render_pdf(html_full, out_pdf))
    print(f"Wrote {out_pdf}  ({out_pdf.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
