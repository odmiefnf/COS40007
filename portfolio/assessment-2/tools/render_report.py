"""Render the complete, executed Portfolio 2 notebook as a readable PDF.

This exporter never executes notebook code. Markdown cells, source code, and all
stored text/PNG outputs are emitted in notebook order. Run from any directory:

    python tools/render_report.py --render

``--render`` additionally rasterises every PDF page into output/pdf/qa/ for
visual inspection. reportlab, nbformat, Pillow and Poppler must be available.
"""

from __future__ import annotations

import argparse
import base64
import io
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from xml.sax.saxutils import escape

ROOT = Path(__file__).resolve().parents[1]
# The bundled notebook dependency is also used by Portfolio 1. No installation
# or network access is performed by this exporter.
for vendor in (ROOT.parent / "portfolio1" / "vendor", ROOT / "vendor"):
    if vendor.is_dir():
        sys.path.insert(0, str(vendor))

import nbformat
from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Flowable,
    HRFlowable,
    Image,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)


DEFAULT_STEM = "Portfolio_Assessment_2_COS40007_NguyenVanTuanKiet"
NAVY = colors.HexColor("#17324D")
TEAL = colors.HexColor("#087F8C")
INK = colors.HexColor("#243447")
GRAY = colors.HexColor("#637381")
LIGHT = colors.HexColor("#E2E9EF")
PAGE_WIDTH, PAGE_HEIGHT = A4
MARGIN_X = 43
TOP_MARGIN = 49
BOTTOM_MARGIN = 46
# SimpleDocTemplate's frame has a default 6 pt padding on each edge.
CONTENT_WIDTH = PAGE_WIDTH - 2 * MARGIN_X - 12
CONTENT_HEIGHT = PAGE_HEIGHT - TOP_MARGIN - BOTTOM_MARGIN - 12
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def clean_text(value: object) -> str:
    """Remove terminal control codes while preserving readable source text."""
    if isinstance(value, list):
        value = "".join(str(part) for part in value)
    text = ANSI.sub("", str(value)).replace("\r\n", "\n").replace("\r", "")
    text = text.replace("\u00a0", " ").replace("\u202f", " ")
    text = text.replace("\u2010", "-").replace("\u2011", "-")
    text = text.replace("\u2012", "-").replace("\u2013", "-")
    text = text.replace("\u2014", " - ").replace("\u2212", "-")
    return "".join(ch for ch in text if ch in "\n\t" or ord(ch) >= 32)


def register_fonts() -> None:
    """Embed common Windows fonts, falling back to PDF standard fonts."""
    font_dir = Path(os.environ.get("WINDIR", "C:/Windows")) / "Fonts"
    specs = {
        "Body": ("arial.ttf", "Helvetica"),
        "BodyBold": ("arialbd.ttf", "Helvetica-Bold"),
        "BodyItalic": ("ariali.ttf", "Helvetica-Oblique"),
        "BodyBoldItalic": ("arialbi.ttf", "Helvetica-BoldOblique"),
        "Mono": ("consola.ttf", "Courier"),
        "MonoBold": ("consolab.ttf", "Courier-Bold"),
    }
    for name, (filename, fallback) in specs.items():
        path = font_dir / filename
        if path.is_file():
            pdfmetrics.registerFont(TTFont(name, str(path)))
        else:
            pdfmetrics.registerFont(pdfmetrics.Font(name, fallback, "WinAnsiEncoding"))
    pdfmetrics.registerFontFamily(
        "Body", normal="Body", bold="BodyBold", italic="BodyItalic",
        boldItalic="BodyBoldItalic",
    )
    pdfmetrics.registerFontFamily(
        "Mono", normal="Mono", bold="MonoBold", italic="Mono", boldItalic="MonoBold",
    )


def make_styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["BodyText"]
    body = ParagraphStyle(
        "ReportBody", parent=base, fontName="Body", fontSize=9.3,
        leading=13.5, textColor=INK, spaceAfter=7,
        allowWidows=0, allowOrphans=0, splitLongWords=True,
    )
    return {
        "body": body,
        "title": ParagraphStyle(
            "ReportTitle", parent=body, fontName="BodyBold", fontSize=24,
            leading=29, textColor=NAVY, spaceBefore=7, spaceAfter=16,
            keepWithNext=True,
        ),
        "h1": ParagraphStyle(
            "ReportH1", parent=body, fontName="BodyBold", fontSize=17,
            leading=21, textColor=NAVY, spaceBefore=15, spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "ReportH2", parent=body, fontName="BodyBold", fontSize=13.2,
            leading=17, textColor=NAVY, spaceBefore=12, spaceAfter=7,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "ReportH3", parent=body, fontName="BodyBold", fontSize=10.8,
            leading=14.5, textColor=TEAL, spaceBefore=10, spaceAfter=6,
            keepWithNext=True,
        ),
        "bullet": ParagraphStyle(
            "ReportBullet", parent=body, leftIndent=12, firstLineIndent=0,
            bulletIndent=0, spaceAfter=4,
        ),
        "quote": ParagraphStyle(
            "ReportQuote", parent=body, leftIndent=12, rightIndent=6,
            borderColor=LIGHT, borderWidth=0.6, borderPadding=7,
            textColor=GRAY,
        ),
        "table": ParagraphStyle(
            "ReportTable", parent=body, fontSize=8, leading=11,
            spaceAfter=0,
        ),
        "table_head": ParagraphStyle(
            "ReportTableHead", parent=body, fontName="BodyBold", fontSize=8,
            leading=11, textColor=colors.white, spaceAfter=0,
        ),
        "caption": ParagraphStyle(
            "ReportCaption", parent=body, fontSize=8, leading=11,
            textColor=GRAY, spaceBefore=3, spaceAfter=8,
        ),
    }


def inline(text: str) -> str:
    """Convert a small, safe Markdown subset to ReportLab paragraph markup."""
    text = clean_text(text)
    fragments: list[str] = []

    def hold(markup: str) -> str:
        fragments.append(markup)
        return f"\x00{len(fragments) - 1}\x00"

    # Stash code before interpreting emphasis. All literal input is XML escaped.
    text = re.sub(
        r"`([^`\n]+)`",
        lambda match: hold('<font name="Mono" size="8.2">' + escape(match[1]) + "</font>"),
        text,
    )
    text = escape(text)
    text = text.replace("\u2028", "<br/>")
    text = re.sub(
        r"\[([^\]]+)\]\((https?://[^\s)]+)\)",
        lambda match: '<link href="' + escape(match[2], {'"': '&quot;'})
        + '" color="#087F8C">' + match[1] + "</link>",
        text,
    )
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"__([^_]+)__", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"\x00(\d+)\x00", lambda m: fragments[int(m[1])], text)
    return text


def fit_prefix(text: str, max_width: float, font: str, size: float) -> int:
    lo, hi = 1, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if pdfmetrics.stringWidth(text[:mid], font, size) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return lo


def wrapped_lines(text: str, width: float, font: str, size: float) -> list[str]:
    """Character-preserving wrap for code and monospaced output."""
    lines: list[str] = []
    for original in clean_text(text).expandtabs(4).split("\n"):
        rest = original
        if not rest:
            lines.append("")
            continue
        while pdfmetrics.stringWidth(rest, font, size) > width:
            take = fit_prefix(rest, width, font, size)
            # Prefer a nearby whitespace boundary without discarding characters.
            boundary = max(rest.rfind(" ", max(1, take - 24), take),
                           rest.rfind(",", max(1, take - 20), take))
            if boundary >= max(1, take - 24):
                take = boundary + 1
            lines.append(rest[:take])
            rest = rest[take:]
        lines.append(rest)
    return lines or [""]


class MonoBlock(Flowable):
    """A monospaced panel that can split at every source/output line."""

    def __init__(self, text: str | list[str], label: str, *, output: bool = False,
                 prewrapped: bool = False):
        super().__init__()
        self.raw = text
        self.label = label
        self.output = output
        self.prewrapped = prewrapped
        self.font_size = 7.25 if not output else 7.1
        self.leading = 9.4 if not output else 9.2
        self.padding = 8
        self.label_height = 13
        self.spaceBefore = 4
        self.spaceAfter = 9
        self._lines: list[str] = []

    def wrap(self, availWidth: float, availHeight: float) -> tuple[float, float]:
        self.width = availWidth
        self._lines = list(self.raw) if self.prewrapped else wrapped_lines(
            str(self.raw), availWidth - 2 * self.padding, "Mono", self.font_size,
        )
        self.height = 2 * self.padding + self.label_height + len(self._lines) * self.leading
        return self.width, self.height

    def split(self, availWidth: float, availHeight: float) -> list[Flowable]:
        self.wrap(availWidth, availHeight)
        count = math.floor((availHeight - 2 * self.padding - self.label_height) / self.leading)
        if count < 2 or count >= len(self._lines):
            return []
        first = MonoBlock(self._lines[:count], self.label, output=self.output, prewrapped=True)
        title = self.label if self.label.endswith(" (continued)") else self.label + " (continued)"
        rest = MonoBlock(self._lines[count:], title, output=self.output, prewrapped=True)
        first.spaceAfter = 3
        return [first, rest]

    def draw(self) -> None:
        c = self.canv
        c.saveState()
        c.setFillColor(colors.HexColor("#F4F6F8") if not self.output else colors.HexColor("#FAFBFC"))
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        c.setStrokeColor(TEAL if not self.output else LIGHT)
        c.setLineWidth(1.8 if not self.output else 0.6)
        c.line(0, 0, 0, self.height)
        c.setFillColor(TEAL if not self.output else GRAY)
        c.setFont("BodyBold", 7.4)
        c.drawString(self.padding, self.height - self.padding - 6, self.label)
        c.setFillColor(INK)
        c.setFont("Mono", self.font_size)
        y = self.height - self.padding - self.label_height - self.font_size
        for line in self._lines:
            c.drawString(self.padding, y, line)
            y -= self.leading
        c.restoreState()


def pipe_row(text: str) -> list[str]:
    return [part.strip() for part in text.strip().strip("|").split("|")]


def is_table_separator(text: str) -> bool:
    row = pipe_row(text)
    return bool(row) and all(re.fullmatch(r":?-{3,}:?", col.replace(" ", "")) for col in row)


def markdown_table(lines: list[str], styles: dict) -> Table:
    rows = [pipe_row(line) for line in lines]
    headers = rows[0]
    rows = [headers] + rows[2:]
    count = max(len(row) for row in rows)
    for row in rows:
        row.extend([""] * (count - len(row)))
    # Allocate by typical text length, bounded so no column becomes unusable.
    weights = []
    for index in range(count):
        lengths = sorted(len(re.sub(r"[*`_]", "", row[index])) for row in rows)
        weights.append(max(10, min(40, lengths[len(lengths) // 2])))
    total = sum(weights)
    widths = [CONTENT_WIDTH * weight / total for weight in weights]
    data = [
        [Paragraph(inline(value) or " ", styles["table_head" if r == 0 else "table"])
         for value in row]
        for r, row in enumerate(rows)
    ]
    table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F2F6F8")]),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LINEBELOW", (0, 0), (-1, 0), 0.5, NAVY),
        ("LINEBELOW", (0, -1), (-1, -1), 0.5, LIGHT),
    ]))
    table.spaceBefore = 4
    table.spaceAfter = 10
    return table


def markdown_flowables(text: str, styles: dict, *, cover: bool = False) -> list[Flowable]:
    text = clean_text(text)
    lines = text.splitlines()
    story: list[Flowable] = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        if line.startswith("```") or line.startswith("~~~"):
            marker = line[:3]
            language = line[3:].strip()
            block = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith(marker):
                block.append(lines[index])
                index += 1
            story.append(MonoBlock("\n".join(block), "Code" + (f" - {language}" if language else "")))
            index += 1
            continue
        heading = re.match(r"^(#{1,6})\s+(.+?)\s*#*\s*$", line)
        if heading:
            level, content = len(heading[1]), heading[2]
            key = "title" if cover and level == 1 else ("h1" if level == 1 else "h2" if level == 2 else "h3")
            story.append(Paragraph(inline(content), styles[key]))
            index += 1
            continue
        if re.fullmatch(r"[-*_]{3,}", line):
            story.append(HRFlowable(width="100%", thickness=0.6, color=LIGHT,
                                    spaceBefore=5, spaceAfter=9))
            index += 1
            continue
        if index + 1 < len(lines) and "|" in line and is_table_separator(lines[index + 1].strip()):
            block = [lines[index], lines[index + 1]]
            index += 2
            while index < len(lines) and "|" in lines[index] and lines[index].strip():
                block.append(lines[index])
                index += 1
            story.append(markdown_table(block, styles))
            continue
        bullet = re.match(r"^([-+*]|\d+[.)])\s+(.+)$", line)
        if bullet:
            marker = "-" if bullet[1] in "-+*" else bullet[1]
            parts = [bullet[2]]
            index += 1
            # Markdown permits unindented wrapped lines within a list item.
            # Keep the question and its answer in one paragraph so they cannot
            # leave an orphaned question line at the bottom of a page.
            while index < len(lines) and lines[index].strip():
                candidate = lines[index].strip()
                if re.match(r"^(#{1,6}\s|[-+*]\s|\d+[.)]\s|```|~~~)", candidate):
                    break
                parts.append(candidate)
                index += 1
            story.append(Paragraph(inline(" ".join(parts)), styles["bullet"], bulletText=marker))
            continue
        quote = line.startswith(">")
        first_part = line.lstrip("> ") if quote else line
        parts = [first_part + ("\u2028" if lines[index].endswith("  ") else "")]
        index += 1
        while index < len(lines) and lines[index].strip():
            candidate = lines[index].strip()
            if re.match(r"^(#{1,6}\s|[-+*]\s|\d+[.)]\s|```|~~~)", candidate):
                break
            if index + 1 < len(lines) and "|" in candidate and is_table_separator(lines[index + 1].strip()):
                break
            part = candidate.lstrip("> ") if quote else candidate
            parts.append(part + ("\u2028" if lines[index].endswith("  ") else ""))
            index += 1
        story.append(Paragraph(inline(" ".join(parts)), styles["quote" if quote else "body"]))
    return story


def image_flowable(payload: object, caption: str, styles: dict) -> list[Flowable]:
    raw = base64.b64decode("".join(payload) if isinstance(payload, list) else str(payload))
    buffer = io.BytesIO(raw)
    with PILImage.open(buffer) as picture:
        width, height = picture.size
    buffer.seek(0)
    scale = min(CONTENT_WIDTH / width, (CONTENT_HEIGHT - 45) / height)
    image = Image(buffer, width=width * scale, height=height * scale)
    image.hAlign = "CENTER"
    return [KeepTogether([image, Paragraph(inline(caption), styles["caption"])])]


def page_decoration(canvas, doc) -> None:
    canvas.saveState()
    if doc.page > 1:
        canvas.setStrokeColor(TEAL)
        canvas.setLineWidth(1)
        canvas.line(MARGIN_X, PAGE_HEIGHT - 29, PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 29)
        canvas.setFillColor(NAVY)
        canvas.setFont("BodyBold", 7.5)
        canvas.drawString(MARGIN_X, PAGE_HEIGHT - 21, "COS40007 | Portfolio Assessment 2")
        canvas.setFillColor(GRAY)
        canvas.setFont("Body", 7)
        canvas.drawRightString(PAGE_WIDTH - MARGIN_X, PAGE_HEIGHT - 21, "Nguyen Van Tuan Kiet")
    canvas.setStrokeColor(LIGHT)
    canvas.setLineWidth(0.5)
    canvas.line(MARGIN_X, 34, PAGE_WIDTH - MARGIN_X, 34)
    canvas.setFillColor(GRAY)
    canvas.setFont("Body", 6.7)
    canvas.drawString(MARGIN_X, 23, "Source: executed notebook | Includes code and stored outputs")
    canvas.drawRightString(PAGE_WIDTH - MARGIN_X, 23, f"Page {doc.page}")
    canvas.restoreState()


def render_notebook(notebook_path: Path, pdf_path: Path) -> dict:
    register_fonts()
    styles = make_styles()
    notebook = nbformat.read(notebook_path, as_version=4)
    story: list[Flowable] = []
    counters = {"markdown_cells": 0, "code_cells": 0, "text_outputs": 0,
                "markdown_outputs": 0, "images": 0}
    for index, cell in enumerate(notebook.cells):
        kind = cell.cell_type
        if kind in ("markdown", "raw"):
            cover = index == 0
            story.extend(markdown_flowables(cell.source, styles, cover=cover))
            counters["markdown_cells"] += 1
            if cover:
                story.append(PageBreak())
            continue
        if kind != "code":
            continue
        counters["code_cells"] += 1
        number = counters["code_cells"]
        execution = cell.get("execution_count")
        label = f"Code cell {number:02d} | Execution {execution if execution is not None else 'not recorded'}"
        story.append(MonoBlock(cell.source, label))
        for out_number, output in enumerate(cell.get("outputs", []), 1):
            out_type = output.get("output_type", "")
            out_label = f"Output {number:02d}.{out_number}"
            if out_type == "stream":
                story.append(MonoBlock(clean_text(output.get("text", "")), out_label + " | " + output.get("name", "stream"), output=True))
                counters["text_outputs"] += 1
            elif out_type == "error":
                text = "\n".join(output.get("traceback", [])) or f"{output.get('ename', 'Error')}: {output.get('evalue', '')}"
                story.append(MonoBlock(text, out_label + " | Error", output=True))
                counters["text_outputs"] += 1
            else:
                data = output.get("data", {})
                # These MIME representations describe the same output. Prefer
                # the rich representation and avoid duplicating its plain text.
                if "image/png" in data:
                    story.extend(image_flowable(data["image/png"], out_label + " | Stored notebook figure", styles))
                    counters["images"] += 1
                elif "text/markdown" in data:
                    story.extend(markdown_flowables(data["text/markdown"], styles))
                    counters["markdown_outputs"] += 1
                elif "text/plain" in data:
                    story.append(MonoBlock(clean_text(data["text/plain"]), out_label + " | Result", output=True))
                    counters["text_outputs"] += 1
                elif data:
                    # Keep unsupported MIME data visible rather than silently
                    # dropping a potentially important notebook result.
                    values = "\n\n".join(f"{mime}\n{clean_text(value)}" for mime, value in data.items())
                    story.append(MonoBlock(values, out_label + " | Stored data", output=True))
                    counters["text_outputs"] += 1
    # A trailing page break creates an empty final page for cover-only notebooks.
    while story and isinstance(story[-1], PageBreak):
        story.pop()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(pdf_path), pagesize=A4, rightMargin=MARGIN_X, leftMargin=MARGIN_X,
        topMargin=TOP_MARGIN, bottomMargin=BOTTOM_MARGIN,
        title="Portfolio Assessment 2: Advanced Data Preprocessing, Feature Engineering & Model Development",
        author="Nguyen Van Tuan Kiet", subject="COS40007 Artificial Intelligence Engineering",
        creator="Executed Jupyter notebook exported with ReportLab",
        pageCompression=1,
    )
    doc.build(story, onFirstPage=page_decoration, onLaterPages=page_decoration)
    return counters


def find_pdftoppm() -> str:
    program = shutil.which("pdftoppm")
    if program:
        return program
    # Runtime path returned by Codex's workspace dependency inventory.
    base = Path.home() / ".cache/codex-runtimes/codex-primary-runtime/dependencies"
    candidates = (
        base / "native/poppler/Library/bin/pdftoppm.exe",
        base / "native/poppler/bin/pdftoppm",
        base / "native/poppler/bin/pdftoppm.exe",
    )
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    raise FileNotFoundError("pdftoppm was not found on PATH or in the bundled Poppler runtime")


def rasterise(pdf_path: Path, dpi: int = 105) -> Path:
    qa_dir = pdf_path.parent / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    prefix = qa_dir / "page"
    # Only remove earlier rasterisations generated by this command, so a new
    # PDF with fewer pages cannot leave misleading old page images behind.
    for old in qa_dir.glob("page-*.png"):
        old.unlink()
    subprocess.run(
        [find_pdftoppm(), "-r", str(dpi), "-png", str(pdf_path), str(prefix)],
        check=True,
    )
    return qa_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--notebook", type=Path, default=ROOT / (DEFAULT_STEM + ".ipynb"))
    parser.add_argument("--output", type=Path, default=ROOT / "output/pdf" / (DEFAULT_STEM + ".pdf"))
    parser.add_argument("--render", action="store_true", help="render every PDF page to output/pdf/qa/*.png")
    parser.add_argument("--dpi", type=int, default=105)
    args = parser.parse_args()
    notebook_path = args.notebook.resolve()
    pdf_path = args.output.resolve()
    if not notebook_path.is_file():
        parser.error(f"notebook does not exist: {notebook_path}")
    counters = render_notebook(notebook_path, pdf_path)
    print(f"PDF: {pdf_path}")
    print("Included: " + ", ".join(f"{key}={value}" for key, value in counters.items()))
    if args.render:
        qa_dir = rasterise(pdf_path, args.dpi)
        print(f"QA images: {qa_dir} ({len(list(qa_dir.glob('page-*.png')))} pages)")


if __name__ == "__main__":
    main()
