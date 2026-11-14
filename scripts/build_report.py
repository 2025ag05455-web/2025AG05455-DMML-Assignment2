"""Render ``reports/final_report.md`` to ``reports/final_report.pdf`` with ReportLab.

Supports the Markdown subset used by the report: ``#``/``##``/``###`` headings, paragraphs,
``*``/``-`` bullet lists (indented continuation lines), pipe tables (``---:`` = right-aligned),
images ``![caption](path)`` and inline ``**bold**``, ``*italic*``, ```code``` and ``[text](url)``.
Layout hints are HTML comments, so they are invisible when the Markdown is viewed on GitHub:

* ``<!-- pagebreak -->``                start a new page
* ``<!-- landscape -->`` / ``<!-- portrait -->``  switch page orientation (starts a new page)
* ``<!-- img max_h=12cm -->``           maximum height of the next image

Usage (from the project root)::

    python scripts/build_report.py [--input reports/final_report.md] [--output reports/final_report.pdf]
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (BaseDocTemplate, Frame, Image, KeepTogether, NextPageTemplate,
                                PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle)
from reportlab.platypus.flowables import HRFlowable

ROOT = Path(__file__).resolve().parents[1]
MARGIN = 1.7 * cm
ACCENT = colors.HexColor("#1F3A68")
GRID = colors.HexColor("#C9CED6")
HEADER_BG = colors.HexColor("#E8ECF3")

_COMMENT = re.compile(r"^<!--\s*(.*?)\s*-->$")
_IMAGE = re.compile(r"^!\[(.*)\]\((.+?)\)\s*$")
_BULLET = re.compile(r"^[*-]\s+(.*)$")
_HEADING = re.compile(r"^(#{1,3})\s+(.*)$")
_LENGTH = re.compile(r"^([\d.]+)\s*(cm|mm|pt)?$")


# --------------------------------------------------------------------------- styles
def _styles() -> dict[str, ParagraphStyle]:
    base = getSampleStyleSheet()["Normal"]
    body = ParagraphStyle("body", parent=base, fontName="Helvetica", fontSize=9.3, leading=12.2,
                          spaceAfter=4.5)
    return {
        "body": body,
        "h1": ParagraphStyle("h1", parent=body, fontName="Helvetica-Bold", fontSize=15.5, leading=19,
                             textColor=ACCENT, spaceAfter=3),
        "h2": ParagraphStyle("h2", parent=body, fontName="Helvetica-Bold", fontSize=11.3, leading=14,
                             textColor=ACCENT, spaceBefore=7, spaceAfter=3),
        "h3": ParagraphStyle("h3", parent=body, fontName="Helvetica-Bold", fontSize=10, leading=13,
                             spaceBefore=5, spaceAfter=2),
        "bullet": ParagraphStyle("bullet", parent=body, leftIndent=11, bulletIndent=2, spaceAfter=2.5),
        "cell": ParagraphStyle("cell", parent=body, fontSize=8.3, leading=10.2, spaceAfter=0),
        "cell_head": ParagraphStyle("cell_head", parent=body, fontName="Helvetica-Bold", fontSize=8.3,
                                    leading=10.2, spaceAfter=0),
        "caption": ParagraphStyle("caption", parent=body, fontName="Helvetica-Oblique", fontSize=8.3,
                                  leading=10.5, alignment=TA_CENTER, textColor=colors.HexColor("#444444"),
                                  spaceBefore=2, spaceAfter=9),
    }


# --------------------------------------------------------------------------- inline markup
def inline(text: str) -> str:
    """Convert inline Markdown to ReportLab paragraph markup (code spans are protected)."""
    codes: list[str] = []

    def _stash(match: re.Match) -> str:
        codes.append(match.group(1))
        return f"\x00{len(codes) - 1}\x00"

    text = re.sub(r"`([^`]+)`", _stash, text)
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1F4E9A">\1</link>', text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"<i>\1</i>", text)

    def _unstash(match: re.Match) -> str:
        code = codes[int(match.group(1))].replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
        return f'<font face="Courier" size="8.3">{code}</font>'

    return re.sub(r"\x00(\d+)\x00", _unstash, text)


def _length(value: str) -> float:
    match = _LENGTH.match(value.strip())
    if not match:
        raise ValueError(f"bad length {value!r} (expected e.g. 12cm)")
    number, unit = float(match.group(1)), match.group(2) or "pt"
    return number * {"cm": cm, "mm": cm / 10, "pt": 1.0}[unit]


# --------------------------------------------------------------------------- block builders
def _table(rows: list[str], styles: dict[str, ParagraphStyle], width: float) -> Table:
    cells = [[c.strip() for c in row.strip().strip("|").split("|")] for row in rows]
    header, sep, body = cells[0], cells[1], cells[2:]
    align = ["RIGHT" if s.strip().endswith(":") else "LEFT" for s in sep]

    # Column widths proportional to the longest markup-free text (monospace code is wider),
    # capped so one long column cannot squeeze the others, plus a constant for cell padding.
    def _visible_len(cell: str) -> float:
        return len(re.sub(r"[*`]", "", cell)) * (1.25 if "`" in cell else 1.0)

    longest = [max(_visible_len(row[i]) for row in [header, *body]) for i in range(len(header))]
    weights = [min(n, 60) + 4 for n in longest]
    col_w = [width * w / sum(weights) for w in weights]

    def _cell(text: str, i: int, head: bool) -> Paragraph:
        style = ParagraphStyle(f"c{i}{head}", parent=styles["cell_head" if head else "cell"],
                               alignment=2 if align[i] == "RIGHT" else 0)
        return Paragraph(inline(text), style)

    data = [[_cell(t, i, True) for i, t in enumerate(header)]]
    data += [[_cell(t, i, False) for i, t in enumerate(row)] for row in body]
    table = Table(data, colWidths=col_w, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), HEADER_BG),
        ("LINEBELOW", (0, 0), (-1, 0), 0.8, ACCENT),
        ("LINEBELOW", (0, 1), (-1, -1), 0.3, GRID),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ("TOPPADDING", (0, 0), (-1, -1), 2.2),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 2.2),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.5),
    ]))
    return table


def _image(path: Path, caption: str, styles: dict[str, ParagraphStyle], max_w: float,
           max_h: float) -> KeepTogether:
    if not path.exists():
        raise FileNotFoundError(f"image referenced by the report not found: {path}")
    img_w, img_h = ImageReader(str(path)).getSize()
    scale = min(max_w / img_w, max_h / img_h)
    image = Image(str(path), width=img_w * scale, height=img_h * scale)
    image.hAlign = "CENTER"
    return KeepTogether([image, Paragraph(inline(caption), styles["caption"])])


# --------------------------------------------------------------------------- document
def build(md_path: Path, pdf_path: Path) -> int:
    """Render ``md_path`` to ``pdf_path``; return the number of pages written."""
    styles = _styles()
    portrait_w, portrait_h = A4
    land_w, land_h = landscape(A4)

    def _footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#666666"))
        page_w = canvas._pagesize[0]  # current page (portrait or landscape), not the doc default
        canvas.drawString(MARGIN, 0.9 * cm, "DM4ML Assignment 2 - End-to-end recommendation pipeline")
        canvas.drawRightString(page_w - MARGIN, 0.9 * cm, f"Page {doc.page}")
        canvas.restoreState()

    doc = BaseDocTemplate(str(pdf_path), pagesize=A4, title="DM4ML Assignment 2 - Final report",
                          author="B. Somugi Priya (2025AG05455)", leftMargin=MARGIN, rightMargin=MARGIN,
                          topMargin=MARGIN, bottomMargin=MARGIN)
    doc.addPageTemplates([
        PageTemplate("portrait", pagesize=A4, onPage=_footer,
                     frames=[Frame(MARGIN, MARGIN, portrait_w - 2 * MARGIN, portrait_h - 2 * MARGIN, id="p")]),
        PageTemplate("landscape", pagesize=landscape(A4), onPage=_footer,
                     frames=[Frame(MARGIN, MARGIN, land_w - 2 * MARGIN, land_h - 2 * MARGIN, id="l")]),
    ])
    frame_w = {"portrait": portrait_w - 2 * MARGIN, "landscape": land_w - 2 * MARGIN}
    frame_h = {"portrait": portrait_h - 2 * MARGIN, "landscape": land_h - 2 * MARGIN}

    story: list = []
    orientation = "portrait"
    next_max_h: float | None = None
    lines = md_path.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        line = lines[i].rstrip()
        stripped = line.strip()
        if not stripped:
            i += 1
            continue

        comment = _COMMENT.match(stripped)
        if comment:
            hint = comment.group(1)
            if hint == "pagebreak":
                story.append(PageBreak())
            elif hint in ("landscape", "portrait"):
                orientation = hint
                story += [NextPageTemplate(hint), PageBreak()]
            elif hint.startswith("img"):
                opts = dict(kv.split("=", 1) for kv in hint.split()[1:])
                next_max_h = _length(opts["max_h"]) if "max_h" in opts else None
            i += 1
            continue

        heading = _HEADING.match(stripped)
        if heading:
            level = len(heading.group(1))
            story.append(Paragraph(inline(heading.group(2)), styles[f"h{level}"]))
            if level == 1:
                story.append(HRFlowable(width="100%", thickness=1.2, color=ACCENT, spaceAfter=5))
            i += 1
            continue

        image = _IMAGE.match(stripped)
        if image:
            max_h = min(next_max_h or frame_h[orientation] * 0.45, frame_h[orientation] - 1.5 * cm)
            story.append(_image((md_path.parent / image.group(2)).resolve(), image.group(1), styles,
                                frame_w[orientation], max_h))
            next_max_h = None
            i += 1
            continue

        if stripped.startswith("|"):
            rows = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                rows.append(lines[i])
                i += 1
            story += [_table(rows, styles, frame_w[orientation]), Spacer(1, 6)]
            continue

        bullet = _BULLET.match(stripped)
        if bullet:
            while i < len(lines) and _BULLET.match(lines[i].strip()):
                text = [_BULLET.match(lines[i].strip()).group(1)]
                i += 1
                while i < len(lines) and lines[i].startswith("  ") and lines[i].strip() \
                        and not _BULLET.match(lines[i].strip()):
                    text.append(lines[i].strip())
                    i += 1
                story.append(Paragraph(inline(" ".join(text)), styles["bullet"], bulletText="•"))
            story.append(Spacer(1, 2))
            continue

        para = [stripped]
        i += 1
        while i < len(lines) and lines[i].strip() and not (
                _HEADING.match(lines[i].strip()) or _IMAGE.match(lines[i].strip())
                or _BULLET.match(lines[i].strip()) or lines[i].strip().startswith(("|", "<!--"))):
            para.append(lines[i].strip())
            i += 1
        story.append(Paragraph(inline(" ".join(para)), styles["body"]))

    doc.build(story)
    return doc.page


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=ROOT / "reports" / "final_report.md")
    parser.add_argument("--output", type=Path, default=ROOT / "reports" / "final_report.pdf")
    args = parser.parse_args()
    pages = build(args.input.resolve(), args.output.resolve())
    print(f"wrote {args.output} ({pages} pages)")


if __name__ == "__main__":
    main()
