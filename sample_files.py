"""
Deterministic PDF and Word fixtures, for the tests and the sample corpus.

The PDF writer is hand-rolled (one Helvetica text object per page, one line per
`T*`) so the tests need no PDF-authoring dependency and every byte is
reproducible. Word files are written with python-docx.

    python sample_files.py      # regenerate the binaries in sample-docs/
"""

from __future__ import annotations

from pathlib import Path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_pdf(path: Path, pages: list[list[str]]) -> None:
    """Write a text-layer PDF: pages[i] is the list of lines on page i+1."""
    objects: list[bytes] = []
    page_ids = [4 + 2 * i for i in range(len(pages))]
    objects.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    kids = " ".join(f"{pid} 0 R" for pid in page_ids)
    objects.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(pages)} >>".encode())
    objects.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, lines in enumerate(pages):
        body = "BT /F1 11 Tf 14 TL 72 760 Td " + " T* ".join(
            f"({_pdf_escape(line)}) Tj" for line in lines
        ) + " ET" if lines else ""
        stream = body.encode("latin-1")
        objects.append(
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            f"/Resources << /Font << /F1 3 0 R >> >> /Contents {page_ids[i] + 1} 0 R >>".encode()
        )
        objects.append(b"<< /Length %d >>\nstream\n" % len(stream) + stream + b"\nendstream")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for num, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out += f"{num} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    out += b"".join(f"{off:010d} 00000 n \n".encode() for off in offsets)
    out += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(bytes(out))


def write_docx(path: Path, blocks: list[tuple]) -> None:
    """
    blocks: ("h", level, text) | ("p", text) | ("table", [[cell, ...], ...]).
    """
    import docx  # python-docx

    doc = docx.Document()
    for block in blocks:
        if block[0] == "h":
            doc.add_heading(block[2], level=block[1])
        elif block[0] == "p":
            doc.add_paragraph(block[1])
        elif block[0] == "table":
            rows = block[1]
            table = doc.add_table(rows=len(rows), cols=len(rows[0]))
            for r, row in enumerate(rows):
                for c, cell in enumerate(row):
                    table.cell(r, c).text = cell
    path.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(path))


SAMPLE_PDF = [
    [
        "Latency Review",
        "",
        "This review covers the p95 latency of the search endpoint after the",
        "index moved from a linear scan to an inverted index.",
        "",
        "Median latency fell from 0.3 ms to 0.04 ms on a 1200 chunk corpus.",
    ],
    [
        "Cold start",
        "",
        "A new process loads per-chunk term counts from the cache, so startup",
        "no longer re-tokenizes unchanged files. Cold start fell to 0.10 s.",
    ],
]

SAMPLE_DOCX = [
    ("h", 1, "Deployment Runbook"),
    ("p", "How to roll the docs server out to a team machine."),
    ("h", 2, "Rollback"),
    ("p", "To roll back, reinstall the previous release tag and restart the client."),
    ("p", "The index cache is versioned, so an older release discards a newer cache."),
    ("h", 2, "Supported versions"),
    ("table", [["Python", "Status"], ["3.10", "tested in CI"], ["3.12", "tested in CI"]]),
]


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "sample-docs"
    write_pdf(root / "reports" / "latency-review.pdf", SAMPLE_PDF)
    write_docx(root / "guides" / "deployment-runbook.docx", SAMPLE_DOCX)
    print("wrote sample-docs/reports/latency-review.pdf, sample-docs/guides/deployment-runbook.docx")
