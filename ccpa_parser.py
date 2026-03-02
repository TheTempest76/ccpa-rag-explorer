"""
CCPA Statute Parser — reads and parses data/ccpa_statute.txt into structured chunks.

All legal content in this system derives exclusively from data/ccpa_statute.txt.
No section numbers, titles, or legal text are hardcoded anywhere in this module.
The parser reads the statute file at runtime and produces a flat list of chunk
dictionaries suitable for TF-IDF indexing and search.
"""

import re
import sys
from pathlib import Path


def _clean_lines(raw_text: str) -> list[str]:
    """Remove table-of-contents lines, page markers, and header from raw statute text."""
    lines = raw_text.splitlines()
    cleaned = []
    for line in lines:
        # Skip TOC lines with dotted leaders
        if "......" in line:
            continue
        # Skip TOC lines ending with dot-space-number patterns (e.g. "Rights . 10")
        if re.search(r"\.\s+\d+\s*$", line) and "1798." in line:
            # Only skip if it looks like a TOC entry (section number + title + page ref)
            if not re.match(r"^\s*\(", line.strip()):
                continue
        # Skip page markers like "Page 3 of 65"
        if re.match(r"^\s*Page\s+\d+\s+of\s+\d+\s*$", line):
            continue
        # Skip footnote lines about monetary adjustments
        if line.strip().startswith("* Pursuant to Civil Code"):
            continue
        if line.strip().startswith("amount."):
            continue
        cleaned.append(line)

    # Skip the header and table of contents at the beginning of the file.
    # Find where the actual body starts: the first section header line whose
    # IMMEDIATE next non-blank line starts with a subsection marker like (a).
    section_re = re.compile(r"^\s*(1798\.\d+(?:\.\d+)?)\.")
    sub_re = re.compile(r"^\s*\([a-z]\)")
    body_start = 0
    for i, line in enumerate(cleaned):
        if section_re.match(line.strip()):
            # Check only the very next 1-2 non-blank lines (must be adjacent)
            for j in range(i + 1, min(i + 3, len(cleaned))):
                sj = cleaned[j].strip()
                if not sj:
                    continue  # skip blanks
                if section_re.match(sj):
                    break  # hit another section header — this was a TOC entry
                if sub_re.match(sj):
                    body_start = i
                    break
                break  # non-blank, non-section, non-subsection — skip
            if body_start > 0:
                break

    return cleaned[body_start:]


def _extract_section_blocks(lines: list[str]) -> list[tuple[str, str, list[str]]]:
    """
    Split cleaned lines into top-level section blocks.
    Returns list of (section_number, title, block_lines).
    """
    # Pattern matches top-level section headers like:
    #   1798.100. General Duties of Businesses...
    #   1798.199.10.
    #   1798.146.
    section_pattern = re.compile(r"^(1798\.\d+(?:\.\d+)?)\.\s*(.*)")

    blocks: list[tuple[str, str, list[str]]] = []
    current_section = None
    current_title = ""
    current_lines: list[str] = []

    for line in lines:
        m = section_pattern.match(line.strip())
        if m:
            # Save previous block
            if current_section is not None:
                blocks.append((current_section, current_title.strip(), current_lines))
            current_section = m.group(1)
            title_part = m.group(2).strip()
            current_title = title_part
            current_lines = []
            # If there's remaining text on the same line after title, keep it
            # The title may span multiple lines — we'll collect those below
        else:
            if current_section is not None:
                # Check if this line is a continuation of the title
                # (non-empty, not starting with (a) or (1), and title not yet finished)
                stripped = line.strip()
                if (
                    current_title
                    and not current_lines
                    and stripped
                    and not re.match(r"^\(", stripped)
                    and not current_title.endswith(".")
                ):
                    # This is a title continuation line
                    current_title += " " + stripped
                else:
                    current_lines.append(line)

    # Save final block
    if current_section is not None:
        blocks.append((current_section, current_title.strip(), current_lines))

    return blocks


def _parse_subsections(
    section_num: str, title: str, block_lines: list[str]
) -> list[dict]:
    """
    Parse a section block into chunk dicts for lettered subsections and
    numbered paragraphs. Also produces a section-level chunk with full text.
    """
    full_text = "\n".join(block_lines).strip()
    chunks: list[dict] = []

    # ── Find lettered subsections: (a), (b), ... ──────────────────────
    letter_pattern = re.compile(r"^\s*\(([a-z]+)\)\s+(.*)", re.DOTALL)

    # Split block into lettered subsection spans
    letter_spans: list[tuple[str, int, int]] = []  # (letter, start, end)
    for i, line in enumerate(block_lines):
        m = letter_pattern.match(line)
        if m:
            letter = m.group(1)
            # Only accept single lowercase letters a-z, or multi-letter codes
            # like (aa), (ab), etc. used in definitions section
            if len(letter) <= 2:
                letter_spans.append((letter, i, -1))

    # Set end indices
    for idx in range(len(letter_spans)):
        if idx + 1 < len(letter_spans):
            letter_spans[idx] = (
                letter_spans[idx][0],
                letter_spans[idx][1],
                letter_spans[idx + 1][1],
            )
        else:
            letter_spans[idx] = (
                letter_spans[idx][0],
                letter_spans[idx][1],
                len(block_lines),
            )

    if letter_spans:
        for letter, start, end in letter_spans:
            sub_lines = block_lines[start:end]
            sub_text = "\n".join(sub_lines).strip()
            sub_label = f"({letter})"

            # ── Find numbered paragraphs within this lettered subsection ──
            num_pattern = re.compile(r"^\s*\((\d+)\)\s+(.*)", re.DOTALL)
            num_spans: list[tuple[str, int, int]] = []
            for j, sline in enumerate(sub_lines):
                nm = num_pattern.match(sline)
                if nm:
                    num_spans.append((nm.group(1), j, -1))

            # Set end indices for numbered spans
            for nidx in range(len(num_spans)):
                if nidx + 1 < len(num_spans):
                    num_spans[nidx] = (
                        num_spans[nidx][0],
                        num_spans[nidx][1],
                        num_spans[nidx + 1][1],
                    )
                else:
                    num_spans[nidx] = (
                        num_spans[nidx][0],
                        num_spans[nidx][1],
                        len(sub_lines),
                    )

            if num_spans:
                # Create numbered paragraph chunks
                for num, nstart, nend in num_spans:
                    para_lines = sub_lines[nstart:nend]
                    para_text = "\n".join(para_lines).strip()
                    para_sub = f"({letter})({num})"
                    chunks.append(
                        {
                            "id": f"{section_num}{para_sub}",
                            "section": section_num,
                            "sub": para_sub,
                            "label": f"Section {section_num}{para_sub}",
                            "title": title,
                            "text": para_text,
                            "depth": 2,
                        }
                    )

            # Always create a lettered subsection chunk too
            chunks.append(
                {
                    "id": f"{section_num}{sub_label}",
                    "section": section_num,
                    "sub": sub_label,
                    "label": f"Section {section_num}{sub_label}",
                    "title": title,
                    "text": sub_text,
                    "depth": 1,
                }
            )

    # Always create a section-level chunk with all text
    chunks.append(
        {
            "id": section_num,
            "section": section_num,
            "sub": "",
            "label": f"Section {section_num}",
            "title": title,
            "text": full_text if full_text else title,
            "depth": 0,
        }
    )

    return chunks


def parse_statute(filepath: Path) -> list[dict]:
    """
    Parse the CCPA statute file into a flat list of chunk dicts.

    Each chunk dict has these keys:
        id      — e.g. "1798.120(c)" or "1798.120"
        section — e.g. "1798.120"
        sub     — e.g. "(c)" or "(a)(1)" or "" for section-level
        label   — e.g. "Section 1798.120(c)"
        title   — section title from the statute
        text    — full cleaned text of this chunk
        depth   — 0=section-level, 1=lettered sub, 2=numbered para

    Args:
        filepath: Path to data/ccpa_statute.txt

    Returns:
        List of chunk dicts parsed from the statute.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        print(
            "ERROR: data/ccpa_statute.txt not found. "
            "Please place the CCPA statute text file in the data/ directory.",
            file=sys.stderr,
        )
        sys.exit(1)

    raw_text = filepath.read_text(encoding="utf-8", errors="ignore")
    lines = _clean_lines(raw_text)
    blocks = _extract_section_blocks(lines)

    all_chunks: list[dict] = []
    for section_num, title, block_lines in blocks:
        section_chunks = _parse_subsections(section_num, title, block_lines)
        all_chunks.extend(section_chunks)

    # Quality check
    if len(all_chunks) < 50:
        print(
            f"WARNING: Parser produced only {len(all_chunks)} chunks. "
            "Expected at least 80. The parser may be too coarse.",
            file=sys.stderr,
        )

    return all_chunks


if __name__ == "__main__":
    base = Path(__file__).parent
    statute_path = base / "data" / "ccpa_statute.txt"
    chunks = parse_statute(statute_path)
    print(f"Parsed {len(chunks)} chunks from {statute_path}")
    for c in chunks[:10]:
        print(f"  {c['label']:40s}  depth={c['depth']}  text={len(c['text'])} chars")
    print("  ...")
    for c in chunks[-5:]:
        print(f"  {c['label']:40s}  depth={c['depth']}  text={len(c['text'])} chars")
