#!/usr/bin/env python3
"""
fix_table_alignment.py

Automatically adjusts markdown table column alignment:
  - Columns where all cells have < THRESHOLD words  →  centered  (:--:)
  - Columns where any cell has >= THRESHOLD words    →  left      (:-- )

Words inside backtick code spans are excluded from the count so that
short inline code labels (e.g. `test_size=0.2`) don't push a column
to left-align on their own.

Usage:
  python fix_table_alignment.py docs/*.md
  python fix_table_alignment.py docs/*.md --threshold 4
  python fix_table_alignment.py docs/*.md --dry-run
"""

import re
import argparse
from pathlib import Path


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------

def strip_markdown(text: str) -> str:
    """Remove code spans, bold/italic markers, and HTML tags from text."""
    text = re.sub(r'`[^`]*`', '', text)               # `inline code`
    text = re.sub(r'\*{1,3}([^*]*)\*{1,3}', r'\1', text)  # *italic* / **bold**
    text = re.sub(r'_{1,3}([^_]*)_{1,3}', r'\1', text)    # _italic_ / __bold__
    text = re.sub(r'<[^>]+>', '', text)                # <tags>
    return text


def count_words(cell: str) -> int:
    """Count English words in a cell, ignoring backtick code content."""
    return len(strip_markdown(cell).split())


def parse_cells(line: str) -> list[str]:
    """Split a markdown table row into a list of stripped cell strings."""
    stripped = line.strip()
    if stripped.startswith('|'):
        stripped = stripped[1:]
    if stripped.endswith('|'):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split('|')]


# ---------------------------------------------------------------------------
# Separator detection
# ---------------------------------------------------------------------------

_SEP_RE = re.compile(r'^[\|\:\-\s]+$')


def is_separator_line(line: str) -> bool:
    """
    Return True if a line looks like a table separator row.

    Catches both well-formed  |:--|:--|  and malformed  | | |  or  | : | : |
    by requiring the line to contain only  |  :  -  and spaces.
    """
    stripped = line.strip()
    return bool(_SEP_RE.match(stripped)) and '|' in stripped


# ---------------------------------------------------------------------------
# Separator builder
# ---------------------------------------------------------------------------

def build_separator(alignments: list[str]) -> str:
    """Return a new separator row string from a list of 'center'/'left' values."""
    parts = [':--:' if a == 'center' else ':--' for a in alignments]
    return '|' + '|'.join(parts) + '|'


# ---------------------------------------------------------------------------
# Core table fixer
# ---------------------------------------------------------------------------

def fix_tables(text: str, threshold: int) -> str:
    """
    Scan *text* for markdown tables and rewrite each separator row so that:
      - columns whose data cells all have < threshold words  → centered
      - columns with any data cell having >= threshold words → left
    """
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    in_code_block = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # ── track fenced code blocks ──────────────────────────────────────
        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue

        if in_code_block:
            result.append(line)
            i += 1
            continue

        # ── table detection: header row followed by a separator ───────────
        if (
            stripped.startswith('|')
            and i + 1 < len(lines)
            and is_separator_line(lines[i + 1])
        ):
            header_cells = parse_cells(line)
            n_cols = len(header_cells)

            if n_cols < 1:
                result.append(line)
                i += 1
                continue

            # Collect data rows (everything after the separator that still
            # looks like a table row and isn't another separator itself).
            j = i + 2
            data_rows: list[str] = []
            while j < len(lines):
                dl = lines[j]
                ds = dl.strip()
                if ds.startswith('|') and not is_separator_line(dl):
                    data_rows.append(dl)
                    j += 1
                else:
                    break

            # Compute the maximum word count for each column across data rows.
            max_words = [0] * n_cols
            for row in data_rows:
                for col, cell in enumerate(parse_cells(row)[:n_cols]):
                    max_words[col] = max(max_words[col], count_words(cell))

            alignments = [
                'center' if w < threshold else 'left'
                for w in max_words
            ]

            new_sep = build_separator(alignments)

            # Preserve the original line ending.
            orig_ending = '\n' if lines[i + 1].endswith('\n') else ''
            new_sep += orig_ending

            result.append(line)
            result.append(new_sep)
            result.extend(data_rows)
            i = j
            continue

        result.append(line)
        i += 1

    return ''.join(result)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            'Fix markdown table column alignment based on cell word count.\n\n'
            'Columns where all cells have fewer than --threshold words are\n'
            'centered; columns with any longer cell are left-aligned.\n'
            'Words inside backtick code spans do not count toward the total.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='+',
        help='Markdown files to process (supports shell globs)',
    )
    parser.add_argument(
        '--threshold', '-t',
        type=int,
        default=5,
        metavar='N',
        help=(
            'A column is left-aligned when any cell has >= N words '
            '(default: 5)'
        ),
    )
    parser.add_argument(
        '--dry-run', '-n',
        action='store_true',
        help='Show which files would change without writing them',
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Print a line for every file, even unchanged ones',
    )
    args = parser.parse_args()

    changed = 0
    skipped = 0

    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f'  missing  {path}')
            skipped += 1
            continue

        original = path.read_text(encoding='utf-8')
        fixed = fix_tables(original, args.threshold)

        if fixed == original:
            if args.verbose:
                print(f'unchanged  {path}')
        elif args.dry_run:
            print(f'  would update  {path}')
            changed += 1
        else:
            path.write_text(fixed, encoding='utf-8')
            print(f'  updated  {path}')
            changed += 1

    if args.dry_run and changed:
        print(f'\n{changed} file(s) would be updated (--dry-run, nothing written)')
    elif not args.dry_run:
        if changed or args.verbose:
            print(f'\n{changed} file(s) updated, {skipped} skipped')


if __name__ == '__main__':
    main()
