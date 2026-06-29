#!/usr/bin/env python3
"""
fix_table_alignment.py

Automatically adjusts markdown table column alignment:
  - Columns where all cells have < THRESHOLD words  →  centered  (:--:)
  - Columns where any cell has >= THRESHOLD words    →  left      (:-- )
  - Columns where any cell contains backtick code   →  left      (:-- )
    (disable with --no-code-left)

Words inside backtick code spans are excluded from the word count so
that short inline code labels don't push a column left via word count
alone — but with --code-left (the default) any backtick in any cell
forces the whole column left regardless of word count.

Usage:
  python fix_table_alignment.py docs/*.md
  python fix_table_alignment.py docs/*.md --threshold 4
  python fix_table_alignment.py docs/*.md --no-code-left
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


def cell_has_code(cell: str) -> bool:
    """Return True if the cell contains at least one backtick code span."""
    return bool(re.search(r'`[^`]+`', cell))


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

def fix_tables(text: str, threshold: int, code_left: bool = True) -> str:
    """
    Scan *text* for markdown tables and rewrite each separator row so that:
      - columns whose data cells all have < threshold words  → centered
      - columns with any data cell having >= threshold words → left
      - columns where any data cell contains backtick code  → left
        (only when code_left is True)
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

            # Compute per-column stats across all data rows.
            max_words = [0] * n_cols
            table_has_code = False
            for row in data_rows:
                for col, cell in enumerate(parse_cells(row)[:n_cols]):
                    max_words[col] = max(max_words[col], count_words(cell))
                    if code_left and cell_has_code(cell):
                        table_has_code = True

            table_has_long = any(w >= threshold for w in max_words)
            force_left = table_has_code or table_has_long

            alignments = [
                'left' if force_left else 'center'
                for _ in range(n_cols)
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
        '--code-left',
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            'Force left-alignment for any column that contains backtick code '
            'in at least one cell (default: enabled). '
            'Use --no-code-left to rely on word count alone.'
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
        fixed = fix_tables(original, args.threshold, code_left=args.code_left)

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
