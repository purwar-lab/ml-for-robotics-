#!/usr/bin/env python3
"""
table_left_align.py

Sets every column in every markdown table to left-aligned.

Usage:
  python3 table_left_align.py docs/*.md
  python3 table_left_align.py docs/*.md --dry-run
"""

import re
import argparse
from pathlib import Path

_SEP_RE = re.compile(r'^[\|\:\-\s]+$')


def is_separator_line(line: str) -> bool:
    stripped = line.strip()
    return bool(_SEP_RE.match(stripped)) and '|' in stripped


def col_count(line: str) -> int:
    stripped = line.strip().strip('|')
    return len(stripped.split('|'))


def make_left_separator(n_cols: int, ending: str) -> str:
    return '|' + '|'.join(':--' for _ in range(n_cols)) + '|' + ending


def fix_tables(text: str) -> str:
    lines = text.splitlines(keepends=True)
    result: list[str] = []
    in_code_block = False
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped.startswith('```') or stripped.startswith('~~~'):
            in_code_block = not in_code_block
            result.append(line)
            i += 1
            continue

        if in_code_block:
            result.append(line)
            i += 1
            continue

        if (
            stripped.startswith('|')
            and i + 1 < len(lines)
            and is_separator_line(lines[i + 1])
        ):
            n_cols = col_count(line)
            ending = '\n' if lines[i + 1].endswith('\n') else ''
            result.append(line)
            result.append(make_left_separator(n_cols, ending))
            i += 2
            continue

        result.append(line)
        i += 1

    return ''.join(result)


def main() -> None:
    parser = argparse.ArgumentParser(description='Set all markdown table columns to left-aligned.')
    parser.add_argument('files', nargs='+', help='Markdown files to process')
    parser.add_argument('--dry-run', '-n', action='store_true', help='Show changes without writing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Print unchanged files too')
    args = parser.parse_args()

    changed = skipped = 0

    for path_str in args.files:
        path = Path(path_str)
        if not path.exists():
            print(f'  missing  {path}')
            skipped += 1
            continue

        original = path.read_text(encoding='utf-8')
        fixed = fix_tables(original)

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
    elif not args.dry_run and (changed or args.verbose):
        print(f'\n{changed} file(s) updated, {skipped} skipped')


if __name__ == '__main__':
    main()
