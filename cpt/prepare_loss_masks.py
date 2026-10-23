#!/usr/bin/env python3
"""Validate/normalize CPT JSONL loss masks without importing ML libraries."""
import argparse
import json
from pathlib import Path
from loss_masks import normalize_loss_masks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='write canonical Swift messages; omit for validation only')
    args = parser.parse_args()
    if args.output and args.output.resolve() == args.input.resolve():
        parser.error('Input and output must be different files')
    # Never overwrite an existing dataset; report line numbers, not training content.
    count = 0
    out = args.output.open('x', encoding='utf-8') if args.output else None
    try:
        with args.input.open(encoding='utf-8') as src:
            for line_no, line in enumerate(src, 1):
                if not line.strip():
                    continue
                try:
                    row = normalize_loss_masks(json.loads(line))
                except (ValueError, TypeError, AttributeError) as exc:
                    raise ValueError(f'Invalid loss mask at line {line_no}: {exc}') from None
                if out:
                    out.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
    except Exception:
        if out:
            out.close()
            args.output.unlink(missing_ok=True)
        raise
    finally:
        if out and not out.closed:
            out.close()
    print(f'Validated {count} rows' + ('; wrote normalized JSONL' if out else ''))

if __name__ == '__main__':
    main()
