#!/usr/bin/env python3
"""检查章节清单、相对 Markdown 链接和 SVG 结构；不检查远程网页。"""
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote

root = Path(__file__).resolve().parent.parent
manifest = json.loads((root / 'book.json').read_text(encoding='utf-8'))
ids = set()
errors = []
for chapter in manifest['chapters']:
    if chapter['id'] in ids:
        errors.append(f"duplicate chapter: {chapter['id']}")
    ids.add(chapter['id'])
    path = root / chapter['path']
    if not path.is_file():
        errors.append(f'missing chapter: {path}')
        continue
    text = path.read_text(encoding='utf-8')
    # Ignore web links and anchor-only links; validate local images and files.
    for dest in re.findall(r'!?\[[^\]]*\]\(([^)\s]+)(?:\s+[^)]*)?\)', text):
        if dest.startswith(('http:', 'https:', 'mailto:', '#')):
            continue
        local = unquote(dest.split('#', 1)[0])
        if local and not (path.parent / local).is_file():
            errors.append(f'{path.relative_to(root)}: missing {dest}')
for path in (root / 'assets/images').glob('*.svg'):
    try:
        ET.parse(path)
    except ET.ParseError as exc:
        errors.append(f'{path}: {exc}')
if errors:
    print('\n'.join(errors))
    raise SystemExit(1)
print(f"OK: {len(ids)} chapters, relative links and SVG syntax")
