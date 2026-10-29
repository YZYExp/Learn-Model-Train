#!/usr/bin/env python3
"""Check footnote definitions and generated citation anchors (not source truth)."""
import json,re
from pathlib import Path
from html.parser import HTMLParser
root=Path(__file__).resolve().parent.parent
errors=[]
class Page(HTMLParser):
    def __init__(self):super().__init__();self.ids=[];self.targets=[]
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.append(a['id'])
        if tag=='a' and a.get('href','').startswith(('#source-','#fn')):self.targets.append(a['href'][1:])
for c in json.loads((root/'book.json').read_text())['chapters']:
    p=root/c['path'];text=p.read_text()
    text=re.sub(r'```[\s\S]*?```','',text)
    defs=re.findall(r'^\[\^([^\]]+)\]:',text,re.M)
    uses=re.findall(r'\[\^([^\]]+)\](?!:)',text)
    for key in set(uses)-set(defs):errors.append(f'{p.name}: undefined footnote {key}')
    if len(defs)!=len(set(defs)):errors.append(f'{p.name}: duplicate footnote definitions')
    page=Page();page.feed((root/'build/site/chapters'/f"{c['slug']}.html").read_text())
    for target in page.targets:
        if target not in page.ids:errors.append(f'{p.name}: missing anchor {target}')
    if len(page.ids)!=len(set(page.ids)):errors.append(f'{p.name}: duplicate HTML IDs')
    if '本页参考资料' not in (root/'build/site/chapters'/f"{c['slug']}.html").read_text():errors.append(f'{p.name}: no reference panel')
if errors:raise SystemExit('\n'.join(errors))
print('OK: all chapter footnotes, reference panels and citation anchors')
