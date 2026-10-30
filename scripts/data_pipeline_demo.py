#!/usr/bin/env python3
"""A tiny reproducible data-preparation audit. No network or ML dependency.

Exact-duplicate groups and repository groups are unioned before split so that
both constraints hold together; this does not detect near duplicates.
"""
import hashlib
import json
import random
from collections import Counter

RECORDS = [
    {'id':'a','repository':'alpha','prompt':'locate session persistence','completion':'session.py'},
    {'id':'b','repository':'alpha','prompt':'locate rendering','completion':'view.py'},
    {'id':'c','repository':'beta','prompt':'locate database connection','completion':'db.py'},
    # copied content bridges beta and gamma; they must be assigned together.
    {'id':'d','repository':'gamma','prompt':'locate database connection','completion':'db.py'},
    {'id':'e','repository':'delta','prompt':'locate tests','completion':'test_api.py'},
    {'id':'f','repository':'epsilon','prompt':'locate cache','completion':'cache.py'},
]

def content_hash(row):
    # Use unambiguous encoding; never strip indentation from actual code.
    payload = json.dumps([row['prompt'], row['completion']], ensure_ascii=False,
                         separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()

def build_components(records):
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    def union(a, b):
        a,b=find(a),find(b)
        if a != b: parent[b] = a
    by_repo, by_hash = {}, {}
    for i, row in enumerate(records):
        for key, table in ((row['repository'], by_repo), (content_hash(row), by_hash)):
            if key in table: union(i, table[key])
            else: table[key] = i
    groups = {}
    for i in range(len(records)):
        groups.setdefault(find(i), []).append(i)
    return list(groups.values())

def split_and_dedupe(records, seed=42):
    # Fixed ordering + fixed seed; changing input contents changes the result.
    groups = sorted(build_components(records), key=lambda g: min(records[i]['id'] for i in g))
    random.Random(seed).shuffle(groups)
    # This toy example allocates one component to evaluation. Production should
    # balance strata and size, reserve test separately and report actual counts.
    eval_ids = set(groups[0])
    # Check *all* inputs before removing duplicates, including dropped copies.
    pre = [{**row, 'split': 'eval' if i in eval_ids else 'train',
            'sha256': content_hash(row)} for i, row in enumerate(records)]
    validate(pre)
    result, seen, duplicates = [], set(), []
    for i, row in enumerate(records):
        digest = content_hash(row)
        if digest in seen:
            duplicates.append(row['id'])
            continue
        seen.add(digest)
        result.append({**row, 'split':'eval' if i in eval_ids else 'train', 'sha256':digest})
    return result, duplicates

def validate(rows):
    repo_split, hash_split = {}, {}
    for row in rows:
        for key, table in ((row['repository'], repo_split), (row['sha256'], hash_split)):
            previous = table.setdefault(key, row['split'])
            assert previous == row['split'], f'leak: {key}'
    assert set(row['split'] for row in rows) == {'train', 'eval'}

if __name__ == '__main__':
    data, removed = split_and_dedupe(RECORDS)
    validate(data)
    assert len(data) + len(removed) == len(RECORDS)
    print(json.dumps({'records_in':len(RECORDS), 'records_out':len(data),
                      'exact_duplicates_removed':removed,
                      'split_counts':dict(Counter(x['split'] for x in data)),
                      'seed':42}, ensure_ascii=False, sort_keys=True))
