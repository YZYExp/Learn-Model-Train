#!/usr/bin/env python3
"""Estimate dense Transformer KV payload, excluding allocator overhead and sharing."""
import argparse

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--layers', type=int, required=True)
p.add_argument('--kv-heads', type=int, required=True)
p.add_argument('--head-dim', type=int, required=True)
p.add_argument('--tokens', type=int, required=True, help='tokens per sequence')
p.add_argument('--sequences', type=int, default=1)
p.add_argument('--bytes-per-element', type=float, default=2.0)
a = p.parse_args()
if min(a.layers, a.kv_heads, a.head_dim, a.tokens, a.sequences, a.bytes_per_element) <= 0:
    p.error('all values must be positive')
per_token = 2 * a.layers * a.kv_heads * a.head_dim * a.bytes_per_element
total = per_token * a.tokens * a.sequences
print(f'KV payload: {per_token:,.0f} bytes/token/sequence')
print(f'Total: {total / 2**30:.3f} GiB (idealized; before allocator overhead)')
print('Assumes full attention, equal layer dimensions, no shared prefixes or sharding.')
