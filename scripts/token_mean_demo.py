#!/usr/bin/env python3
"""Show why averaging rank-local means is wrong for unequal token counts.

This uses scalar losses only; no distributed runtime or GPU is involved.
"""
from fractions import Fraction

# Each inner list represents one data-parallel rank's valid token losses.
ranks = [[Fraction(1), Fraction(2), Fraction(3)], [Fraction(10)]]
rank_means = [sum(losses) / len(losses) for losses in ranks]
naive = sum(rank_means) / len(rank_means)
correct = sum(sum(losses) for losses in ranks) / sum(map(len, ranks))
assert naive == 6 and correct == 4
# DDP averages gradients across W ranks. Scale each rank's sum by W/N.
world_size = len(ranks)
total_tokens = sum(map(len, ranks))
local_scaled = [sum(xs) * world_size / total_tokens for xs in ranks]
assert sum(local_scaled) / world_size == correct
print('rank means:', ', '.join(map(str, rank_means)))
print('naive mean:', naive, 'global token mean:', correct)
print('DDP-average scaled local sums:', ', '.join(map(str, local_scaled)))
