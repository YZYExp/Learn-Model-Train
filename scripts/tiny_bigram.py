#!/usr/bin/env python3
"""零依赖字符二元模型：演示交叉熵及梯度下降，不是 Transformer。"""
import math

# 玩具数据：每个字符预测后一个字符。现实数据应按来源拆分并防止泄漏。
text = "abcabcabcabc"
vocab = sorted(set(text))
idx = {c: i for i, c in enumerate(vocab)}
pairs = [(idx[a], idx[b]) for a, b in zip(text, text[1:])]
weights = [[0.0] * len(vocab) for _ in vocab]

def step(learning_rate):
    gradients = [[0.0] * len(vocab) for _ in vocab]
    loss = 0.0
    for current, target in pairs:
        logits = weights[current]
        # 减去最大值可防止指数溢出；不会改变 softmax 的结果。
        scores = [math.exp(x - max(logits)) for x in logits]
        probs = [x / sum(scores) for x in scores]
        loss -= math.log(probs[target])
        for j, probability in enumerate(probs):
            gradients[current][j] += (probability - (j == target)) / len(pairs)
    for row in range(len(vocab)):
        for col in range(len(vocab)):
            weights[row][col] -= learning_rate * gradients[row][col]
    return loss / len(pairs)

if __name__ == '__main__':
    for i in range(101):
        loss = step(1.0)
        if i in (0, 10, 100):
            print(f"step={i:3d} training_loss={loss:.4f}")
    print('注意：这里只演示训练集拟合，没有独立评估。')
