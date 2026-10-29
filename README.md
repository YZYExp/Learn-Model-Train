# 大模型训练：从目标到系统

面向有编程经验的读者，自顶向下讲解决策、数据、训练方法、工程、评估和交付，底层原理作为可选深入路径。

当前为研究与书稿初期，参见 [规划](PLAN.md) 和 [章节清单](book.json)。资料核查截至 2026-10-29；实际训练与性能尚未验证。

检查章节清单、相对链接与 SVG：`python scripts/check_book.py`。运行训练循环教学示例：`python scripts/tiny_bigram.py`。代码示例中除明确记录已运行的案例外，均是待结合环境验证的示意；远程引用可用性未由上述检查脚本验证。

本地构建阅读站：`npm ci`，然后运行 `node scripts/build_site.cjs --base-path /Learn-Model-Train/`；结果位于 `build/site/`。GitHub Actions 从 `book.json` 构建章节并发布 Pages。
