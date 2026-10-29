# 大模型训练：从目标到系统

面向有编程经验的读者，自顶向下讲解决策、数据、训练方法、工程、评估和交付，底层原理作为可选深入路径。

当前为研究与书稿初期，参见 [规划](PLAN.md) 和 [章节清单](book.json)。资料核查截至 2026-10-29；实际训练与性能尚未验证。

检查章节清单、相对链接与 SVG：`python scripts/check_book.py`。运行训练循环教学示例：`python scripts/tiny_bigram.py`。代码示例中除明确记录已运行的案例外，均是待结合环境验证的示意；远程引用可用性未由上述检查脚本验证。

本地构建阅读站：`npm ci`，然后运行 `node scripts/build_site.cjs --base-path /Learn-Model-Train/`；结果位于 `build/site/`。GitHub Actions 从 `book.json` 构建章节并发布 Pages。

## 教学主线

当前目录已改为项目设计、计算与训练方法、工具链完整实现、系统与专题、速查五部分。先读 [顶层工程设计](chapters/16-engineering-design.md) 与 [底层计算](chapters/17-computation-depth.md)，再进入 [HF 工作流](chapters/18-hf-workflow.md)、[NeMo/Megatron](chapters/19-nemo-megatron-workflow.md) 和 [verl](chapters/20-verl-workflow.md)。详细阅读说明见 [阅读路线](appendices/reading-routes.md)。章节中的业务数字为标明条件的算例，GPU 配方仍需在目标环境验证。

阅读站采用 Markdown-it、markdown-it-footnote 和 Highlight.js；代码高亮在构建时生成并随站发布。构建后运行 `python scripts/check_citations.py` 检查脚注、参考资料和回跳锚点。当前“每页”指每个章节网页；本项目尚未交付逐物理页脚注的 PDF。
