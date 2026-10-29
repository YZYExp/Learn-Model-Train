# 04 训练工程栈：从单卡到集群

训练方案可按层认识，避免把训练算法、分布式机制和生成引擎混成同一类工具。

| 层次 | 代表方案 | 用途与边界 |
| --- | --- | --- |
| 实验与后训练 | [Transformers/PEFT](https://huggingface.co/docs/peft/index)、[TRL](https://huggingface.co/docs/trl/index)、[LlamaFactory](https://llamafactory.readthedocs.io/en/latest/) | 快速尝试微调和后训练；模型及算法支持按版本核对 |
| 分布式状态 | [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)、[DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) | 分片参数、梯度或优化器状态；通信与显存存在权衡 |
| 大规模并行 | [TorchTitan](https://github.com/pytorch/torchtitan)、[Megatron Core](https://docs.nvidia.com/megatron-core/developer-guide/latest/user-guide/parallelism-guide.html) | 组合数据、张量、流水线、上下文及专家并行；实施复杂度较高 |
| 在线 RL | [verl](https://verl.readthedocs.io/en/latest/) | 协调生成、训练和奖励；生成后端与训练算法是不同层 |
| 权重与训练桥接 | [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | HF 与 Megatron 生态衔接及训练流程；须核查模型兼容性 |

推荐实验路径：单卡小模型基线 → LoRA → 多卡分片 → 性能分析 → 小型在线 RL → 更大规模并行。每一步保存数据版本、模型和 tokenizer、聊天模板、依赖与驱动、随机种子、训练 token、吞吐、峰值显存、评测和失败记录。训练与推理使用的硬件、模型架构及算子兼容性要逐项验证，不以厂商名称推断支持。

数据并行复制模型并分配样本；FSDP 分片模型状态，计算时需要通信；张量并行拆分层内计算，流水线并行把层分到不同设备。并行方式可组合，但会引入通信、流水线空闲和调试成本。后续章节将以实际配置和实验说明，而不是仅按术语罗列。
