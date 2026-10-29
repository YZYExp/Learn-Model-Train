# 06 工业训练工具链：按工作流选型

工具名称经常被并列比较，但数据处理、训练框架、分布式机制、采样引擎和调度平台并非同一层。这里记录项目官方文档所描述的能力，并区分可核实的公开案例、项目方自述与本书尚未验证的结论。检索日期：2026-10-29；接口和支持矩阵会变化。

| 环节 | 方案与作用 | 工程选型重点 | 证据和限制 |
| --- | --- | --- | --- |
| 数据整理 | [NeMo Curator](https://docs.nvidia.com/nemo/curator/home/welcome)：文本及多模态处理、过滤和去重 | 数据格式、分布式调度、GPU 需求、可复现性 | [项目公开记录](https://github.com/NVIDIA-NeMo/Curator)称其文本流程用于 Nemotron-4 数据整理；这是项目方案例，非本书复现 |
| 常规微调 | [Transformers/PEFT](https://huggingface.co/docs/peft/index)、[TRL](https://huggingface.co/docs/trl/index) | Python 可定制、模型与模板兼容、数据和指标控制 | 官方支持多种方法；具体 trainer 的成熟度和接口随版本变化 |
| 配置式微调 | [Axolotl](https://docs.axolotl.ai/)、[LlamaFactory](https://llamafactory.readthedocs.io/en/latest/) | 团队能否维护配置、复现模型模板、分布式和导出 | 官方提供训练流程；不能仅凭支持列表判断特定模型稳定性 |
| 本地资源优化 | [Unsloth](https://unsloth.ai/docs) | 硬件、模型和精度组合；速度与内存应在目标工作负载实测 | 文档提供微调与导出路径；项目宣传性能不直接作为本书结论 |
| 分布式训练 | [PyTorch FSDP2](https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html)、[DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/) | 状态分片、通信、检查点、参数与优化器兼容性 | 是机制/运行时组件，不等同完整训练应用 |
| 大规模训练 | [Megatron Core 与 Bridge](https://docs.nvidia.com/megatron-core/developer-guide/latest/get-started/overview.html)、[TorchTitan](https://github.com/pytorch/torchtitan) | 模型实现、TP/PP/CP/EP、权重转换、网络拓扑及恢复 | 官方分别提供训练和并行实现；特定模型与集群组合须验证 |
| 在线后训练 | [verl](https://verl.readthedocs.io/en/latest/)、[NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | 采样、训练和环境调度；权重同步、吞吐和容错 | 项目方提供在线 RL 流程；生产适用性须按部署测试 |
| 生成与 rollout | [vLLM](https://docs.vllm.ai/en/latest/training/rlhf/) | 推理吞吐、生成和训练权重同步、token 一致性 | vLLM 文档将其定位为 RLHF 生成组件；不是完整的训练算法框架 |
| 蒸馏及压缩 | [TRL GKD](https://huggingface.co/docs/trl/main/gkd_trainer)、[NVIDIA Model Optimizer](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html) | 教师访问、词表对齐、训练与导出兼容 | GKD 文档当前使用 experimental 命名空间；接口需锁定版本 |
| AMD 路线 | [ROCm verl](https://rocm.docs.amd.com/projects/verl/en/latest/)、[AMD Primus](https://rocm.docs.amd.com/projects/primus/en/main/02-user-guide/pretraining.html) | 驱动、通信库、算子及模型组合 | 官方文档列举工作流；不能推断 CUDA 配置可原样运行 |

## 三条典型工程路径

**单机探索。** 数据版本与许可清单 → Transformers/PEFT 或 Axolotl/Unsloth 的小模型 SFT → 独立任务评估 → 导出与目标推理环境验证。先固定模型、模板、数据和资源预算；在此基础上比较全参、LoRA 或量化训练。工具选择取决于模型支持、可调试性和部署路径。

**集群预训练。** 数据处理与 tokenization → Megatron 或 TorchTitan 训练 → 分布式检查点及恢复 → 权重转换 → 独立评测。除峰值吞吐外，还需观察有效 token/s、集群利用率、故障恢复时间与最终任务效果。训练框架和存储、调度系统都属于完整方案的一部分。

**Agent 在线 RL。** 任务分发 → 采样引擎生成轨迹 → 隔离环境执行工具 → 验证器产生奖励 → 训练框架更新 → 同步权重 → 独立测试。需检查轨迹和 token 对齐、奖励漏洞、环境重置、训练和推理资源分配；vLLM 官方文档描述它与 RL 库组合用于生成。

## 评价成熟度的证据标准

1. 官方明确支持的模型、硬件、方法与版本，而不是仅有论文或仓库名称。
2. 可复现的安装、固定配置、检查点恢复和导出；验证结果区分项目方和本书实测。
3. 公开生产案例应提供工作负载、规模、时间、工程细节和可核查出处；厂商宣传不自动等于独立验证。
4. 维护与升级成本：版本发布、API 变化、依赖冲突、模型适配及故障诊断。
5. 在目标工作负载上比较任务质量、有效 token/s、峰值显存、故障恢复和全生命周期成本。

本章是第一轮方案地图，**不是完整工业采用调查，也没有实测排名**。待补充商业托管平台、调度与实验追踪、数据版本、评估栈、实际组织案例和逐项版本兼容表。优先选择一个固定模型和数据集，运行并记录单卡对照后再下性能结论。
