# 07 NeMo：贯穿生命周期的模块化工具链

NeMo 不只是一个训练算法或数据清洗工具。NVIDIA 将 NeMo 组织为覆盖数据、训练、后训练、评估、运行和部署的多个组件；并非每个项目都需要全部使用。原 NeMo 2.0 仓库逐步侧重语音，其他能力拆分到独立项目，阅读旧教程时尤其要核对版本和组件名称。[NeMo 总览](https://docs.nvidia.com/nemo/)；[项目组织说明](https://github.com/NVIDIA-NeMo)。本章依据官方文档整理，检索于 2026-10-29，尚未实测。

| 阶段 | 组件 | 用途与边界 |
| --- | --- | --- |
| 数据 | [Curator](https://docs.nvidia.com/nemo/curator/home/welcome) | 多模态数据抽取、过滤、去重和处理；输出需记录数据及规则版本 |
| 预训练与微调 | [AutoModel](https://docs.nvidia.com/nemo/automodel/latest/get-started/repo-structure) | PyTorch 原生预训练和微调，支持 Hugging Face 生态与分布式策略 |
| 大规模预训练与微调 | [Megatron Bridge](https://docs.nvidia.com/nemo/megatron-bridge/latest/) | 基于 Megatron Core，提供预训练、SFT、LoRA 和受支持模型的权重转换 |
| 蒸馏及量化感知训练 | [AutoModel 的训练能力](https://docs.nvidia.com/nemo/automodel/get-started/key-features)、[ModelOpt](https://nvidia.github.io/Model-Optimizer/guides/4_distillation.html) | 提供相应训练或优化路径；模型与方法的具体支持按版本核对 |
| 在线 RL | [NeMo RL](https://docs.nvidia.com/nemo/rl/latest/index.html) | 管理后训练算法与训练后端，配合生成引擎和环境 |
| Agent 环境 | [NeMo Gym](https://docs.nvidia.com/nemo/gym/main/about/ecosystem/) | 组织数据、agent harness 与 verifier，驱动评估或训练；环境质量与奖励设计仍需自行验证 |
| 评估 | [Evaluator](https://github.com/NVIDIA-NeMo/Evaluator) | 模型和任务评测的执行与结果管理；应保留独立测试集 |
| 作业运行 | [NeMo Run](https://docs.nvidia.com/nemo/) | 配置、运行和追踪本地或集群训练/评估任务 |
| 合成数据与技能 | [NeMo Skills](https://docs.nvidia.com/nemo/) | 合成数据生成、训练和评测参考流程 |
| 推理与产品 | [Export and Deploy](https://docs.nvidia.com/nemo/)、[Guardrails](https://docs.nvidia.com/nemo/guardrails/latest/index.html) | 前者面向导出和部署，后者在应用侧实施可编程约束；不能等同于训练时对齐 |
| 语音 | [NeMo Speech](https://docs.nvidia.com/nemo/) | 语音识别和语音生成等模型的训练与部署 |

## 它们怎样组合

一个代码 Agent 的可选路径：Curator 准备任务数据，AutoModel 或 Megatron Bridge 进行 SFT；NeMo Gym 提供任务、工具执行环境与 verifier，NeMo RL 协调后训练，生成后端完成 rollout；Evaluator 和独立的隐藏任务衡量效果，最后导出并在目标推理栈测试。并非要求所有组件同用；可以保留现有的 Hugging Face、verl、vLLM 或其他系统，只接入需要的一层。Gym 的官方说明将环境、harness、验证器与 RL 训练器区分开来。[Gym 生态](https://docs.nvidia.com/nemo/gym/main/about/ecosystem/)

AutoModel 与 Megatron Bridge 有覆盖重叠：前者以 PyTorch 和 Hugging Face 模型集成为主要路径，后者围绕 Megatron Core 并行与受支持模型的权重转换。二者不是以规模简单划定的上下级；应比较具体模型、并行策略、检查点、硬件与团队维护能力。[AutoModel 说明](https://docs.nvidia.com/nemo/automodel/latest/get-started/repo-structure)；[Bridge 说明](https://docs.nvidia.com/nemo/megatron-bridge/latest/)。

## 实践前核对

- 固定每个组件、容器、驱动和 CUDA 版本，核对模型架构、权重格式和检查点转换是否支持。
- 明确单卡或集群资源、网络拓扑、样本与 token 计数、恢复策略、训练与 rollout 权重同步方式。
- 对训练、评估和部署分别建立基线；开源库、托管产品与特定商业容器的功能和许可不能混为一谈。
- 记录质量、有效 token/s、峰值显存、恢复时间和总成本。NVIDIA 文档对支持能力的描述不等于本书已经验证性能或所有模型都兼容。
