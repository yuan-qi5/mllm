# Note :  Qwen2.5 Technical Report

## Induction 

Qwen2.5 有 7 个不同版本 ：0.5B，1.5B，3B，7B，14B，32B，72B，不仅提供 bfloat16 精度的原始模型，还提供不同精度的量化模型。

### What improvations are Qwen2.5 ?

- pre-training stages : 数据集规模从 7 万亿 tokens 到 18 万亿 tokens 

- post-training stages : 超过 1 百万样本进行监督微调和多阶段强化学习（包括离线学习 DPO 和在线学习 GRPO）

### What key features of Qwen2.5 ?

- 更多样化尺寸

- 更多训练数据

- 更好使用 ：生成长度从 2K tokens 到 8K tokens，更好的支持结构化输入和输出（如表格和 JSON ）

> quantized（量化）: 一种压缩和加速模型推理的技术，将模型中连续的、精度较高的浮点数（如 FP32）转换为低精度表示（如 INT8、INT4 等），从而减少存储空间与计算负担
> 
>
> proprietary model（专有模型）：指非开源模型。
>
> mixture-of-experts（MoE，专家混合模型）：一种深度学习模型架构，用于解决模型的能力平静问题。核心思想是将单个大模型分解为多个更小、更专业的 “专家”，每个专家只负责处理特定的子任务或特定的输入数据类型。
>
> MMLU(Massive Multi-task Language Understanding) ：测评多任务语言理解能力 + 专业知识广度
>
> BBH(big-Bench Hard) ：从 Big-Bench （谷歌提出的大型语言模型评测平台）中选出的 hard 子集，评测模型的复杂推理 + “边缘、冷门、未见任务”泛化能力 + 类人智能灵活性
>
> MBPP(Mostly Basic Python Programming) ：测评 Python 编程能力
> 

![Qwen_series_development]()

强调了数据规模的作用。

## Architecture 




## Pre-training


## Post-training 


## Evaluation

