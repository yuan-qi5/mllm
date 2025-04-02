# Note DeepSeek-VL2：Mixture-of-Experts Vision-Language Models for Advanced Multimodal Understanding 

## Introduction

### What advancements are DeepSeek-VL2 ?

- a dynamic, high-resolution vision encoding strategy

  - a dynamic tiling vision encoding , 类似 slicing-tile methods , 高分辨率图像分块，再通过共享参数的 ViT 处理 与 DeepSeek-VL 的混合视觉编码（以两种固定分辨率 384 * 384，1024 * 1024）相比，可处理不同纵横比的高分辨率图像
    
- an optimized language model architecture

  - 利用 Multi-head Latenet Attention(MLA) 机制，通过压缩 KV cache 极大减少计算消耗
 
  - 使用 DeepSeekMoE framework 提高效率，采用三种 MoE 变体，3B、16B、27B，对应 0.57B、2.4B、4.1B 的激活参数
    
- a refined vision-language construction pipeline

  - 改进后数据集新增视觉接地和图形用户界面感知功能
    
![DeepSeek-VL2_overview]()


> llava-style architecture :
>
> 

## Model Architecture

DeepSeek-VL2's core components :

- a vision encoder

- a vision-language adaptor

- a Mixture-of-Experts language model

### Dynamic Tiling Strategy

pre-trained SigLIP-SO400M-384 要求输入图像尺寸为 384 *384

处理方式 :

- 定义一个候选尺寸集 CR，将图像分为多个 $384 \times 384$ 的小 "tile"
 
CR = { (m × 384, n × 384) | m ∈ ℕ, n ∈ ℕ, 1 ≤ m, n, mn ≤ 9 }

- 先调整长边与目标分辨率一致（保持宽高比），再填充另一边，以填充区域最小来选择切割方式

- 将切割后的 tile 以及全局缩略图(global thumbnail tile)输入视觉编码器处理（每个 tile 输出 $27 \times 27$ 个 token）

- 如果一次输入多张图，不再使用动态 tile 策略，

![DeepSeek-VL2_dynamic_tiling_strategy_illustration]()  

### Vision-Language Adaptor

- 先使用 $2 \times 2$ pixel shuffle 将 $27 \times 27$ 压缩为 $14 \times 14$ tokens

- 引入三个特殊 token，在全局缩略图每行后加一个 <tile_newline>，将 $m_i \times n_i$ 局部 tile 重排成形状为( $m_i \times 14$, $n_i \times 14$ ) 2D 网格，在最后一列末尾加上 $m_i \times 14$个 <title_newline>，展平后再在全局缩率图与 local tiles 间插入一个 <view_separatir>

- 使用两层 MLP 将其投影到语言模型的嵌入空间

> pixel shuffle (像素重排操作) ：
>

### DeepSeekMoE LLM

- 利用 Multi-head Latenet Attention(MLA) 机制，通过压缩 KV cache 极大减少计算消耗
 
- 使用 DeepSeekMoE framework 提高效率，在 MoE 训练阶段，为每位专家引入全局偏差以解决负载平衡

> load imbalance : 是 MoE 中的一个经典问题，在 MoE 中，每个样本由一个 “门控网络” 决定走哪个 expert, 有时会出现部分 expert 经常被使用，部分 expert 几乎闲置，这会导致训练不充分、容易过拟合以及计算资源浪费等问题


![DeepSeek_architectural_configuration]()

## Data Construction



## Training Methodology



## Evaluation 





