# Note Qwen2.5-VL Technical Report
 
## Introduction
Qwen2.5-VL 有三个版本 ： 72B, 7B, 3B

### What are current LVLMS's main drawbacks ?
- 像 “夹心饼干的中间层” ---- 即夹在视觉和语言大模型之间，什么都能做，但都做得不够好
- 细粒度的视觉任务构成了类比中的基础层

### What are Qwen2.5-VL's main advantages ?
- 全能文档解析能力 ：可处理多场景、多语言、多形式（如手写、表格、乐谱等）文档
- 跨格式精准物体定位 ：在 detecting objects, pointing to objects, counting objects 准确性得到提升，且支持绝对坐标与 JSON 格式输出 
- 超长视频理解与细粒度视频定位（在几秒钟内提取事件片段）
- 面向电脑和移动设备的增强性智能体功能

> detect objects : 识别图中有哪些目标
> point to objects : 用点/坐标/框等之处目标具体位置
> count objects : 数有多少个同类物体
> 上述三个任务构成了 "细粒度空间理解" 的核心
> 
> second-level event localization（二级事件定位）：通常指在"**秒级别**"的时间精度上，对视频中特定事件进行定位（localization 指第几秒开始第几秒结束）的任务

### What innovations are Qwen2.5-Omin ? 

- 视觉编码器 ViT 采用 **window attention** 以优化推理速度，此外将 ViT 中 FFN 整合 **SwiGLU 激活函数**，采用 **RMSNorm 正则化**
- **动态 FPS 采样**，将动态分辨率扩展到时间维度（即动态帧），支持对不同采样率的全面视频理解
- 将时间域上 **MRoPE 与绝对时间对齐**，促进复杂地时间序列学习 
 

> FPS (Frame Per Second) : 每秒中采样多少帧图像
>
> dynamiac FPS Sampling : 根绝视频内容变化动态地调整帧采样频率
>
> RMSNorm (Root Mean Square Normalization) : 层归一化的一种变体
>
> $$ LayerNorm(x) = \frac{x - \mu }{\sigma} * \gamma + \beta $$
>
> $$ RMSNorm(x) = \frac{x}{RMS(x)} * \gamma $$
>
> $$ RMS(x) = \sqrt{\sum_{i=1}^n x_{i}^{2}} $$
>
> SwiGLU 激活函数 ： 由 Gated Linear Units(GLU) 和 Swish 激活函数结合起来，常用于 Transformer 的前馈层 (Feedforward Layer)。相比原始 GLU 中使用的 sigmod，它更 “平滑” 、“非线性强”。 
>
> $$ GLU(x) = (xW_1) * \sigma(xW_2) $$
>
> $$ Swish(x) = x * \sigma(x) $$
>
> $$ SwishGLU(x) = (xW_1) * Swish(xW_2) $$
> 

![Qwen2.5-VL framework](./Qwen2.5-VL_framework.png) 

## Model

### Model Architecture

Qwen2.5-VL 整体上由三部分组成 ：
- Large Language Model : 使用 Qwen2.5 LLM 中预训练权重进行初始化，将 1D-RoPE 修改为与绝对时间对其的 MRoPE 

- Vision Encoder ：redesigned ViT ，在训练和推理时，都会把输入图像的高和宽调整为 28 的倍数，采用 141 * 14 patch

- MLP-based Vision-Languange Merger : 为提高长序列图像处理效率，


![Qwen2.5-VL_configuration](./Qwen2.5-VL_configuration)

## Pre-Training






## Post-Training








## Experiments

