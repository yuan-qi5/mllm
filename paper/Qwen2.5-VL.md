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

- 视觉编码器 ViT 采用 **window attention** 以优化推理速度，此外将 ViT 中 FFN 整合 SwiGLU 激活函数，采用 RMSNorm 正则化
- **动态 FPS 采样**，将动态分辨率扩展到时间维度（即动态帧），支持对不同采样率的全面视频理解
- 将时间域上 MRoPE 与绝对时间对齐，促进复杂地时间序列学习 

![Qwen2.5-VL framework]()  

> FPS (Frame Per Second) : 每秒中采样多少帧图像
> dynamiac FPS Sampling ； 根绝视频内容变化动态地调整帧采样频率
> 
## Model







## Pre-Training






## Post-Training








## Experiments

