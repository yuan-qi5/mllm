# Note : Qwen2.5-Omni Technical Report
> 由于刚接触 mllm ,会记录更多的基础知识,以引用方式区分论文部分与基础知识

## Introduction

### What is Qwen2.5-Omni ?
端到端多模态(text,image,audio,video)大模型,可同时流式产生 text 和 natural speech
> streaming manner ：模型一边生成内容，一边将结果逐步输出，与普通生成相对(non-straming) <br>
> end-to-end : 直接从输入到输出，中间所有步骤都由一个整体模型自动完成，无需手工干预(eg:特征工程)或中间处理步骤

### What challenges does Omni model face ?   
- 多种模态的联合训练 (尤其对于 video 需要 audio 和 visual 对齐)
- 控制不同种模态在训练时的相互干扰
- 减少 audio output streaming 延迟，通过探索模型架构 


### What innovations are Qwen2.5-Omin ?
- **T**ime-aligned **M**ultimodal **RoPE** **TMRoPE** position embedding
- **Thinker-Talker** architecture
- **block-wise processing** in audio and visual encoder
- **sliding-window** DiT decoding audio

## main

### Thinker-Talker architecture
Thinker (brain) : 处理和理解多模态输入，生成高级表示和对应的文本   
  - a Transformer decoder <br>
  
Talker (mouth): 流式接受来自 Thinker 的输出并且输出 speech token
  - a dual-track autoregressive Transformer decoder

> **dual-track model**(双轨模型) : 模型在生成内容时**并行建模两个序列**(或模态/轨道)，这两个序列可能存在依赖或交互，从而提升整体建模能力
>

### Perceivation
**tokenize text** : Qwen's tokenizer  
  - 应用包含 151,643 regular token词汇表的**字节级 BPE 编码**
> **BPE (Byte-Pair Encoding)** : 一种字词分词算法，将词语分为更小的单位，用频率更高的字串合并来减少词表大小，同时覆盖未登录词(unknown words) <br>
> **Byte-level BPE** : 用**字节**而不是**字符**来进行 BPE 操作

**audio input & audio from video** : Qwen2-Audio encoder
  - 先重采样至 16kHZ 的频率，再将原始波形转换为具有 128 通道的 Mel频谱图，所使用窗口大小为 25ms,跳帧(步长)为 10ms
  - 
> **采样率 (sampling rate)** : 单位时间内从连续音频信号中采集样本的次数，通常单位为 赫兹 HZ 或 kHZ(千赫) eg : 16kHZ 采样率指每秒采集16,000个样本点  <br> 
> **原始音频波形 (raw waveform)** : 采样后得到的一维时间序列，是一长串浮点数/整数，每个值表示声波再某一时间点的振幅大小 <br>
> **Mel 频谱图 (Mel-Spectrogram)** : 将音频波形转化为"时间 * 频率形式的二维图像"，其中频率轴式按人耳听觉感知方式(Mel标度)非线性划分的<br>
> **window** : 窗口大小(window size)为每次分析的时间长度, 步长(hop size)为每次窗口移动的时间长度 

**vision encoder** : Qwen2.5-VL
  - 基于 ViT , 大约有 67.5亿参数，可同时处理 image 和 video
  - 采用融合图像与视频数据的混合训练方案，以增强理解
  - 采用动态帧率对视频采样同时对齐音频采样
  - 将每张图片当作两帧完全相同的帧来处理

> 动态帧率 (dynamic frame rate) : 在对视频进行帧抽样时，所用帧率不是固定，而是根据默写外部因素动态调整每秒取多少帧

## todo list
- Mini-Omni
