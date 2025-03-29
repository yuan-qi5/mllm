# Note : Qwen2.5-Omni Technical Report
> 由于刚接触 mllm ,会记录更多的基础知识,以引用方式区分论文部分与基础知识

## Introduction

### What is Qwen2.5-Omni ?
端到端多模态(text,image,audio,video)大模型,可同时流式产生 text 和 natural speech
> streaming manner ：模型一边生成内容，一边将结果逐步输出，与普通生成相对(non-straming) <br>
> end-to-end : 

### What challenges does Omni model face ?   
- 多种模态的联合训练 (尤其对于 video 需要 audio 和 visual 对齐)
- 控制不同种模态在训练时的相互干扰
- 探索模型架构为有效地 audio output streaming ,为减少延迟


### What innovations are Qwen2.5-Omin ?
- **T**ime-aligned **M**ultimodal **RoPE** **TMRoPE** position embedding
- **Thinker-Talker** architecture
- **block-wise processing** in audio and visual encoder
- **sliding-window** DiT decoding audio

## main

### Thinker-Talker architecture

> dual-track







