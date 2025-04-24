# VideoEspresso : A Large-Scale Chain-of-Thought Dataset for Fine-Grained  Video Reasoning via Core Frame Selection

## Induction

### What is primary limiataion in video question-answering (VideoQA) research ?

- the scarcity of high-quality, large-scale datasets

### What is the limitation of existing methods ?

- manual annotation : costly for limiting scalability and lack the granularity needed for detailed understanding,

- using LLMs :

   - use video metadata (high-level descriptions) to leverage LLMs togenerate QA pairs  : missing of crucial video details for finegrained reasoning
 
   - analyzing every video frames : granular understanding is feasible, but video content is too redundant and keyinformation dispersed sparesely

### What is VideoEspresso's pipeline ?

- **extract key information from viedo** : design a semantic-aware key information extraction method

  - firstly map video frames to linguistic space using an LVLM

  - secondly remove similar frames based on semantic similarity to reduce redundancy in video data
 
  - thirdly group video frames to retain frame-level details and inter-frame correlation
 
  - lastly instruct GPT-4o to generate initial QA pairs and filter out low-quality data with designed prompts

- **introduce video Chain-ofThought annotations**

  - use GPT-4o to extract logical relationship evidence from QA pairs and videos including interations of key objects in spatial and temporal flow
 
### What are the main contributions ?

- a evaluation benchmark based on dataset including a set of GPT-4o-based open-ended evaluation metrics

- a novel framework, Hybrid LVLMs Collaboration  

## Dataset

### Video Data Curation 

- collect raw video from 7 dataset , 包含丰富的时间动态、逻辑序列和因果关系，且包含多种 video type

- predefine 14 tasks to assess model's capability

### Redundancy Removal in Videl Frames

- 动态决定采样间隔基于视频类型，因为不同视频表现出不同的内容和场景变化率

- 使用 InternVL2-8B 生成 frame-level caption 对每一个采样帧

- 使用语言检索模型 BGE-M3 去过滤掉冗余帧，通过计算 frame descriptions 之间的的余弦相似度，采用 LIFO 策略

![]()

> LIFO (Last-In-First-Out) : 后进先出与 FIFO 先进先出形成对比

### Question-Answer Pair Construction

- 每 15 连续帧字幕被分为一组，以保持语义连续性和避免一次输入过多 token

- 采用预定义指令提示 GPT-4o 在多帧描述基础上生成 QA pair 

- 使用额外的 LLM 去验证 QA pair 的质量，eg: 幻觉，准确性，过滤过于开放问题答案

### Multimodal Chain-of-Thought Annotation

focusing on annotating multimodal evidence that contains key spatio-temporal information 

- 将之前得到的 frame sequences with Q-A pairs 作为输入

- 使用提示指导 GPT-4o 生成 (1) 与问题最接近的字幕 (2) 关键 objects from captions (3) 将这些 key objects 组织成自然语言描述

- 使用 GroundingDINO 标记关键物体的边界框 (bounding boxes)进行空间注释，再使用 CLIP-ViT-B/32 验证边界框内 label 和 object 的一致性；由于 GPT-4o 生成的核心帧 caption 无法直接匹配原始 caption，使用 BGE-M3 去检索原始 caption set ，并获取 temporal grounding informaiton

### Data Analysis

**investiage temporal distribution of key information in video** : 测量不同任务中相邻关键帧之间的距离分布，发现不同任务距离不一样，表明传统统一采样策略是次优并会引入冗余

**compare with MVBench** : 

- 我们数据集在 QA set length 更长

- 在词云 (world cloud) 上的关键词更能反应推理

















