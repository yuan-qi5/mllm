# Deep Double Descent: Where Bigger Models And More Data Hurt

## Introduction

经典的统计学习理论中 *bias-variance trade-off* 认为模型复杂度越高，偏差越低但方差会越高。 根据该理论，一旦模型复杂度超过一定阈值，模型就会 “过拟合”，由方差项主导测试误差。因此从这点之后，增加模型复杂度枝会降低性能。

> 偏差（bias）：用于简化模型来近似真实世界问题（可能很复杂）引入的误差。高偏差意味着模型对数据做出了强烈的假设，导致欠拟合，即模型无法捕捉数据中的底层模式。
>
> 方差（variance）：模型对训练数据中小变化的敏感性导致的误差。高方差意味着模型将训练集中的噪声或随机波动视为真实模式，导致过拟合，即模型在训练数据上表现好，但在新数据上表现差。

因此经典统计学中传统观点是，“once we pass a certrain threshole, larger models are worse”。

而神经网络具有数百万个参数，甚至足以适应随机标签，但它们在许多任务上的表现要比较小的模型好得很多。相关人员的观点是 “larger models are better”。

神经网络中训练时间对测试性能的影响存在争议，在某些情况下，“提前停止” 可以提高测试性能。而在其他情况下，将神经网络训练到零训练误差可以提高性能。但经典统计学家和深度学习实践者都同意 “more data is always better”。

**main contributions**：
- 双重下降（double descent）在不同任务、模型架构及优化方式下是一种稳健的现象。

- 提出用训练中的 **EMC**(*effective model complexity*) 来替代参数量，即给定一个数据分布和小误差阈值，EMC 是在给定训练过程能使平均训练误差小于误差阈值的最大样本数。因为 EMC 不仅取决于数据分布、分类器架构，还取决于训练过程，尤其是训练时间增加会增加 EMC。

- 我们假设对许多模型和学习算法，双重下降是 EMC 的函数。若固定模型，增加训练时间，会观察到性能遵循 "epoch-wise double descent"， 








TODO:
Understanding deep learning requires rethinking generalization.：神经网络足以适应随机标签。
