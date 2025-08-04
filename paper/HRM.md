# Hierarchical Reasoning Model

## Induction

**CoT** 面临问题：CoT 通过将复杂任务分解为更简单的中间步骤，使用浅模型依次生成文本，CoT 将推理拓展为 token-level 语言。因此它依赖于脆弱的、人为定义的分解，其中一个步骤的错误或步骤的错误顺序会完全破坏推理的过程。

仿照人脑的层级和多时间尺度的结构，提出 HRM(Hierarchical Reasoning Model)。HRM 涉及去提高有效计算深度（*effective computational depth*） 。

HRM 具有两个耦合的循环模块，一个高级（**H**）模块用于抽象、深思熟虑的推理，一个低级（**L**）模块用于快速、详细的计算。通过层次收敛来避免了标准循环模型的快速收敛。即只有在快速更新的 L-module 完成多个计算步骤并达到局部平衡后，缓慢更新的 H-module 才会前进，此时 L-module 会被重置一开始新的计算阶段。

此外提出一种用于训练 HRM 的一步梯度近似方法，该方法提高了效率并消除了对 BPTT 的要求。这种涉及在整个反向传播过程中保持恒定的内存占用（与 **BPTT** 的 **O(T)** 相比，*T* 时间步占用为 **O(1)**），使其更具有可扩展性。

仅使用 1,000 input-output 样本，没有预训练或 CoT-supercision，HRM 在 *Sudoku-Extreme Full* 实现了近乎完美准确率和 $30 \times 30$ 迷宫中实现了最佳寻路。在 Abstraction and Reasoning Corpus(ARC) AGI Challenge 中仅使用官方数据集（~ 1000 个示例）从头开始训练，只有 27M 个参数和 $30 \times 30$ 个网格上下文，实现了 40.3% 性能。

## Hierarchical Reasoning Model











## Results







