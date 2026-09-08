# ChunkGatedDeltaRuleFwd 编译期类型分发设计

方案设计规则版本：V1。

本章补充既有 arch22/arch35 实现的编译实例选择设计，不重新设计其数学 Stage、调度或存储。
既有接口与支持范围见 [README](../README.md)，数学标杆为仓内 ATK 的 `gdn_reference.py`。
修改前基线为 `3baf8f646535f9633e0b1d2990b1edc8ad0e5bac`。该基线已经把 A5 H
收窄为四条可达类型路径；本次保留这一实现和最新同步/流水修改，并继续消除外层 InputT
重复分支、固定 O 类型，同时补齐 arch22 的 H/O 收窄。

## 1. 目标

使四组外部 dtype 变体中的每次编译只展开当前 Q/K/V 类型及 FP32 gate 的 H/O 实现，
降低 megakernel 编译工作量。六个公开单算子不变，内核优化级别、计算顺序与同步不变。
不以牺牲运行性能、精度或支持组合换取编译速度。

调用链仍为 cumsum/KKT/SolveTri → recompute → H → O；A2/A3 与 A5 的私有实现隔离。

## 2. 范围与 shape

- A2/A3：arch22；A5：arch35。
- Q/K/V 为 FP16/BF16，K=128，V=128/256，chunkSize=64/128。
- 保留 fixed/varlen、MHA/GVA、tail、可选初始和最终状态以及 full/inference 辅助输出合同。
- OpDef 的四组 input/state 类型及两个 TilingKey 不变。

### 编译类型选择

| 维度 | 依据 | 本次处理 |
| --- | --- | --- |
| Q/K/V/A/O | OpDef 类型对齐与生成宏 `DTYPE_Q` | 通过 `InputT` 传至 Phase6/H/O |
| H/O 的 g | L2/tiling 已要求 FP32，内部 cumsum 为 FP32 | 固定 `float` |
| H 的 initial/final state | `hTiling->stateDataType` | 保留 FP32／InputT 两路 |
| H 的 kGated | `hTiling->useGk` | 保留 true/false 两路 |
| V 对应 TileShapes | 原 Key 1/2 | 不变 |

不使用 `DTYPE_FINAL_STATE` 特化 StateT：L2 在不返回最终状态时创建 FP32 占位 tensor，
它不一定等于真实初始状态 dtype。可选输入的 dtype 宏也不作为本轮新的分发前提。

## 3. 算子接口

公开 ACLNN、L0、Python 返回顺序、默认值、可选输出、参数校验、OpDef 和 tiling 序列化均不修改。
beta 在 L2 转成 FP32 的既有行为不变；内部和外部 h/vNew 的存储精度不变。

## 4. 既有 Stage 数学语义与分发

各计算阶段仍使用原实现：

1. cumsum 与 KKT 生成三角系统系数，SolveTri 生成 A；保留各架构原先的有效区、cast 和尾块路径。
2. recompute 使用 A、K、V、gate、beta 生成 W/U。
3. H 按序传播 chunk 状态、生成 h/vNew，使用原 `RunFwdH` 的完整模板开关。
4. O 从 q/k/h/vNew/g 生成输出，使用原 `RunFwdO` 的完整模板开关。

本次仅把“从 tiling 重新读取 input/g dtype”替换为“从编译入口传递 InputT、固定 float gate”。
stateDataType/useGk 四种合法路径映射到修改前同一个 `RunFwdH` 实例；O 映射到
修改前 FP32 gate 的同一个 `RunFwdO<InputT, float>` 实例。

arch22 的 H 模式保持 `kGated,true,false,true`；arch35 按最新基线在 A5 编译时保持
`kGated,true,false,true`，非 A5 兼容分支仍为 `false`。
不得合并两者以减少文件或模板数量。

开发参考：`CHUNK_DEPENDENT` V1，来源 commit `a14961e9dfe8c8264768d67fb447101873da4ecf`。
这里只采用类型从 host/生成宏进入模板的组织方法，不采用参考中的其他 tile、slot 或同步参数。

## 5. 资源与验证

本轮不改变任何 Stage 的 L1/UB/L0 地址、容量、份数、workspace offset、数据生命周期、
跨核参与者或 event/flag。所有初始化、等待、发布与末次消费调用按原顺序保留。
R01–R19 对本轮差异的检查结果是“未改变 Stage 与资源方案”；这不等于为历史实现补做了
完整资源审计，也不宣称历史实现自动符合新规则的每项设计要求。

验证包括：

- 编译入口与真实 helper 分发合同，state 占位边界，架构模式和同步序列不变。
- A2/A5 同工具链、同 jobs 的基线／候选冷编译，单独记录 core 与总构建墙钟及产物代码段。
- 用户指定 ATK 矩阵在隔离副本执行，不修改原矩阵、阈值、CPU 标杆或原 ATK 目录。
- 全部受影响 Key、FP16/BF16、状态存在／缺失、V128/V256、chunk64/128、tail 和 varlen。
- 确定性、边界和用户矩阵精度回归；不以本地结构测试替代真机精度。本轮目标是编译耗时，
  不把 event 或 msprof 运行时间混入编译收益。

预期是模板工作和代码规模减少。准确编译收益及精度放行结果在硬件测试完成后补录；
出现新精度失败或运行回退时停止放行，保留原始证据并定位，不放宽比较标准。
