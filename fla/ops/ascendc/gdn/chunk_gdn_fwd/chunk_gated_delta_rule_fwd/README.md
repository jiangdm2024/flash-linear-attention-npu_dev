# ChunkGatedDeltaRuleFwd

## 功能

`ChunkGatedDeltaRuleFwd` 实现 Gated Delta Rule 的分块前向计算。A5 上 `useExp2=true`
且 `useQkL2norm=true`
时依次调度 `ChunkGatedDeltaRuleFwdPrepare`、`ChunkGatedDeltaRuleFwdH` 和 `ChunkFwdO`；
其他组合继续使用原 Phase6 kernel。当前实现支持定长和变长序列、GVA、可选初始状态
以及可选最终状态输出。

用于精度对比的公开算子链依次由以下算子组成：

1. `ChunkLocalCumsum`（`chunk_local_cumsum`）；
2. `ChunkScaledDotKkt`（`chunk_scaled_dot_kkt`）；
3. `SolveTri`（`solve_tri`）；
4. `RecomputeWUFwd`（`recompute_w_u_fwd`）；
5. `ChunkFwdH`（`chunk_fwd_h`，固定 `use_exp2=true`）；
6. `ChunkFwdO`（`chunk_fwd_o`）。

融合 kernel 内部实现上述等价计算阶段，不调用或链接这些公开算子的 ACLNN 实现。
公开算子链只作为 ATK 精度标杆使用。

## 输入

令 `B` 为物理 batch size，`Hk` 为 q/k 头数，`Hv` 为 v 头数，`T` 为 token 数，
`K=128`，`V` 为 value 维度，`N` 为逻辑序列数。

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `q` | 必选 | 由 `layout` 决定；FP16/BF16 | Query |
| `k` | 必选 | 与 q 同 shape/dtype | Key |
| `v` | 必选 | 由 `layout` 决定；与 q 同 dtype | Value |
| `g` | 必选 | `[B,T,Hv]`；FP32 | 门控值，固定为 sequence-major |
| `beta` | 必选 | 与 g 同 shape；FP32 或与 q 同 dtype | Delta 系数 |
| `aLogOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `dtBiasOptional` | 当前未支持 | - | 扩展接口预留，必须为空 |
| `initialStateOptional` | 可选 | `stateVFirst=false` 时为 `[N,Hv,K,V]`，否则为 `[N,Hv,V,K]`；FP32 或与 q 同 dtype | 初始状态 |
| `cuSeqlensOptional` | 可选 | `[N+1]`；INT64 | 变长序列累计长度，需与 `chunkIndicesOptional` 同时提供 |
| `chunkIndicesOptional` | 可选 | `[2*Nc]`；INT64 | canonical sequence-major chunk 索引 |

四种 QKV 布局均使用四维输入：`BNSD/NTD=[B,H,T,D]`，
`BSND/TND=[B,T,H,D]`。`Hv` 必须能被 `Hk` 整除。变长模式使用物理 `B=1`，`cuSeqlensOptional` 必须从 0 开始、
以 `T` 结束且单调不降。

## 输出

| 名称 | 必选性 | Shape/Dtype | 说明 |
| --- | --- | --- | --- |
| `oOut` | 必选 | `[B,T,Hv,V]`（BSND）；与 q 同 dtype | 前向输出固定使用 sequence-major 布局 |
| `finalStateOutOptional` | 可选 | 末两维由 `stateVFirst` 决定；与初始状态同 dtype，无初始状态时为 FP32 | 是否为空直接决定是否计算并输出最终状态 |
| `gCumsumOutOptional` | 可选 | 与 g 同 shape；FP32 | chunk 内门控累加结果；为空时使用内部临时张量 |
| `aOutOptional` | 可选 | `[B,Hv,T,chunkSize]`；与 q 同 dtype | 系数矩阵；为空时使用内部临时张量 |
| `qHatOutOptional`、`kHatOutOptional` | A5 `useExp2=true` 可选 | 与 q/k 相同 | L2Norm 结果 |
| `qRstdOutOptional`、`kRstdOutOptional` | A5 `useExp2=true` 可选 | q/k layout 去掉最后一维；FP32 | L2Norm rstd |
| `betaEffOutOptional` | A5 `useExp2=true` 可选 | 与 beta 同 shape；FP32 | 非空时启用并输出 beta sigmoid |
| `hOutOptional` | A5 `useExp2=true` 可选 | `stateVFirst=false` 时末两维为 `[K,V]`，否则为 `[V,K]`；与 q 同 dtype | 分块状态 |

Python ctypes 入口通过 `disable_recompute=False` 选择训练输出，返回 `gCumsum` 和 `A`；
设为 `True` 选择推理输出，仍返回四元组，但后两项为 `None`，底层公共输出指针也为空。
该参数统一替代原 `output_a` 和 `return_aux`，默认值为 `False`；不再接受这两个旧关键字。
此开关控制辅助输出保存，不跳过前向计算所需的内部 recompute 阶段。

## 属性

| 名称 | 当前支持范围 | 说明 |
| --- | --- | --- |
| `layout` | A5 `useExp2=true` 支持 `BNSD/BSND/NTD/TND`；其他路径保持原有支持范围 | q/k/v 的输入布局；BSND/TND 输入在拼接路径内转为 head-first，o 固定输出 BSND |
| `scale` | 通常为 `K**-0.5` | Query 缩放因子 |
| `chunkSize` | `64`、`128` | 分块大小 |
| `useExp2` | A5 支持 `true`；其他路径为 `false` | A5 true 时走三小算子拼接路径 |
| `useQkL2norm` | `true/false` | 仅与 `useExp2=true` 同时成立时走三小算子拼接路径；false 时走原 Phase6 kernel |
| `allowNegEigval` | A5 `useExp2=true` 支持 | 为 true 时必须提供 `betaEffOutOptional` |
| `stateVFirst` | A5 三小算子拼接路径支持 `true/false` | 控制初始状态、分块状态和最终状态的末两维采用 `[V,K]` 或 `[K,V]` |

当前未实现的扩展组合会返回参数错误，不会静默忽略。

## 支持范围

- A2（`ascend910b`）、A3（`ascend910_93`）、A5（`ascend950`）。
- 原 Phase6 路径支持 FP16、BF16，`K=128`、`V=128/256`、`chunkSize=64/128`。
- A5 `useExp2=true` 路径支持 BF16、`K=V=128`、`chunkSize=64`、`Hv/Hk in {1,2,3,4}`。
- 支持 MHA、GVA、定长和变长序列。
- A2/A3 使用 `arch22` 私有实现，A5 使用 `arch35` 私有实现；两套架构代码隔离维护。

## 验证

### A5 同步与任务分配

A5 保留 cumsum 到系数生成的全核发布、KKT 的全 AIV 会合、Solve 到 recompute 的阶段私有
全 AIC 会合，以及 H 到 O 的 MTE3/MTE2 全核交接。H/O/Solve 的局部交接使用原有 mode2
完成标志；融合内部 H 的 partial MMAD 按实际尾块形状处理。独立 H 的修改不包含在本 PR 中，
其尾块清零问题另交开发责任人确认。

融合 H 在定长序列上使用 `kChunkPipeline=true` 的 balanced-wave 任务分配，变长序列沿用
原有序列调度。公共 O 头文件和 H 调度入口的相关选择由 A5 架构条件隔离，A2/A3 保留各自实现。

同步调整的回归范围包括冷启动、连续重复调用、多 task 和 slot 复用、短尾块、变长多序列、
64 个 H 任务、FP16/BF16、V128/V256，以及独立 H 的 `gk` 可选入口。精度标杆、输入数值范围
和比较标准沿用既有 ATK 契约；同步的必要性依据硬件消融和精度回归验证。

### 测试入口

ATK 用例和执行说明位于
[`tests/atk/chunk_gated_delta_rule_fwd`](../../../../../../tests/atk/chunk_gated_delta_rule_fwd/README.md)。
精度双标杆分别使用融合算子、上述公开算子链和 FP64 recurrence，并在相同冻结输入上比较结果。

ACLNN ABI 合同可通过以下命令检查：

```bash
python3 tests/atk/chunk_gated_delta_rule_fwd/aclnn_abi_contract.py
```
