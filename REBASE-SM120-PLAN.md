# SM120 Rebase 方案（rebase 到最新 main，取本地 ∪ 远程 的最优并集）

> 状态：**执行中**（GPU 已释放，可做 GPU 验证）。目标硬件：4× RTX PRO 6000 Blackwell（sm_120，TP4，PCIe 无 NVLink）。

---

## 0.5 进度日志（已完成）

分支 `sm120/rebase-main`（自 `origin/main` a1bf8ac12d），**19 个提交 / 57 文件 / +4706 −204**。
备份：tag `backup/pre-rebase-20260919` → `dc753a76fb`；未提交工作冻结在 `wip/pre-rebase-snapshot`（`fc1c473b7d`）。

**后续（2026-09-19）**：系列落地后又加了 5 个清理提交（`df4a09bcf4` → `78fa50bfe6`），
把新增的 36 个环境变量收敛到 8 个，详见 §12；备份 tag `backup/pre-cleanup-20260919`。

| 阶段 | 提交 | 内容 |
|---|---|---|
| **S1** | `406b7f7a25` | family 12 页几何（`64 × ratio` 使 `num_states` 恒 64）+ backend 声明 + `max(1, ratio)` 守卫 |
| **S2** | `02bbab31d4` | 捕获后 KV 安全（默认清 block 0，`VLLM_ZERO_ALL_KV_AFTER_CAPTURE` 全清）+ `KVBlockZeroer` teardown + top-k 越界上界 |
| **S3a** | `42ba508d0c` | decode/prefill 边界统一到 verify 宽度（上游无 issue 无 PR） |
| **S3c** | `f5e2aafc21` | V2 runner + 投机时走 wide eager region |
| **S3b** | `a9d9765ffc` | `#52499` overlay：SM12x sparse-MLA spec-decode 形状（4-D query / ≤64 段 / 空 chunk） |
| **S4a** | `aaa336ea85` | 调用方可提供 global top-k 缓冲（图内不再分配） |
| **S4b** | `bac7b9d1a8` | MXFP8 空 batch（`#57028` 的非几何部分） |
| **S4c** | `a842e7ce0e` | 多流 CUDA event 句柄按图隔离（保留图内重叠） |
| **S5a** | `d917c92c31` | Engram host 表按精确尺寸映射并注册（替代 `pin_memory` 幂次取整）+ warmup id 宽度 |
| **S5b** | `314ff839e9` | 预热 Engram row gather |
| **S6** | `b60dbb7274` | loader 跳过 checkpoint 中无对应参数的张量 |
| **S7** | `f51cc1c8b2` | GEMM 形状 / 集合通信 / 图节点标注埋点（全默认关） |
| **S8a** | `9eb1077ff1` | family 12 的 MXFP8 调优后端 + per-shape 策略（默认走上游） |
| **S8b** | `2735ce0e2c` | wo_a 分组投影小批快路径（SM120 实测数据在提交信息里） |
| **S8c** | `17e673ab4b` | mHC prenorm split 阶梯 + 可选融合 eager RMSNorm |
| **S8d** | `75a16e2849` | 只对 grouped 布局真正读取的行做 act-quant |
| **S10** | `24ec782996` | 大于主机内存的 checkpoint 加载（分片 + page-cache 驱逐 + cuFile 后端） |
| **S9/S11** | `156d89ff7f` | verify 宽度成本模型求解（默认关）+ 置信度头解耦 + 异步调度胶水 |

**被 DROP 的本地提交（有证据）**：
- `9b18c761fd`（engram 常量 grid）→ 上游已把 `GRID` 放进 `do_not_specialize`，不再按 token 数重编译，且不浪费 program；
- `5912d3cbce` 的 vision 加宽部分 → 已被合并的 **#57152**（`mm_prefix_clamp_sliding_window`，V4.1 恒不加宽）取代；
- `8e40aab9c9`（投机批按 FULL 捕获）→ 相对 main 净效果为零（继承值本就是 `UNIFORM_BATCH`）；
- `0fd525aeb7` / `6d74b9a0f2` / `09c74046c5` → 依赖未验证能力或本身是"默认关的取证工具"（见 §5.4/§5.5）；
- `#57388` / `#55692` → 他人仍在 review 的 open PR，不 vendor。

**未纳入（与你原方案一致）**：H1 breakable-serves-FULL 三连（自测仍 degenerate）——留在 `wip/pre-rebase-snapshot`。

全量验收（GPU 默认模式）：
```
pytest -q tests/v1/attention/test_deepseek_v41_block_size.py \
          tests/v1/attention/test_attention_splitting.py \
          tests/v1/worker/test_utils.py \
          tests/models/deepseek_v4_1/test_global_topk_bounds.py
-> 109 passed
```

**执行中发现、需要决策的两处事实**

1. `tests/v1/worker/test_utils.py` 的 4 个 HiSparse 用例在 `CUDA_VISIBLE_DEVICES=""` 下**必然失败**（需要真实 CUDA），与本次改动无关 → 该文件必须在放开 GPU 时跑；CPU 受限模式只用于被 mock 的用例（§8 已更新）。
2. **`main` 已有 adaptive verification**（`v1/worker/gpu/spec_decode/adaptive_verification.py`：cost tables + `argmax(tokens/cost)` 定 draft budget），而本地未提交的 `v1/spec_decode/verify_width.py` 是**同一目标的另一套实现**（device 端不动点求解、逐请求宽度）。本地代码自己的注释写明：上游那条路要求注意力后端能在 device 上裁剪 query 长度，**family 12 没有这样的后端** → 见 §5.5 决策。

---

## 0. 硬约束

| 约束 | 落地规则 |
|---|---|
| 不额外引入非 sm120 优化 | 任何补丁不得新增 `is_device_capability_family(100)` / `sm_100` / tcgen05 / MegaMoE / Mega-Gate / Mega-mHC(main 版) / DeepSelect / gvr_2 / GEMM-RS / SP 依赖；被 SM100 门控的上游代码**已经拿到的不动，但不主动引入**（见 §7 排除清单） |
| patch 简洁正确 | 每个补丁：单一主题、可独立 revert、附 CPU 可跑的测试或静态不变量断言；无"默认开启的未验证开关" |
| 取最大并集 | 本地有的 + 上游 sm12x/通用有的，都要在；重叠处取实现更优的一侧（§5） |
| 只 CPU 自测 | Tier 0/1 全绿为合并门槛；Tier 2（GPU e2e）不在本次范围，另列验收清单 |

---

## 1. 现状（实测事实，可复现）

| 项 | 值 |
|---|---|
| 本地分支 | `dsv41-sm120-migration`(4688a08e) / `exp/dspark-cg`(9b18c761) / `fix/clear-kv-and-full-guard`(dc753a76) / `fix/decode-prefill-split`(6d74b9a0) |
| 共同 base | `bde4feb06`（2026-09-12） |
| 目标 main | `a1bf8ac12d`（2026-09-19），**领先 base 405 个 commit** |
| 本地独有 commit | **20 个**（另有 3 个 stash 伪提交，忽略） |
| 逐个 patch 对 main `git apply --check` | 仅 5/20 干净应用（`09c74046`、`64d65cf2`、`d6159f75`、`0fd525ae`、`1b86a93f`） |
| 主要冲突源 | 包改名 `vllm/models/deepseek_v4_1/` → `deepseek_v41/`（上游 #56741 `cf1584f373`） |
| 未提交工作 | 27 文件 +1870/−243，另 6 个新文件（`vllm/v1/spec_decode/verify_width.py`、`vllm/profiler/gemm_shapes.py`、`graph_annotations.py`、`mxfp8/b12x_tuned.py`、`tests/v1/spec_decode/test_verify_width.py`、`V4.1-STRUCTURE-TREE.md`） |
| CPU 自测环境 | `~/anaconda3/envs/vllm-ds41/bin/python` = **指向本仓库的 editable 安装**（`vllm 0.1.1.dev59+gbde4feb06`），torch 2.13.0+cu130 在；pytest/ruff 通过 `--target /tmp/cpu-tools` 提供，**不改动用户 env** |
| 基线 | `tests/v1/attention/test_attention_splitting.py` → **30 passed / 12s**（当前 tip） |

**结论：不做字面 rebase，改为「在最新 main 上按补丁系列重放」。** 20 个 commit 里 15 个冲突、且包路径已变，字面 rebase 会产生一堆"改名噪声 commit"，无法评审也无法 revert。

---

## 2. 目标拓扑

```
origin/main (a1bf8ac12d)
        │
        └── sm120/rebase-main            ← 新分支，线性历史
              S1 geometry-sparse-mla-pages
              S2 kv-capture-safety            (null block + top-k bounds)
              S3 spec-decode-verify-width     (+ #52499 overlay + #57388 port)
              S4 cudagraph-multistream        (event pool + top-k buffers + mxfp8 empty batch)
              S5 engram-host-tables
              S6 checkpoint-loader
              S7 instrumentation (默认关)
              ── 可选/延后 ──
              H1 breakable-serves-full   (HOLD：自测仍 degenerate，见 §6)
              H2 forensic switches       (DSpark draft graph / compressor serialize)
```

规则：

- 每个 Sx 是**一个主题、可独立 revert**；Sx 内部若含"未验证默认开关"，一律改为 opt-in。
- 不产生"改名 commit"：代码直接写在 main 的 `deepseek_v41/` 路径下。
- main 已合并的 sm12x 相关优化**自动继承、不再重复实现**（§7 已合并列表）。

---

## 3. 逐 commit 决策（本地 20 个）

决策含义：**KEEP** 原样/重放 · **REWRITE** 取并集后重写 · **OVERLAY** 以上游未合并代码形式暂存 · **DROP-X** 删除（X=原因）。

| 本地 commit | 主题 | 决策 | 依据 |
|---|---|---|---|
| `cca1c51eba` | V4.1 缓存几何 ×SM12x 页 | **REWRITE** | 与 #56509/#57292/#57028 重叠；取本地算法 + 上游守卫/测试（§5.1） |
| `5912d3cbce`(a) | Engram 精确尺寸 pinned host 表 | **KEEP** | 上游无；社区 bitMacrocode patch 0001 同解；需按 main 的 engram 结构重放 |
| `5912d3cbce`(b) | Engram warmup id 张量宽度 IMA | **KEEP** | 上游无 |
| `5912d3cbce`(c) | V4.1 不做 vision 索引加宽 | **DROP-SUPERSEDED** | 已合并 **#57152**（新配置位 `mm_prefix_clamp_sliding_window`，V4.1 恒不加宽，并加 `assert builder.max_image_tokens == 0`） |
| `09499e40f2` | 预热 Engram row gather | **KEEP** | 上游无（main 无 engram warmup 模块） |
| `de085303dd` | 跳过 checkpoint 缺失参数 | **KEEP** | main 仍是 `param = params_dict[name]` 直取（两处），会 KeyError |
| `7b7c76df22` | top-k 索引上界检查 | **KEEP** | main 仍 `is_valid = local_idx >= 0`；与已合并 #57454（NaN 分数）**互补** |
| `1240340526` | cherry-pick #52499 | **OVERLAY** | #52499 仍 open 且 9/06 后无更新；SM12x spec-decode 形状修复不能丢，标记为"上游合并即删" |
| `9b18c761fd` | Engram gather 常量 grid | **KEEP** | main 仍 `grid = min(cdiv(rows,16), num_sms)` → 每个 token 数一次 JIT |
| `4688a08e78` | V2+投机时走 wide eager region | **REWRITE** | 只保留 wide-eager 分支；`UNIFORM_SINGLE_TOKEN_DECODE` 部分净效果为零（main 继承值本就是 `UNIFORM_BATCH`，`sparse_swa.py:411`） |
| `8e40aab9c9` | 投机验证批按 FULL 捕获 | **DROP-NETZERO** | 与 `4688a08e78` 第二部分互抵，相对 main 无净变化 |
| `1b86a93f01` | 清零被捕获污染的 KV block | **REWRITE** | 与 **#57158** 重叠；取上游形态（共享 helper + MRV1/MRV2 双调用 + 单测）+ 本地 profiling 生命周期安全；all-pages 变体走 opt-in（§5.2） |
| `dc753a76fb` / `2b8a46f1` | decode/prefill 边界统一到 verify 宽度 | **KEEP** | 上游**无 issue 无 PR**；main 仍 `1+2N` vs `1+N` 分歧；已含 CPU 回归测试 |
| `c7bf4a65e9` | 多流 CUDA event 句柄按图隔离 | **REWRITE** | 与 #57028 的"捕获即串行"重叠 → 取本地（保留重叠），并把 #57028 的 mxfp8 空 batch 修复一并纳入 |
| `6608f6813e` | 调用方提供 global top-k 缓冲 | **KEEP** | 上游无；避免分配落进图私有池 |
| `b061f6bcda`+`64d65cf2a2`+`d6159f7531` | breakable 捕获服务 FULL | **HOLD → H1** | 潜在收益最大，但本地自测记录 "output is still degenerate"（`6d74b9a0f2` 备注）→ 不进 v1 |
| `0fd525aeb7` | 修 `replay_for_batch_desc` 装饰器 | **DROP-DEP** | 只是 H1 自引入 bug 的补丁，脱离 H1 无意义 |
| `6d74b9a0f2` | compressor 串行化开关 | **HOLD → H2** | 默认关、自测结论"不是（唯一）缺陷"→ 取证工具，转 issue |
| `09c74046c5` | DSpark draft 图 opt-in | **HOLD → H2** | 负面结论（acceptance 1.00/0.0%）→ 转 issue（与已合并 #57432 同族） |

---

## 4. 未提交工作的分组（27 文件 + 6 新文件）

按内容分 5 组，**全部纳入并集**，但按组拆成独立提交、未验证的默认开关一律改 opt-in：

| 组 | 文件 | 归属 | 处理 |
|---|---|---|---|
| G1 spec-decode 宽度 | `v1/spec_decode/verify_width.py`(新)、`tests/v1/spec_decode/test_verify_width.py`(新)、`model_runner.py:2357`、`spec_decode/dspark|dflash/speculator.py` | sm12x 核心 | 与 `dc753a76fb` 合并进 **S3**；`backends/utils.py::spec_decode_verify_query_len` 与 `solve_verify_widths` 去重为单一真源 |
| G2 cudagraph/异步 | `v1/worker/gpu/model_runner.py`(+369)、`cudagraph_utils.py`(+37)、`async_utils.py`(新 +90)、`v1/core/sched/{scheduler,async_scheduler}.py`、`v1/outputs.py` | sm12x 核心 | 进 **S4**，按"捕获/零化/缓冲"与"异步调度"两个提交拆开 |
| G3 MXFP8 / o_proj 调优 | `mxfp8/{__init__ 相关,emulation,flashinfer}`, `mxfp8/b12x_tuned.py`(新 307 行)、`models/deepseek_v4/nvidia/ops/o_proj.py`(+206)、`layernorm.py`、`fp8_utils.py` | sm120（b12x = family 12） | 进 **S8**；`VLLM_MXFP8_*`、`VLLM_O_PROJ_*` 默认值改回 main 行为，性能开关 opt-in |
| G4 checkpoint/主机内存 | `model_loader/{weight_utils,default_loader}.py`、`envs.py` 中 `VLLM_SAFETENSORS_DROP_CACHE` / `VLLM_DEVICE_LOADER_MAX_TENSOR_BYTES` / `VLLM_INSTANTTENSOR_*` | 通用（sm120 部署 557 GB ckpt / 256 GB RAM 必需） | 进 **S6** |
| G5 instrumentation | `profiler/gemm_shapes.py`(新)、`profiler/graph_annotations.py`(新)、`_note_moe_padding`/`_note_moe_layout`/`_note_collective`/`note_gemm` 埋点、`envs.py` 中 `VLLM_GRAPH_KERNEL_ANNOTATIONS` / `VLLM_GEMM_SHAPE_LOG` | 诊断 | 进 **S7**，默认关；被 S8/mHC 引用的 import 要有 no-op 兜底 |

> 注：`fused_moe/deep_gemm_utils.py`(+137)、`experts/deep_gemm_moe.py`(+200)、`runner/moe_runner.py`(+35) 经查主要是埋点（`_note_moe_*`）＋ layout/padding 记账，不含 SM100 专属路径 → 归 G5/G3，需在 S7/S8 落地时逐 hunk 复核。

---

## 5. 重叠处取优（逐项结论）

### 5.1 缓存几何：本地算法 + 上游守卫与测试

| | 本地 `cca1c51eba` | #56509（open） | #57292（draft） |
|---|---|---|---|
| 页尺寸 | `block = max(PAGED_MQA_PAGE_SIZES) * ratio` → `num_states` 恒 64 | indexer 声明 `[64,128]`，压缩 spec 用 `block_size * max(1, ratio)` | 只声明 `[64]` |
| ratio-2 | 64-state 页 ✅ | 64-state 页 ✅ | 落到 32-state 页，作者自认 DeepGEMM SM120 仍拒绝 page32 ❌ |
| 是否 block split | 不 split（KV pool 不缩水） | 不 split | 需要 split 或直接报错 |
| 平台判定 | `family(120)`（`to_int()//10`，**已覆盖 sm121**） | `family(120) or family(121)`（冗余） | 同左 |
| ratio-0 安全 | ❌ 无 `max(1, ratio)` | ✅ | — |
| 显式报错 | 无 | 无 | ✅ `--block-size % 64 != 0` → `ValueError` |
| 测试 | 无 | 407 行（mock capability，**CPU 可跑**） | 94 行（同） |

**取优 = 本地算法 + 上游三件**：`max(1, self.compress_ratio)`（或 main 的 `get_num_kernel_states()`）、`--block-size` 非 64 倍数时早失败、把 #56509/#57292 的 mock 测试并成一份落到 `tests/v1/attention/test_deepseek_v41_block_size.py`。丢弃上游冗余的 `or family(121)` 与 try/except。

### 5.2 捕获后 KV 清零：上游形态 + 本地生命周期安全 + all-pages 走 opt-in

| | 本地 `1b86a93f01` | #57158（open） |
|---|---|---|
| 位置 | MRV2 `model_runner._zero_cudagraph_capture_kv_blocks()` | `v1/worker/utils.zero_null_kv_block()` |
| 覆盖面 | 仅 MRV2 | **MRV1 + MRV2** |
| profiling 生命周期 | ✅ 在 `_teardown_profiling_state` 里把缓存地址的 zeroer 置空（防野指针） | 未处理 |
| 测试 | 无 | 2 个纯 CPU 单测 |
| 范围 | block 0 | block 0 |
| 部署证据 | DSpark FULL 路径实测产出连贯 | 17.2 → 32.6 tok/s |

**取优 = #57158 的骨架（helper + 双 runner + 测试）+ 本地的 profiling teardown 安全**。另按 bitMacrocode 的现场证据（PIECEWISE + UVA offload 下 **35 个 page，含真实页** 带 NaN），把"清全部 kv_caches"作为 `VLLM_ZERO_ALL_KV_AFTER_CAPTURE`（默认 0）落地，**默认值维持 block-0 的已验证行为**，GPU 空闲后用它做对照实验再决定是否翻默认。

### 5.3 多流捕获：本地保留重叠 vs #57028 捕获即串行

| | 本地 `c7bf4a65e9` | #57028 |
|---|---|---|
| 机制 | `CUDAGraphCaptureEventPool` + `vllm_cudagraph_capture_scope`（ContextVar 追踪，覆盖 aux stream） | `... or torch.cuda.is_current_stream_capturing()` → 捕获期串行 |
| 重叠 | **保留**（图内仍是并行） | 图中重叠**永久消失** |
| 体积/风险 | 670 行、事件生命周期需 review | 3 行、稳 |
| 其他内容 | — | 几何（重复）+ **MXFP8 空 batch 安全**（独有） |

**取优 = 本地事件池（性能）+ 移植 #57028 的 MXFP8 空 batch 修复**；#57028 的几何部分不取（5.1 已更优）。

### 5.4 spec-decode：三条互补，全取（执行中修正）

原计划全取 `#52499` + `#57388` + 本地两条。执行后收敛为：

| 项 | 决策 | 理由 |
|---|---|---|
| 本地 `dc753a76fb`（边界统一） | ✅ 已落地 `42ba508d0c` | 上游连 issue 都没有；附回归测试 |
| 本地 `4688a08e78`（V2 wide eager） | ✅ 已落地 `f5e2aafc21` | 上游只在 MRV1 侧加宽；V2 侧空白 |
| `#52499`（4-D query / 索引 reshape / ≤64 段 / 空 chunk） | ⏳ 仍建议 overlay | 它修的是我们使用的 SM120 FlashInfer sparse 路径的 spec-decode 形状；上游仍未合并（9/06 后无更新）。需按 §5.5 决策确认 |
| `#57388`（prefill 索引 2D→3D + 分离 workspace） | ❌ **不 vendor** | 是他人仍在 review 的 PR；我们的分支已用 wide-eager + 边界统一覆盖同一故障面。改为"监视上游合并"，避免把别人的未定稿代码带进我们的历史 |
| `#55692`（bound sparse MLA gathers） | ❌ 不 vendor（同上） | 我们只取其中与 top-k 越界同源的那一半，已在 S2 以更小改动覆盖 |

### 5.5 `main` 已有 adaptive verification：本地 `verify_width.py` 的归属

**结论（2026-09-19，已执行）**：实测**调度开销大于收窄 verify batch 的收益**，该路径不进生产，
在 §12 的清理中整段删除（模块 + 测试 + 接线，备份 tag `backup/pre-cleanup-20260919` 保留）。
以下是当时的判断依据。

事实：`main` 的 `AdaptiveVerificationManager` 用实测 cost table + `argmax(tokens/cost)` 选 draft budget（host 侧 numpy，离散 budget）；本地 `v1/spec_decode/verify_width.py` 用 device 端不动点求解、**逐请求**宽度、无 host 同步，并配线上拟合的 `RowCostModel`。本地 dspark speculator 的注释指出：上游 adaptive 路径需要注意力后端能在 device 上裁剪 query 长度，**family 12 没有这样的后端**，所以本地那套是 sm120 的替代方案，不是重复实现。

当时的决策（待确认）：**不进 S1–S8 核心系列，单列为 S9（opt-in）**。理由：
1. 它是**未提交的 WIP**（342 行模块 + 452 行测试 + model_runner/speculator 接线），在本环境**无法验证**——效果必须用真实 DSv4.1-Flash（557 GB checkpoint，本地不存在）跑 acceptance/tok-s 才能判定；
2. 与上游功能面重叠而"更优"未被证明，按 §0 的"简洁正确"与"重叠取最优（无证据则暂缓）"原则，不能让它进默认路径；
3. 其小而有用的诊断部分（`VLLM_DSPARK_CONFIDENCE_HEAD` 与上游 gate 解耦、`_probe_confidence`）可以单独作为 opt-in 提交，代价小、无行为改变。

覆盖该结论的证据：`DECODE-BUDGET.md` / `WIDTH-COST-MODEL.md` 的行宽扫描显示逐请求宽度带来的
ragged verify batch 在调度侧的开销盖过了它省下的行；`serve_real.sh` 里那两个只为该测量存在的
开关（`VLLM_DSPARK_FORCE_REJECT` / `VLLM_DSPARK_WIDTH_PATTERN`）也随之失效。

---

## 6. 补丁系列（执行顺序）

| # | 提交 | 内容 | 依赖 |
|---|---|---|---|
| S1 | geometry | §5.1 的取优实现 + 测试 | — |
| S2 | kv-capture-safety | §5.2 + `7b7c76df22`（top-k 上界） | — |
| S3 | spec-decode-verify-width | `dc753a76fb` + G1 去重 + `4688a08e78`(wide eager) + #52499 overlay + #57388 端口 | S2 |
| S4 | cudagraph-multistream | `c7bf4a65e9` + `6608f6813e` + #57028 的 mxfp8 修复 | S1 |
| S5 | engram-host-tables | 5912d3cbce(a)(b) + `09499e40f2` + `9b18c761fd` | — |
| S6 | checkpoint-loader | `de085303dd` + G4 | — |
| S7 | instrumentation | G5（默认关） | — |
| S8 | mxfp8-o_proj | G3（默认开关回退为 main 行为） | S7（埋点） |
| H1 | breakable-serves-full | 三连 + decorator 修复（**待 GPU 验证**，不进 v1） | S4 |
| H2 | forensic switches | `6d74b9a0f2` / `09c74046c5`（默认关，或直接转 issue） | S4 |

S1–S8 之间除标注依赖外互不阻塞，可并行开发、逐个 CPU 验收。

---

## 7. main 自动继承（不重复实现，也不额外引入）

**已合并且 sm12x 生效**：#57454（NaN 候选块）、#56633（mHC post 折叠，走 `is_deep_gemm_supported()`，**在 family 120 为真**）、#57432（DSpark 非因果）、#57152（V4.1 causal image SWA）、#56441（KV-only 插入）、#56562（Triton metadata 融合）、#57209/#54267/#57512/#54600/#53680（sm12x 修正组）。

**已合并但 SM100/ROCm 专用 —— 保持现状、不主动引入**：#56935、#56962、#56266、#56344、#56893、#57204/#56568/#57604/#57220、#57206、#57109、#57428、#57643、#57603、#57257、#56986 及全部 ROCm PR。

**必须提醒的 sm120 地雷**：`support_deep_gemm()`（`vllm/platforms/cuda.py:719`）把 family 120 视为支持 → #56633 引入的 DeepGEMM `tf32_hc_prenorm_gemm` 会在 sm120 上被选中，而社区实测该路径数值错误（vLLM 自测 34/34 失败，TileLang 142/142 通过）。S8/S7 里的 mHC 埋点与 divisibility 处理即为此服务。

---

## 8. CPU 自测矩阵（本次唯一验收手段）

**运行环境（已验证）**
```bash
PY=~/anaconda3/envs/vllm-ds41/bin/python          # editable → 本仓库，改动即时生效
export PYTHONPATH=/tmp/cpu-tools                   # pytest 9.1.0 / pytest-asyncio / tblib / ruff
# 已装：pytest==9.1.0 pytest-asyncio==1.4.0 pytest-forked==1.6.0 pytest-timeout==2.3.1 \
#       pytest-rerunfailures==14.0 tblib==3.1.0 （与 requirements/test/cuda.txt 对齐，未改动用户 env）
```

### Tier 0（秒级，每个补丁都跑）
```bash
git apply --check <patch>                     # 对 origin/main 干净应用
python3 -m py_compile $(git diff --name-only --diff-filter=ACM | grep '\.py$')
$PY -m ruff check $(git diff --name-only --diff-filter=ACM | grep '\.py$')
$PY -m ruff format --check <同上>
python3 tools/sm120_scope_check.py            # 新增：静态不变量（见下）
```
`sm120_scope_check.py`（本次新增的小脚本，AST/文本级）断言四件事：
1. 本系列新增代码中**不出现** `family(100)`/`sm_100`/`tcgen05`/`deep_gemm_mega_moe`/`gvr_2`/`deep_select`；
2. `deepseek_v41` 下 SM12x 分支的 SWA 页 = 64、压缩/索引器 `num_states` 恒 64；
3. 缺失参数跳过、top-k 上界、null-block 清零三处守卫存在；
4. 所有新增 env 开关默认值 = main 行为（opt-in）。

### Tier 1（分钟级；按文件选模式）

GPU 已释放，因此**默认放开 GPU 跑**；只有需要证明"完全不碰 CUDA"时才加 `CUDA_VISIBLE_DEVICES=""`。

```bash
# 默认（GPU 可用）——已实测 109 passed
$PY -m pytest -q -p no:cacheprovider \
  tests/v1/attention/test_deepseek_v41_block_size.py \
  tests/v1/attention/test_attention_splitting.py \
  tests/v1/worker/test_utils.py \
  tests/models/deepseek_v4_1/test_global_topk_bounds.py

# CPU 受限模式（只用于被 mock 的用例；HiSparse 等真实 CUDA 用例会失败，属预期）
CUDA_VISIBLE_DEVICES="" $PY -m pytest -q -p no:cacheprovider \
  tests/v1/attention/test_deepseek_v41_block_size.py \
  tests/v1/attention/test_attention_splitting.py
```
**基线（实测）**：S1–S6 后 109 passed（37 s）。回归判据：不低于基线、无新增 skip/error、无 `_C` 导入失败。
未落地的测试文件随对应 Sx 一起提交。

风险与兜底：editable 安装的 `_C*.so` 是 base 提交预编译的，若 main 侧 Python 代码引用了新 C++ 符号，`import vllm` 会失败 → 该模块退化为 Tier 0 静态检查 + 明确标注"待 GPU/重编译验证"。

### Tier 2（GPU 恢复后的验收清单，本次不执行）
serve 冒烟（eager + 图）、1M 上下文 needle、DSpark k=5 tok/s 与 acceptance、`VLLM_ZERO_ALL_KV_AFTER_CAPTURE` 对照、`--block-size 128` 起服、`CUDA_LAUNCH_BLOCKING=1` 长 prompt prefill。

---

## 9. 风险与回滚

| 风险 | 缓解 |
|---|---|
| #52499 overlay 与上游漂移 | overlay 提交单独标记，PR 合并后 `git revert` 一条即可 |
| #57158/#56509/#57388 上游先合并 | 我们的 S1/S2 与其重叠部分按 §5 取优后可能不再需要 → 合并前先查这三条状态，若已合并则退化为"只用上游" |
| 未提交工作含未验证默认开关 | 一律回退为 opt-in；默认值 = main 行为 |
| H1 组带入 degenerate 风险 | 不进 v1，单独分支 + Tier 2 验收 |
| 静态检查盲区（Tier 0 无法证明数值正确） | 每个 Sx 明确列出"本补丁未被 CPU 证明的部分"，写进提交信息 |

**回滚**：整个系列线性提交，`git revert S4..S1` 或直接切回 `fix/clear-kv-and-full-guard`；旧分支在方案执行期间保持不动。

---

## 10. 执行清单（已完成）

- [x] §3 的 DROP 与 HOLD 全部执行（证据见 §0.5 末尾）
- [x] all-pages 默认值：`VLLM_ZERO_ALL_KV_AFTER_CAPTURE` 默认 0（block-0），全清 opt-in
- [x] G3 性能开关：**改变数值/几何的**一律 opt-in（`VLLM_IR_RMS_NORM_EAGER_FUSED`、`VLLM_MHC_PRE_MAX_SPLITS` 回上游值）；**同数学、只换 kernel 的**保持默认开（o_proj 两条、`VLLM_ACT_QUANT_INDEXED`），理由写在各提交信息里
- [x] S7 instrumentation 进主线，独立提交，默认关
- [x] `sm120/rebase-main` 建成，S1→S9/S11 顺序落地，每步附 Tier 0/1 结果
- [x] 旧分支与备份未动；GPU 空闲期间完成了 GPU 侧验收（Tier 2 的真实模型 e2e 仍缺 checkpoint）

## 11. 验收记录

**Tier 0（全部改动文件）**
```
python3 -m py_compile <57 个改动文件>            -> OK
ruff check <57 个改动文件>                       -> All checks passed
ruff format --check <57 个改动文件>              -> 57 files already formatted
python3 tools/sm120_scope_check.py origin/main   -> passed
```
`tools/sm120_scope_check.py`（本次新增）断言：增量代码不出现 `family(100)` / `sm_100` / tcgen05 /
`deep_gemm_mega_moe` / `deep_select` / `gvr_2`；四道守卫（top-k 上界、缺失参数跳过、捕获后 KV 恢复、family-12
页几何）都在；17 个新开关默认值为保守值。

**Tier 1（GPU 空闲，默认模式）**
```
pytest -q tests/v1/attention/test_deepseek_v41_block_size.py \
          tests/v1/attention/test_attention_splitting.py \
          tests/v1/attention/test_deepseek_v4_swa_visible.py \
          tests/v1/worker/test_utils.py \
          tests/test_multi_stream_utils.py \
          tests/models/deepseek_v4_1/ \
          tests/model_executor/test_deepseek_v4_engram_warmup.py \
          tests/kernels/quantization/test_mxfp8_empty_batch.py
```
其中 `test_deepseek_v4_swa_visible.py::test_builder_in_image_visibility` 因**本机无 HF 网络**
（`Network is unreachable`，失败点在 `tests/v1/attention/utils.py` 构造 `ModelConfig`）失败，与本次改动无关。

**Tier 2（未做）**：真实 `DeepSeek-V4.1-Flash` e2e（557 GB checkpoint 不在本机）、tok/s 与
acceptance 测量。

---

## 12. 开关清理（2026-09-19，完成）

系列落地后复查了它新增的 36 个环境变量，按「默认路径是否就是最优路径」分三类处理。
备份 tag：`backup/pre-cleanup-20260919`。五个提交（`df4a09bcf4` → `78fa50bfe6`）：

| 类别 | 处理 | 变量 |
|---|---|---|
| **开了只有好处**（同数学 / 逐比特验证过 / 无回退价值） | 删开关与旧分支，只留优路径 | `VLLM_MXFP4_MOE_COMPACT_W13`、`VLLM_O_PROJ_GROUPED_MM`、`VLLM_O_PROJ_DIRECT_WRITE`、`VLLM_ACT_QUANT_INDEXED`、`VLLM_IR_RMS_NORM_EAGER_FUSED`、`VLLM_MHC_PRE_MAX_SPLITS`(→64) |
| **只为 A/B 固定选择存在** | 删开关，保留分支默认 | `VLLM_MXFP8_SPLIT_K`、`VLLM_ZERO_ALL_KV_AFTER_CAPTURE`、`VLLM_FORCE_VARLEN_DECODE_CAPTURE` |
| **未被交付配方使用** | 连同被守卫的代码一起删 | DSpark 宽度求解器 7 个 + `VLLM_RAGGED_DEBUG`（实测调度开销大于收益，不进生产）；InstantTensor 加载器实验 4 个（`serve_real.sh` 记录阻塞 4 致命 + DROP_CACHE 实测否决） |
| **真诊断** | 保留，规范化进 `vllm/profiler/` | `VLLM_GRAPH_KERNEL_ANNOTATIONS`、`VLLM_GEMM_SHAPE_LOG`、`VLLM_COMM_SHAPE_LOG`、`VLLM_MOE_PAD_LOG`、`VLLM_MEM_LOG`、`VLLM_KV_ACCOUNTING_LOG`(+`VLLM_PROFILE_DIR`) |
| **真旋钮** | 保留 | `VLLM_MXFP8_BACKEND_POLICY` |

结果：**36 → 8**，`tools/sm120_scope_check.py` 的 `CONSERVATIVE_DEFAULTS` 同步收敛到
这 8 个；`model_loader/` 两个文件回到与上游逐字节一致，DSpark 宽度路径整体消失（约
1400 行含测试）。

诊断开关的规范形态见 `vllm/profiler/instrumentation.py`：一个开关注册表、一次
「同一条观测只记一次」、一个产物目录 `VLLM_PROFILE_DIR`（4 rank 共工作目录时按
`<kind>.log` 分文件）。原来两个只表示「产物放哪」的变量
（`VLLM_GRAPH_KERNEL_ANNOTATION_DIR` / `VLLM_GEMM_SHAPE_LOG_PATH`）合并掉了。

