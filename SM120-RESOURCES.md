# SM120 专用优化资源清单（2026-09-19 调查）

> 起因：前面按"DSv4.1 PR 标题"过滤，结论是"上游几乎没有 sm120 专用优化"。**那个结论是错的** ——
> sm120/121 的优化在上游是一条独立主线（B12X），标题里不带 DSv4.1，所以被过滤掉了。
> 本文按"**以 sm120 本身为目标**"重查：vLLM 上游代码级检索 + 跨仓库 + 社区 fork，并给出已实测的收益。

---

## 1. vLLM 上游：B12X 后端栈（sm120/sm121 专用，已合并）

代码级检索 `is_device_capability_family(120)` 命中 30 个文件，其中 `b12x` 是一个完整栈：

| 文件 | 行数 | 内容 |
|---|---|---|
| `vllm/v1/attention/backends/b12x.py` | 1096 | **sm120/121 因果分页 attention 后端** |
| `vllm/model_executor/layers/fused_moe/b12x.py` | 770 | MoE（FP4 权重） |
| `vllm/model_executor/kernels/linear/scaled_mm/b12x.py` | 339 | FP8 blockwise |
| `vllm/model_executor/kernels/linear/{nvfp4,mxfp8,mxfp4}/b12x.py` | 141/151/130 | 三种 FP4/FP8 线性 |
| `vllm/model_executor/layers/fused_moe/experts/flashinfer_b12x_moe.py` | 300 | FlashInfer CuteDSL MoE（另一条） |
| `vllm/model_executor/warmup/b12x_warmup.py` | 89 | 启动预热 |
| `vllm/utils/b12x.py` | 134 | 可选包探测 |
| `docs/features/quantization/b12x.md` | 37 | **官方文档** |

引入 PR（全部 merged）：**#40082**（整合 FlashInfer b12x MoE/FP4 GEMM on SM120/121）、**#52016**（dense linear）、
**#52017**（attention 后端）、**#52018**（FP4 MoE）、**#52368**（重构+预热）。测试 5 个文件约 3000 行
（`tests/model_executor/kernels/test_b12x_linear.py` 861 行等）。

**启用方式**（`docs/features/quantization/b12x.md`）：
```bash
uv pip install "vllm[b12x]"
vllm serve <model> --linear-backend b12x --moe-backend b12x
```
支持面：linear = per-tensor FP8 / 128x128 block FP8 / MXFP8 / NVFP4 / MXFP4；MoE = TP 的 MXFP4（BF16 或
MXFP8 激活）与 NVFP4（BF16/NVFP4/MXFP8 激活）；**MoE 不支持 EP、expert map、EXL3、NF3**。
另有 `--linear-backend-per-quant`（`linear_backend_per_quant`）可按量化类型分别指定。
`VLLM_B12X_MOE_FP4_FORCE_A16=1` 可强制 BF16 激活。

> 注意两组同名后端：`b12x`（需外部包，native FP8/FP4 kernel）与 `flashinfer_b12x`（FlashInfer 内置 CuteDSL）。
> **MXFP4 MoE 只接受 `b12x`**：`flashinfer_b12x` 会直接报
> `moe_backend='flashinfer_b12x' is not supported for MXFP4 MoE`（已实测踩到）。

## 2. `b12x` 库本身（社区）

- **`local-inference-lab/b12x`（222★，今天仍在更新）** —— "**DGX Spark and RTX 6000-focused inference
  kernel library**"，作者 Luke Alonso，Apache-2.0，PyPI `b12x`（当前 1.3.0，纯 Python wheel + JIT 编译 kernel）。
- 依赖 `nvidia-cutlass-dsl==4.6.2` / `cuda-python` / `apache-tvm-ffi` —— **本机 env 全部已满足**
  （实测 `nvidia-cutlass-dsl 4.6.2`、`cuda-python 13.3.1`、`apache-tvm-ffi 0.1.11`），所以可以
  `pip install --no-deps` 装到独立目录用 `PYTHONPATH` 注入，**完全不动生产 env**。

## 3. 其它 sm120/121 专属的上游改动（已 merged，值得逐条对照我们是否吃到）

| PR | 内容 | 与我们的关系 |
|---|---|---|
| #55170 | NVFP4 优先选 W4A4 线性 kernel（而非 weight-only）on SM120/121 | 已在我的 rebase 里（main） |
| #55180 | SM12.x blockwise FP8：权重超出 L2 时切换 CTA raster swizzle | 同上 |
| #55715 | FlashInfer GDN prefill 在 SM12x 上启用 | 同上 |
| #53574 | SM120 DSv4：C128A decode topk 索引连续化 | 同上 |
| #54465 | FlashInfer sparse MLA 的 BLHNC 寻址修复 | 同上 |
| #54110 | 低 shared-memory GPU 上放弃 persistent top-k | 同上 |
| #56478 | per-token-group 量化的奇数行性能悬崖 | 同上 |
| #53576 | **FlashInfer PCIe IPC all-reduce（opt-in）** | **本机 PCIe-only、TP4，值得试** |
| #47751 | consumer Blackwell 上禁用 custom all-reduce | 已生效（日志可见） |
| #48210 | sm_12x 上把 bmm_fp8 钉到 cuBLAS | 同上 |
| #54267 | CUDA < 12.9 时禁用 FlashInfer on sm_120 | 同上 |
| #54600 / #53680 | DeepGEMM 排除 SM121 / pin 回退 | 同上 |

## 4. FlashInfer 侧（sm120 kernel 与 autotune）

- sm120 sparse-MLA decode/prefill（`sparse_mla_sm120_*`）、b12x mxfp8/moe CuteDSL runner、XQA 等。
- **我们实测撞到的缺口**：`[AutoTuner]: No tuned config covers mxfp8_gemm input_shapes=((48,1280),(1280,4096),…);
  falling back to runner=B12xMxfp8GemmRunner tactic=-1. This shape is outside the tuning bucket range --
  expand tuning_buckets / max_num_tokens during the next tuning pass to avoid this perf cliff.`
  → 我们的 serving 形状（M 最大 48、以及 N/K 组合）不在默认 tuning bucket 覆盖内，走的是默认 tactic。
  vLLM 侧有 `VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR` + `flashinfer_autotune_cache` 机制，可以**离线 tune 一次、
  落盘缓存、上线加载**（交付配置里 autotune 被关掉正是因为 GMU 0.97 下 autotune 的 scratch 会 OOM；
  有缓存就不需要开着 autotune）。

## 5. 社区 fork / recipe（明确以 sm120/sm121 为目标）

| 仓库 | 目标 | 备注 |
|---|---|---|
| `ciprianveg/gb10-vllm` (31★) | GB10/sm121：KIMI-K3 **B12X_MLA** + DSpark、GLM-5.2 | b12x MLA 用法参考 |
| `MiaAI-Lab/Ornith-1.5-35B-A3B-DGX-Spark` (30★) | **vLLM b12x recipe + CUDA-graph 修复** | 与我们的图问题同族 |
| `hikarioyama/dsv4-flash-0731-sm120` (10★) | **2× RTX PRO 6000（同款卡）**：vLLM **B12X r16** + DSpark K5 | 最贴近我们的对照配方 |
| `r0b0tlab/DeepSeek-V4-Flash-DSpark-v026-SM121` (32★) | sm121：NVFP4 KV + **B12X** + DSpark K6 | NVFP4 KV 路线 |
| `luxingcom/…DeepSeek-v4-Flash-DSpark-DGXspark-TP4-Ring` (13★) | DSv4-Flash TP4 ring：`VLLM_MOE_W4A4=2` + **`VLLM_B12X_SHARED_WRAPPER=1`** | 这两个 env 不在上游 main（本树 grep 为 0），来自 b12x 包或 fork |
| `brandonmmusic-max/sm120-kernels` (6★) | "SM120 Kernels" | 独立 kernel 集合 |
| `bhaktatejas922/b12x` (0★) | "B12x SM120 kernel work for **DSv4 Flash**" | 同题材 |
| `jarrelscy/dspark-b12x` (3★) | DSpark + b12x | 与 DSv4.1 spec decode 相关 |
| `seanyourhighness/vllm-sm120-nvfp4-mtp` | 单卡 5090：NVFP4 权重+KV + MTP-3 | |
| `lna-lab/gemma4-12b-vllm-sm120`、`adrienbrault/qwen3.8-27b-rtx5090`、`ekojsalim/…`、`nickus/vllm-sm120`、`ethanfel/vllm-sm120-unraid` | sm120 recipe / overlay | 通用参考 |

---

## 6. 本机实测：b12x 对 DSv4.1-Flash 的收益（新增，2026-09-19）

同一分支（`sm120/rebase-main`）、同网格 `[6,12,24,48]`、DSpark N=5、同协议，唯一变量是后端：

| | 基线（auto） | **b12x** | 差 |
|---|---|---|---|
| MoE 后端 | `DEEPGEMM_MXFP4` | **`B12X_MXFP4_MXFP8`**（`B12xExperts`） | — |
| MXFP8 linear | `B12xTunedMxfp8LinearKernel` + 部分层 Emulation | **`B12xMxfp8LinearKernel`** | — |
| C=1 decode | 244.6 tok/s | **270.4** | **+10.6%** |
| C=2 decode | 354.7 | 332.7 | −6% |
| C=4 decode | 501.3 | **528.3** | **+5.4%** |
| C=8 decode | 734.7 | **766.9** | **+4.4%** |
| C=8 prefill | 24,804 | **27,387** | **+10.4%** |
| C=1 TTFT | 237 ms | 229 ms | −3% |
| 单流 math NT/T | 267 / 279 | 257 / 260 | −4% / −7% |
| 单流 creative NT/T | 165 / 174 | 164 / 167 | −1% / −4% |

（2 轮协议，±10% 噪声；C=1/C=4/C=8 与 C=8 prefill 三格同向，单流持平或略低。）

**复现（不动生产 env）**
```bash
pip install --no-deps --target /tmp/b12x-site b12x         # 依赖已在 env 内满足
cd ~/ds41mig/tools
PYTHONPATH=/tmp/b12x-site SEQS=16 CB=8192 GMU=0.93 MAXLEN=32768 \
  ./bench_arm.sh b12x-full full --moe-backend b12x --linear-backend b12x
```

---

## 7. 建议的下一步（按收益/代价排序）

1. **把 b12x 纳入交付配方**：装 `b12x`（或 `pip install "vllm[b12x]"`）+ `--moe-backend b12x`
   —— 本机实测 C=1 +10.6%、C=8 +4.4%、C=8 prefill +10.4%，且启动/捕获无回归。
2. **补 MXFP8 linear 的 Emulation 回退**：基线里仍有层走 `EmulationMxfp8LinearKernel`（≈dequant+bf16），
   用 `--linear-backend b12x` 或扩 `b12x_tuned` 的 shape 覆盖消掉它。
3. **FlashInfer mxfp8_gemm 的 tuning bucket 覆盖**：离线 tune 一次落缓存（`VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR`），
   消掉 `tactic=-1` 的性能悬崖，之后就能在高 GMU 下保持 autotune 关闭。
4. **试 `FLASHINFER_PCIE_IPC` all-reduce**（#53576）：本机 TP4 全 PCIe、无 NVLink，正好是它的目标场景。
5. **长期长上下文**：本清单全部结论只在 32k 验证过；1M 配置（`kv-cache-memory-bytes=4GiB`）需另测。
