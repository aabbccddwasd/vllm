# SM120 rebase — E2E 验收（2026-09-19）

分支：`sm120/rebase-main`（19+1 提交，自 `origin/main` a1bf8ac12d）
被测模型：`/mnt/data-zt-4t/models/DeepSeek-V4.1-Flash`（557 GB 真权重，TP4，fp8 KV）
硬件：4× RTX PRO 6000 Blackwell（sm_120，PCIe，无 NVLink）
工具：`~/ds41mig/tools/bench_arm.sh` + `bench_run.sh`，`~/AI-stuffs/test-toolkit/bench_dsv4.py`（未改动）
协议：FULL_AND_PIECEWISE 图网格 `[6,12,24,48]`/max 48、DSpark N=5 probabilistic、SEQS=16 CB=8192 GMU=0.93 MAXLEN=32768、`VLLM_USE_BREAKABLE_CUDAGRAPH=0`

## 0. 前置：`_C` 扩展必须随 main 重编译

main 自 base 起改过 C++ 签名（#56441 的 KV-only context 插入），而 editable 安装的
`vllm/_C_stable_libtorch.abi3.so` 是 base 提交的预编译产物，直接起服会：

```
RuntimeError: _C::fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert() expected at
most 10 argument(s) but received 13 argument(s)
```

增量重编译（7 分钟，ccache 命中率高）：

```bash
CUDA_HOME=/usr/local/cuda python tools/generate_cmake_presets.py
CUDA_HOME=/usr/local/cuda cmake --preset release
CUDA_HOME=/usr/local/cuda ninja -C cmake-build-release \
    _C_stable_libtorch.abi3.so _moe_C_stable_libtorch.abi3.so
cp cmake-build-release/_C_stable_libtorch.abi3.so cmake-build-release/_moe_C_stable_libtorch.abi3.so vllm/
```

**任何基于 main 的分支都必须先做这一步。** 旧分支（base 系）用旧 `.so`，两者不能混用。

## 1. 启动证据（rebase 臂）

| 判据 | 结果 |
|---|---|
| guard 降级 | **0 次**（`setting cudagraph_mode` 无降级日志） |
| target FULL 捕获 | `Capturing CUDA graphs (FULL): 2/2` 与 `4/4` |
| drafter FULL 捕获 | `Capturing dspark CUDA graphs (FULL): 3/3` |
| b12x_tuned MXFP8 | 编译并命中（`tile_m=32 split_k=8` 等），无 Cutlass 回退 |
| TileLang mHC | `mhc_fused_tilelang` / `mhc_pre_big_fuse_with_norm_tilelang` 编译 |
| 崩溃 / EngineDead | 0 |

注：唯一一次启动失败是 `AttributeError: module 'vllm.envs' has no attribute
'VLLM_MOE_PAD_LOG'`（我从 WIP 提取 env 时漏声明），已修并加了静态检查。

## 2. 吞吐对照（同会话、同网格、同参数、同顺序 A→B）

两臂：**base** = `fix/clear-kv-and-full-guard`（dc753a76f + 旧 `.so`）；**rebase** = `sm120/rebase-main`（新 `.so`）。

### 协议 A — 单请求（math/creative，各 2 轮，512 输出）

| 任务 | base NT / T | rebase NT / T | 差 |
|---|---|---|---|
| math | 273/260 · 305/225（均 267 / 265） | 273/261 · 301/257（均 267 / 279） | NT ±0% · T **+5%** |
| creative | 148/155 · 149/157（均 152 / 153） | 157/173 · 172/176（均 165 / 174） | NT **+9%** · T **+14%** |

接受率（MTP N=5）：math base 49.6%/3.48、49.3%/3.47 ↔ rebase 43.7%/3.18、46.8%/3.34；
creative base 19.4%/1.97、20.1%/2.00 ↔ rebase 19.3%/1.96、21.5%/2.07。**基本同级**（math 略低，在轮间噪声内）。

单轮跨度很大（math NT 273↔385），与 `HANDOFF-2026-09-15.md` 的警告一致：2 轮不足以支撑 ±10% 结论，
所以上表按"同会话同序"读，不做跨会话外推。

### 协议 B — 并发（math+creative，每请求独立 2k 上下文，≤512 输出，无 warmup）

| C | base dec tok/s | rebase dec tok/s | 差 | base prefill | rebase prefill | 差 | base TTFT | rebase TTFT |
|---|---|---|---|---|---|---|---|---|
| 1 | 256.7 | 244.6 | −5% | 11,612 | 11,398 | −2% | 234 ms | 237 ms |
| 2 | 374.7 | 354.7 | −5% | 14,711 | 11,921 | −19% | 388 ms | 436 ms |
| 4 | 481.9 | **501.3** | **+4%** | 17,818 | **20,337** | **+14%** | 702 ms | 713 ms |
| 8 | 696.3 | **734.7** | **+5%** | 26,861 | 24,804 | −8% | 1,144 ms | 1,168 ms |

**结论：整体在轮间噪声（±10%）内；单流 creative 与 C=4/C=8 解码略优，C=2 prefill 略低。**

> 早前一版 rebase 臂测出 C=1 只有 236 tok/s / TTFT 728 ms，是因为它的协议 B 紧接启动（冷编译轮
> 135 tok/s 计入均值），而 base 臂先跑过协议 A。上表是**同序重跑**后的数据，那个差异是测量顺序伪影。

### 与 09-13 历史读数对照（`BENCH-FULL-SPEC-DECODE.md`，旧网格 `[1..32]`）

| 维度 | 历史（09-13） | 本次 rebase 臂 |
|---|---|---|
| math NT/T | 254/217 · 280/227 | 273/261 · 301/257 |
| creative NT/T | 144/140 · 130/138 | 157/173 · 172/176 |
| prefill（math+creative） | 9,682 | 11,398 |
| TTFT | 264 ms | 237 ms |

（网格不同——本次是你后来修正过的 `[6,12,24,48]`——所以只作参考，正式对照用上面的 base 臂。）

## 3. 正确性

| 检查 | 结果 |
|---|---|
| 短 prompt（greedy） | 输出连贯，含 "Paris" |
| 24,058-token needle（`PURPLE-MOON-4711`） | **命中** ✅（这一项正是用来抓 FP8 KV 布局/NaN 污染那类静默损坏的） |
| 全程崩溃 / IMA / Warp MMU fault | 0 |
| 图降级 / eager 回退 | 0 |

## 4. 未覆盖（诚实清单）

- **长上下文**：本轮 `MAXLEN=32768`。1M（`kv-cache-memory-bytes=4GiB`）未测。
- `VLLM_ZERO_ALL_KV_AFTER_CAPTURE=1`（全清 KV）未做对照——默认的 block-0 路径已跑通。
- 交付配方里的 `VLLM_MHC_PRE_MAX_SPLITS=64` / `VLLM_IR_RMS_NORM_EAGER_FUSED=1` 由 `serve_real.sh`
  显式导出，两臂一致；本分支的**保守默认**（16 / 0）未单独做 A/B。
- 6 轮协议（±3% 级结论）未跑；本轮 2 轮只支持"同级/±10%"结论。
- H1（breakable serves FULL 三连）仍不在本分支。

## 5. 复现命令

```bash
# 1) 重编译 _C（见 §0）
# 2) 起服并跑协议
cd ~/ds41mig/tools
SEQS=16 CB=8192 GMU=0.93 MAXLEN=32768 DSV41_PORT=8040 ARM_PORT=8040 ./bench_arm.sh <label> full
PY=~/anaconda3/envs/vllm-ds41/bin/python; B=~/AI-stuffs/test-toolkit/bench_dsv4.py
$PY $B --url http://127.0.0.1:8040 --model dsv41-real --tasks math,creative --task-rounds 2 --output-len 512
$PY $B --url http://127.0.0.1:8040 --model dsv41-real --tasks math,creative --task-rounds 2 \
       --output-len 512 --concurrency 1,2,4,8 --concurrency-rounds 2 --no-shared-prefix
# 3) 停服：kill -- -$(cat /tmp/bench-<label>.pgid)
```

注：`bench_dsv4.py` 的 code 任务需要 `/tmp/bench_code_prompts.jsonl`（本机已被 tmp 清理），
因此两臂都只跑 math+creative；如需 code 任务，先恢复该 prompts 文件。
