# DSV4-Vision 工作树修改归档（按功能分组）

来源：远程容器 `rl-dsv4-vision`（80.5.25.126）内各 git 仓库工作树的未提交修改，2026-10-15 以只读方式（git status / git diff / git show / cat）提取。
所有 patch 均为标准 unified diff（git 风格 a/ b/ 路径），应用方式统一为：

```bash
cd <仓库根目录> && patch -p1 < <patch文件>
```

各 patch 文件头注释中已写明目标仓库绝对路径、对应旧编号 patch（本目录上级 `../0001~0010`）及同文件链式前置依赖。

## 涉及仓库与基线

| 仓库（容器内路径） | HEAD | 修改文件数 |
|---|---|---|
| /workspace/verl | 3d36367e | 19（+2 untracked 需归档） |
| /workspace/MindSpeed-Bridge | 6015a5b | 5 |
| /vllm-workspace/main/zyy/vllm-ascend | c04c5db02 | 10 |
| /vllm-workspace/main/zyy/vllm | 2cf0a6915c | 1 |
| /workspace/TransformerEngineNPU | 3fa460a | 1 |
| /workspace/Megatron-LM、/workspace/Megatron-Bridge | — | 干净，无修改 |

## 分组总表

### 01-consistency-fixes/ — 训推一致性修复链（旧 4 层模型时代已验证闭环）

| patch | hunk | +/- | 旧编号 | 必须合入 |
|---|---|---|---|---|
| verl__images-forward-fix.patch | 3 | +58/-1 | 0002 | 是 |
| verl__megatron-vision-model-flag.patch | 2 | +10/-2 | — | 是 |
| verl__processor-tokenizer-wiring.patch | 9 | +68/-3 | — | 是 |
| verl__agent-loop-media-collapse.patch | 1 | +38/-10 | — | 是 |
| verl__rollout-server-tweaks.patch | 5 | +17/-4 | — | 部分（sleep level/单 replica 为调参） |
| verl__env-config-knobs.patch | 3 | +34/-2 | — | 否（profiler/日志级别/数据集路径，环境相关） |
| verl__npu-expert-layout-restore.patch | 2 | +78/-0 | — | 是（含 untracked 新文件 verl/utils/vllm/npu_expert_layout.py） |
| vllm-ascend__vision-kernel-align.patch | 5 | +26/-10 | 0001 | 是 |
| vllm-ascend__mm-preprocess-preexpanded-prompt.patch | 2 | +72/-0 | — | 是 |
| vllm-ascend__310p-moe-method-revert.patch | 6 | +2/-58 | — | 是（整文件归入，未细分） |
| vllm-ascend__kv-cache-sleep-pool.patch | 7 | +32/-14 | — | 是 |
| vllm-ascend__routed-experts-comment-experiments.patch | 5 | +157/-0 | — | 否（**未拆分**：全部为注释掉的 [fix -modify]/[fix -debug] 实验代码+空行，无行为变化） |
| vllm__routed-experts-mapping-and-transposed.patch | 14 | +65/-9 | — | 部分（**未拆分，混合功能**：expert params mapping 前缀修复 + is_fused_checkpoint_transposed 硬置 True 为功能性；另含一行 weight_loader logger.info 调试日志与注释实验代码） |

涉及文件：verl 侧 model_forward.py、transformer_impl.py（仅 vision_model 2 hunk）、tokenizer.py、deepseek.py、continuous_token_wiring.py、trainer_base.py、agent_loop.py、main_generation_server.py、engine_workers.py、third_party/vllm/__init__.py、dapo_30b shell、ppo_megatron_trainer.yaml、constants_ppo.py、geo3k.py、utils.py（仅 layout 还原 hunk）+ npu_expert_layout.py（新文件）；vllm-ascend 侧 vision.py、mm_preprocess.py、_310p/fused_moe.py、worker.py、model_runner_v1.py（仅 kv_cache_allocation_context 4 hunk）、ops/fused_moe/routed_experts.py；vllm 仓 routed_experts.py。

### 02-exp16-bringup/ — Exp16 跑通修复

| patch | hunk | +/- | 旧编号 | 必须合入 |
|---|---|---|---|---|
| TransformerEngineNPU__m0-guard.patch | 2 | +16/-0 | 0003 | 是 |
| verl__run-dsv4-flash-vision-megatron-script.patch | 1 | +675/-0 | 0004 | 是（untracked 新文件，env 转发列表 + 发射器） |

### 03-weight-transfer-perf/ — 权重传输性能（EP 专家过滤）

| patch | hunk | +/- | 旧编号 | 必须合入 |
|---|---|---|---|---|
| verl__xfer-expert-filter.patch | 12 | +72/-1 | 0008 | 是（env 开关控制，默认关） |
| vllm-ascend__load-params-dict-guards.patch | 4 | +9/-3 | — | 是 |

涉及文件：verl bucketed_weight_transfer.py（expert_filter 参数 + _XFER_EXPERT_RE + 过滤 continue）、vllm_rollout.py（DSV4_XFER_SENDER_EXPERT_FILTER 发送侧开关，另含 1 个纯空行 hunk）、utils.py（_xfer_drop_nonlocal_experts/_xfer_local_expert_ids + patch_vllm_moe_model_weight_loader 调用）、utils/vllm/patch.py（**纯空白 hunk，无功能变化**，一并在此）；vllm-ascend model.py（4 处 params_dict 缺失 guard，只含功能性 hunk）。

### 04-mc2-workaround/ — MoE 通信绕路

| patch | hunk | +/- | 旧编号 | 必须合入 |
|---|---|---|---|---|
| vllm-ascend__force-moe-alltoall.patch | 2 | +4/-0 | — |  workaround（DSV4_FORCE_MOE_ALLTOALL=1 生效） |
| vllm-ascend__rand-dummy-inputs.patch | 1 | +11/-0 | 0009 | workaround（VLLM_RANDOMIZE_DP_DUMMY_INPUTS=1 生效） |

### 05-bridge-fixes/ — MindSpeed-Bridge 修复

| patch | hunk | +/- | 旧编号 | 必须合入 |
|---|---|---|---|---|
| MindSpeed-Bridge__csa-smla-fused-guard.patch | 1 | +2/-0 | — | 是（_CSA_SMLA_FUSED 总开关） |
| MindSpeed-Bridge__debug-leftovers.patch | 2 | +4/-0 | — | 否（csa_fused 遗留 print("[is_paded]") 每次调用都会输出；hybrid_attention 为注释掉的 ROPE-CHECK 实验代码） |

bridge 另 2 个文件（deepseek_v4_flash_vision_model.py、forward_step.py）的全部 hunk 均为休眠探针，归 06 组。

### 06-debug-probes/ — 调试探针（可选/休眠，不合入主线）

| patch | hunk | +/- | 说明 |
|---|---|---|---|
| verl__dbg-metrics-dump.patch | 1 | +19/-0 | metrics.py 一次性 dump dbg_step1.pt |
| verl__dbg-megatron-hash-io-probes.patch | 2 | +84/-0 | transformer_impl.py [DBG_MHASH]×2 + [DBG_IO] |
| verl__dbg-xfer-probes.patch | 9 | +93/-2 | bucketed_weight_transfer.py + utils.py 的 [DBG_XFER]/[LOAD_PROF] 计时探针与 _exps 遗留（母本 ~/dsv4-exp16/add_xfer_prof.py、add_load_prof.py） |
| vllm-ascend__dbg-mm-io-emb-probes.patch | 3 | +37/-1 | vl_model.py [DBG_IO]/[DBG_EMB]（/tmp/dsv4_ioidump 触发） |
| vllm-ascend__dbg-load-map-zero-bias.patch | 5 | +24/-0 | model.py _dbg_map/[DBG_SKIP]/[DBG_ZERO_BIAS_VL] + 注释掉的 [dbg] 打印 |
| vllm-ascend__dbg-mc2-combine-trace.patch | 1 | +13/-0 | token_dispatcher.py DSV4_MOE_TRACE/[MC2COMBINE] |
| MindSpeed-Bridge__dbg-io-emb-probes.patch | 2 | +34/-0 | vision model [DBG_EMB] + forward_step.py [DBG_IO]（相关旧 0007 ltrace 版本） |

说明：[DBG_VHASH]/[DBG_HASH] 哈希探针已从工作树删除（旧 0010 即删除它们的 patch），故不在任何 diff 中；git HEAD 里也不含它们，无需处理。

## 同文件多 patch 的应用顺序（链式依赖）

同一文件被拆到多个组时，patch 按"功能在前、探针在后"生成，**必须按以下顺序应用**（各 patch 头部也注明了前置依赖）：

| 文件 | 应用顺序 |
|---|---|
| verl transformer_impl.py | 01 verl__megatron-vision-model-flag → 06 verl__dbg-megatron-hash-io-probes |
| verl utils.py | 01 verl__npu-expert-layout-restore → 03 verl__xfer-expert-filter → 06 verl__dbg-xfer-probes |
| verl bucketed_weight_transfer.py | 03 verl__xfer-expert-filter → 06 verl__dbg-xfer-probes |
| vllm-ascend model.py | 03 vllm-ascend__load-params-dict-guards → 06 vllm-ascend__dbg-load-map-zero-bias |
| vllm-ascend model_runner_v1.py | 01 vllm-ascend__kv-cache-sleep-pool → 04 vllm-ascend__rand-dummy-inputs |

跳过中间某个 patch（如不应用 03 直接打 06）可能因上下文不匹配失败。以上每条链均已验证：按序应用后文件内容与容器工作树逐字节一致；每个拆分 patch 的标记断言（功能 patch 不含探针标记、探针 patch 不含功能代码）均已通过。

## 归属判定备注

- verl metrics.py：侦察规范同时列于 01 组文件清单与 06 组探针清单；其唯一 hunk 是 DEBUG_DUMP 落盘，归 06。
- verl ppo_megatron_trainer.yaml（profiler 配置）、constants_ppo.py（日志级别）、geo3k.py（数据集路径）、dapo shell（enable_sleep_mode）、main_generation_server（num_replicas=1）：按规范归 01，但均为环境/调参性质，已在 patch 备注与上表标明"可选"。
- vllm-ascend ops/fused_moe/routed_experts.py：全部 5 个 hunk 均为注释代码/空行，无行为变化，整文件归 01 并标注未拆分。
- vllm 仓 routed_experts.py：功能修复与调试残留混合，难以可靠拆分，整文件归 01 并标注。
- verl utils.py 的 restore_expert_checkpoint_layout hunk 依赖 untracked 新文件 verl/utils/vllm/npu_expert_layout.py，二者合为一个 patch（01 组），判为权重布局一致性（正确性）而非性能。
- bucketed_weight_transfer.py / utils.py 的功能/探针拆分以容器内 .bak 备份链还原应用次序后切分：bwt 为 HEAD →(xferdbg)→ bak_epfilter →(epfilter)→ bak_xferprof →(xferprof)→ 当前；utils 为 HEAD →(手工 layout/loader)→ bak_epfilter(=bak_xferfusion) →(epfilter)→ bak_xferprof →(xferprof)→ bak_loadprof →(loadprof)→ 当前。model_runner_v1.py 为 HEAD →(kv-pool)→ bak_randdummy →(rand dummy)→ 当前。
- 旧编号 0005/0006（Megatron-LM token_dispatcher/mappings 探针）：Megatron-LM 当前工作树干净，探针已不存在，无对应新 patch。

## 未覆盖项（untracked 杂项，仅记录，不做 patch）

- /workspace/verl 下 21 个 core dump（core_VLLM::Worker_*、core_python3_*）—— 待清理垃圾。
- /workspace/verl/W2_WEIGHT_LAYOUT_FIX.md —— 调查笔记，待清理（其结论已体现为 01 组 verl__npu-expert-layout-restore.patch）。
- /workspace/verl/extra-info/、npu-smi —— 运行时产物，待清理。
- /workspace/verl/test_dsv4_flash_vision_vllm.sh、test_dsv4_flash_vision_vllm_by_step.sh、test_rollout_available.py、test_rollout_only.sh、trans_data_countbenchqa.py —— 一次性测试脚本，未归档（如需可后续补）。
- 各仓库内 *.bak_* 备份文件（.bak_m0guard/.bak_allocconf/.bak_epfilter/.bak_xferprof/.bak_loadprof/.bak_randdummy/.bak_vhash/.bak_mc2dbg/.bak_mc2probe2/.bak_xferdbg/.bak_xferfusion/.bak_randomize 等）—— 补丁制作过程的备份，待清理。

## 完整 diff 统计（git diff --stat，每仓库每文件）

### /workspace/verl（19 文件，+504/-25；另有 untracked 新文件 run_dsv4_flash_vision_megatron.sh +675、verl/utils/vllm/npu_expert_layout.py +67）

```
 examples/data_preprocess/geo3k.py                  |   2 +-    (+1/-1)
 verl/experimental/agent_loop/agent_loop.py         |  48        (+38/-10)
 verl/experimental/fully_async_policy/shell/dapo_30b_a3b_base_math_fsdp.sh | 2 +- (+1/-1)
 verl/models/mcore/model_forward.py                 |  59        (+58/-1)
 verl/third_party/vllm/__init__.py                  |   2 +-    (+1/-1)
 verl/trainer/config/ppo_megatron_trainer.yaml      |  32        (+32/-0)
 verl/trainer/constants_ppo.py                      |   2 +-    (+1/-1)
 verl/trainer/main_generation_server.py             |  15        (+14/-1)
 verl/trainer/ppo/v1/trainer_base.py                |  14        (+14/-0)
 verl/utils/debug/metrics.py                        |  19        (+19/-0)
 verl/utils/tokenizer/continuous_token_wiring.py    |   5        (+4/-1)
 verl/utils/tokenizer/deepseek.py                   |  13        (+12/-1)
 verl/utils/tokenizer/tokenizer.py                  |  39        (+38/-1)
 verl/utils/vllm/patch.py                           |   3        (+3/-0)  ※纯空白
 verl/workers/engine/megatron/transformer_impl.py   |  96        (+94/-2)
 verl/workers/engine_workers.py                     |   2 +-    (+1/-1)
 verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py | 65 (+63/-2)
 verl/workers/rollout/vllm_rollout/utils.py         | 104        (+102/-2)
 verl/workers/rollout/vllm_rollout/vllm_rollout.py  |   7        (+6/-1)
```

### /workspace/MindSpeed-Bridge（5 文件，+40/-0）

```
 mindspeed_bridge/models/deepseek_v4_flash/csa.py                          |  2 (+2/-0)
 mindspeed_bridge/models/deepseek_v4_flash/csa_fused.py                    |  1 (+1/-0)
 .../modelling_dsv4_hybrid/deepseek_v4_hybrid_attention.py                 |  3 (+3/-0)
 .../deepseek_v4_flash_vision/deepseek_v4_flash_vision_model.py            | 14 (+14/-0)
 .../deepseek_v4_flash_vision/forward_step.py                              | 20 (+20/-0)
```

### /vllm-workspace/main/zyy/vllm-ascend（10 文件，+388/-87）

```
 vllm_ascend/_310p/fused_moe/fused_moe.py        |  60 (+2/-58)
 vllm_ascend/ascend_forward_context.py           |   4 (+4/-0)
 vllm_ascend/models/deepseek_v4/mm_preprocess.py |  72 (+72/-0)
 vllm_ascend/models/deepseek_v4/model.py         |  38 (+34/-4)
 vllm_ascend/models/deepseek_v4/vision.py        |  36 (+26/-10)
 vllm_ascend/models/deepseek_v4/vl_model.py      |  38 (+37/-1)
 vllm_ascend/ops/fused_moe/routed_experts.py     | 157 (+157/-0) ※全部注释/空行
 vllm_ascend/ops/fused_moe/token_dispatcher.py   |  13 (+13/-0)
 vllm_ascend/worker/model_runner_v1.py           |  24 (+19/-5)
 vllm_ascend/worker/worker.py                    |  33 (+24/-9)
```

### /vllm-workspace/main/zyy/vllm（1 文件，+65/-9）

```
 vllm/model_executor/layers/fused_moe/routed_experts.py | 74 (+65/-9) ※未拆分，混合功能
```

### /workspace/TransformerEngineNPU（1 文件，+16/-0）

```
 transformer_engine/pytorch/ops/gemm.py | 16 (+16/-0)
```
