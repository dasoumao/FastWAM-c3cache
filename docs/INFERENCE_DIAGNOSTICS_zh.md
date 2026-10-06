# Compile 与 C3ache 推理诊断

目标是定位 compile 后 C3ache 相对加速变小的原因。它是归因工具；正式速度和成功率仍使用诊断关闭的同脚本评测结果。

## 配置与输出

LIBERO/RoboTwin 均可在原有命令后追加：

```bash
EVALUATION.inference_diagnostics_enabled=true \
EVALUATION.inference_diagnostics_every_n_chunks=1 \
EVALUATION.inference_diagnostics_max_chunks=200 \
EVALUATION.inference_diagnostics_cuda_events=true
```

默认关闭。开启后默认记录每个任务前 200 个 chunk，不额外执行 warmup，不重置 residual 或 compiled functions。任务切换会清空本任务记录；worker 调用序号和 core 调用序号继续累计。episode reset 只重置原有策略/cache 状态，诊断保留 episode 标签。

`every_n_chunks=1` 能同时覆盖刷新和命中。不要用与 τ 相同的采样间隔来估计两者的平均占比，否则容易只采到刷新 chunk。有限前缀采样也不是完整 episode 分布的无偏估计。

- LIBERO：原逐任务 `gpu*_task*_results.json` 新增 `inference_diagnostics`，含 metadata、sampling、records 和 summary。
- RoboTwin：任务结果位置增加 `fastwam_inference_diagnostics_<task_config>.jsonl` 与 `.json`。前者增量落盘，后者为任务结束后汇总；不与原 timing JSONL 混用。
- 原 `Inference wall/CUDA` 和成功率输出继续保留。manager 不把采样诊断混入正式时间汇总；跨任务诊断由本文分析脚本生成。
- RoboTwin 新增 `EVALUATION.compile_action_infer`，默认 false 保持原行为；LIBERO 原默认 true 不变。开启前会验证模型接口支持。阶段诊断当前仅支持 base FastWAM 的 infer_action，joint/IDM 不支持时显式报错。

## 新增信号

| 阶段或字段 | 用途 |
| --- | --- |
| input_setup | 输入检查、初始噪声与初始搬运 |
| vae_encode | 当前观察的 VAE 编码 |
| context_prepare | 文本编码或预计算 context 的准备 |
| cache_identity | 缓存任务身份检查，含预计算 context 的 digest |
| proprio_and_cache_signature | proprio 拼接及缓存签名准备 |
| video_prepare | video embedding、调制及 attention mask |
| compile_dispatch_setup | compiled callable 选择/创建；真正首次编译通常发生在 core 首次调用 |
| video_prefill | video DiT 生成观察 K/V |
| video_kv_clone | compiled prefill 输出的持久化 clone |
| schedule_and_cache_lookup | schedule 构建、缓存 begin |
| step_setup | 每步 timestep、分支选择和 cudagraph_mark_step_begin |
| action_full | 普通完整 action core，baseline 和 cache 尾部共同经过 |
| action_refresh | 完整 action core，额外输出 h_L−h_0 |
| residual_clone | residual 持久化 clone |
| action_reuse | embedding + cached residual + head |
| scheduler_step | Euler 更新 |
| output_to_cpu | 最终动作 CPU 输出及随之发生的等待 |
| cache_commit | 更新 Python 缓存状态 |
| worker_chunk_index / core_call_index | 定位 worker 与各 core 首次调用；均从 0 开始 |
| cache_chunk_index / cached_steps_before | 本 episode 缓存周期及缓存内容 |
| compiler_counter_delta | process-global Dynamo/Inductor 数值 counters 的差值，缺失为 null |
| metadata | GPU 名称/容量/计算能力、PyTorch/CUDA/Python、PID、TORCH_LOGS |

记录还保留 image/action/context shape、context stride、dtype 和实际 sigma_shift，用来关联 shape 或布局变化与重编译。metadata 中的 compiled callable 配置只描述启用编译时采用的设置，是否实际开启以逐 chunk 的 `compile_action_infer` 为准。

每个阶段记录 `host_ms` 和 `cuda_ms`，action 阶段额外标记 `step_index`。summary 按 baseline / refresh / reuse / mixed 分组，分别输出均值、P50、P95、每次调用成本与每个采样 chunk 的累计成本。`reuse` chunk 通常仍包含尾部 action_full，不表示所有步骤都被跳过。

阶段采用当前 CUDA stream 的连续 Event 边界，只在采样 chunk 结束读取前等待一次，不逐阶段同步。跨 stream 工作需要在模型返回前 join 当前 stream，与已有 sampler 的计时假设一致。

`cuda_ms` 是 stream 上的 elapsed time，可能包含 GPU 等待 CPU 提交的间隙；不是 kernel 时间总和。`host_ms` 是 host 经过该阶段的时间，含 dispatch/同步阻塞，不是独占 CPU 运算时间。二者会重叠，不能相加，也不能用两者之差直接计算 CPU 开销。阶段 CUDA 时间之和应等于采样 chunk 的 CUDA 时间；阶段 host 时间之和另加 `final_sync_host_ms` 对应 chunk 的 `wall_ms`。

事件记录、编译计数器读取和汇总本身有成本，外层原始 InferenceTimer 会包含其中一部分或全部。内部 chunk span 不包含所有诊断准备/收尾成本，因此无需与外层时间精确相等。开启诊断采样的 run 不能与关闭诊断的 run 直接用于新的性能主张。

## LIBERO 四组诊断命令

先在 GPU 机器用同一任务做较小对照，例如 spatial_0 的 10 个 episode。下面仍使用现有 `eval_libero_single.py`，不新增模型执行入口；本地只检查代码，不运行这些命令。

```bash
run_root=./evaluate_results/compile_cache_diagnostics_20261006
for compile_flag in false true; do
  for cache_flag in false true; do
    python experiments/libero/eval_libero_single.py \
      task=libero_uncond_2cam224_1e-4 \
      ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
      EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
      EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 \
      EVALUATION.num_trials=10 EVALUATION.num_inference_steps=10 \
      EVALUATION.sigma_shift=5.0 \
      EVALUATION.compile_action_infer=$compile_flag \
      EVALUATION.c3cache_enabled=$cache_flag \
      EVALUATION.c3cache_start_step=0 EVALUATION.c3cache_end_step=6 \
      EVALUATION.c3cache_refresh_interval=4 \
      EVALUATION.inference_diagnostics_enabled=true \
      EVALUATION.inference_diagnostics_every_n_chunks=1 \
      EVALUATION.inference_diagnostics_max_chunks=200 \
      EVALUATION.output_dir="$run_root/compile_${compile_flag}_cache_${cache_flag}"
  done
done
```

每轮实验使用新的 run_root。为研究跨任务重编译，也可以继续用原 `run_libero_manager.py`，在原命令后追加相同诊断配置，保持 `MULTIRUN.num_gpus=1`。不要用 manager 的 task_id 限制任务列表；需要限制时使用它已有的 task_file 或 task_suite_names。

这四组闭环轨迹不保证相同，只能做阶段成本对照。若发现异常差异，应在 GPU 机器补充固定观察、文本、proprio、噪声与缓存历史的回放；目前没有新增或运行这种回放。

## 编译日志

针对编译组，可在已有命令前添加以下环境变量；不需要安装包：

```bash
TORCH_LOGS="recompiles,graph_breaks,perf_hints" python experiments/libero/run_libero_manager.py ...
```

`...` 表示保留你的完整原命令参数。manager 子进程继承环境，查看对应 worker log。使用 single 脚本时日志在终端。日志本身也会扰动耗时，建议作为单独诊断运行。

`recompiles` 给出 guard failure 等重编译原因；`perf_hints` 辅助检查 CUDA Graph 不适用等性能提示。`fullgraph=True` 下正常 core 内不应静默拆图。counter/log 没有变化不能证明 CUDA Graph replay 成功；相关计数器是版本相关的私有接口，缺失不会阻止推理。[PyTorch 编译排查文档](https://docs.pytorch.org/docs/2.7/torch.compiler_troubleshooting.html)

## 分析已有诊断文件

分析脚本只使用 Python 标准库，可在无 GPU 的本地运行，不导入真实 torch：

```bash
python scripts/analyze_inference_diagnostics.py \
  --run eager_baseline="$run_root/compile_false_cache_false" \
  --run eager_cache="$run_root/compile_false_cache_true" \
  --run compiled_baseline="$run_root/compile_true_cache_false" \
  --run compiled_cache="$run_root/compile_true_cache_true" \
  --output reports/compile_cache_diagnostics_all
```

另外保留一个剔除每个 worker 前 8 个实际推理 chunk 的敏感性分析版本：在同一命令增加 `--min-worker-chunk 8` 并使用不同 `--output`。可再加 `--exclude-compile-activity`，剔除观测到新图/编译/graph-break counters 增长的样本。此过滤是 best effort，不意味着余下样本一定已稳态，也不会从原始记录中删除数据。

支持目录或单个任务 JSON；RoboTwin 读任务结束后的 `.json`，不同时读取 JSONL，避免重复计数。四个固定标签会校验记录中的 compile/cache 配置，并计算 cache speedup、compile gain、action core 与其余阶段的节省时间。其他标签也可用于单组分析。

输出 `.md` 和 `.json`，包含逐 stage、逐 step、刷新/命中分组，以及前若干 worker 调用与编译计数变化。原始结果中的旧任务 JSON 没有新字段时会明确提示需要新诊断数据，不会伪造阶段时间。

## 如何判断原因

1. 先看 action_full 的 compile 收益是否显著大于 VAE/video/其他部分；这能检验优化收益重叠。
2. 再按同 step 比较 action_refresh 与 baseline action_full，单列 residual_clone；检查是否存在缓存刷新路径的额外成本。
3. 查看 action_reuse 的 host/CUDA 时间、core 首调用和编译日志；不要只看平均 cache 命中率。
4. 对比全部样本、过滤首调用样本和过滤编译活动样本，识别冷启动影响。
5. 对比关闭诊断的正式总时间，确认现象并非观测造成。若需区分 kernel 融合、GPU busy 与 launch gaps，后续必须有 GPU profiler trace，现有 Event 观测不足以直接回答。

代码检查：`python scripts/check_inference_diagnostics.py` 使用假 CUDA Events 检查边界、单次同步、采样上限、任务/worker 序号、异常保留和聚合，不运行模型。`scripts/check_inference_timing.py` 与 `scripts/check_c3cache.py` 保留原有无 GPU 检查。

2026-10-06 本地验证：以上 8+6+6 项检查通过，另用四组人工 JSON 完成分析 CLI 的 dry run，并通过 Python AST、`git diff --check` 和文档命令 `bash -n` 检查。没有运行真实模型、GPU 评测或安装依赖；实际 CUDA Graph 行为与阶段数值仍需 GPU 机器上的新诊断运行确认。
