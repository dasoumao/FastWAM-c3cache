# C³ache 复现说明

这是在当前 FastWAM 仓库上实现的 C³ache 推理路径，依据 [C³ache v1](https://arxiv.org/html/2606.08962v1)。本次只做源码检查和标准库逻辑检查，没有加载权重、运行 GPU 推理、安装依赖或验证论文成功率与加速比。

## 方法与代码的对应关系

默认 `hidden` 模式缓存 **action expert 的整个 DiT 堆栈 residual**：`R = h_L - h_0`。每个去噪 step 有独立缓存，在同一 episode 的后续 action chunk 中复用。命中时使用当前 `h_0 + R`，继续执行输出 head 和 scheduler；只跳过 DiT blocks。图像编码和 video KV prefill 每个 chunk 仍然执行。[论文 §3](https://arxiv.org/html/2606.08962v1#S3)

在本仓库中：

- `ActionDiT.prepare()` 中的 `action_encoder(latents_action)` 产生 `h_0`。
- `MoT.forward_action_with_video_cache_tensor()` 返回 `h_L`。
- `ActionDiT.post()` 把隐藏特征投影为 action velocity。
- `WanContinuousFlowMatchScheduler.step()` 更新 noisy action。

因此默认模式的 residual 是隐藏特征张量，形状为 `[batch, action_horizon, hidden_dim]`，并非 action 空间里的 velocity 或相邻 sampler 状态之差。默认 action hidden dimension 为 1024。新增 `velocity` 模式在输出空间缓存同一堆栈的增量，定义见下文。

**关于新观测：**当前代码的 `h_0` 只编码 noisy action，不直接编码图像或 proprio。图像通过 video KV 的 joint attention 进入 action blocks，文本/proprio 通过 context attention 进入。因此命中的 step 不会重新读取这些条件；保留的完整计算步骤会读取当前条件。实现没有额外添加“把新观测加进 h_0”的运算。

仓库 scheduler 从高噪声向低噪声推进，step delta 为负。实现沿用原 scheduler 的方向与 velocity 定义，不因为论文采用另一种时间记号而改变符号。缓存索引始终是循环顺序 `0, 1, ...`，不是数值 timestep。

## 修改计划与落点

1. 在 `fastwam.py` 的 action-only 推理路径增加 residual 提取和复用，在 Python 层管理跨 chunk 状态。
2. 用 `c3cache.py` 管理 inclusive step 范围、刷新间隔、配置失效和 episode reset。
3. 接入 LIBERO、RoboTwin 的评测参数和 episode 边界；默认关闭缓存。
4. 用标准库检查缓存调度与边界条件，保留服务器上需要完成的数值与性能验收步骤。

训练、checkpoint 参数格式、原有 video KV 缓存的含义保持原样。本功能适用于 `model=fastwam` 对应的 action-only 路径；启用缓存时，不支持的 joint/IDM 评测入口会报错。

## 参数和生命周期

| `EVALUATION` 参数 | 默认值 | 含义 |
| --- | --- | --- |
| `c3cache_enabled` | `false` | 开关；关闭即原始推理路径 |
| `c3cache_residual_space` | `hidden` | `hidden` 或 `velocity`，切换会清空旧缓存 |
| `c3cache_start_step` | `0` | 缓存区间起点，含端点 |
| `c3cache_end_step` | `6` | 缓存区间终点，含端点 |
| `c3cache_refresh_interval` | `4` | τ，按生成的 chunk 数计数 |
| `num_inference_steps` | 沿用评测配置 | 总去噪步数 N；必须大于 end_step |

论文用 K 表示总去噪步数。这里用 N 区分总步数和缓存步数 M：`M = end_step - start_step + 1`。例如 N=10、范围 `[0,6]` 时，命中 chunk 有 7 步跳过 DiT、3 步完整计算，但仍执行 10 次 scheduler 更新。[论文 §3–4](https://arxiv.org/html/2606.08962v1#S4)

| τ | 刷新 chunk（从 0 编号） | 其余 chunk |
| --- | --- | --- |
| 0 | 仅 0 | 复用首个 chunk 的缓存 |
| 1 | 每个 chunk | 不发生复用，可作完整计算对照 |
| 4 | 0、4、8、12… | 复用最近刷新 chunk |
| 8 | 0、8、16、24… | 同上 |

缓存只在当前 episode 内有效。LIBERO 每次 `run_single_episode()`、RoboTwin 每次 policy `reset()` 都清空缓存及 chunk 计数。chunk 计数随 `infer_action()` 调用增加，与环境步数、执行了多少个 action 无关。`replan_steps` 会影响一个 episode 产生多少个 chunk，因而实验对比必须固定它。

直接调用模型时，必须在新 episode 前显式 reset：

```python
model.reset_c3cache()
for observation in episode_observations:
    result = model.infer_action(
        prompt=instruction,
        input_image=observation["image"],
        proprio=observation["proprio"],
        action_horizon=32,
        num_inference_steps=10,
        c3cache_enabled=True,
        c3cache_start_step=0,
        c3cache_end_step=6,
        c3cache_refresh_interval=4,
    )
    # 使用 result["action"] 执行或更新动作队列。
print(model.get_c3cache_stats())
```

## Velocity residual 模式

令 `H = action_expert.post`。刷新时计算：

```python
v_full = H(h_L)
R_v = v_full - H(h_0)
```

刷新步直接使用 `v_full`；后续 chunk 命中对应 step 时使用：

```python
v_cached = H(action_encoder(current_noisy_action)) + R_v
```

`R_v` 的 shape 为 `[batch, action_horizon, action_dim]`。它不是完整 velocity，也不是相邻去噪步的 velocity 差值。每个 step 各存一份，与原 hidden residual 的范围和 tau 调度相同；所有 scheduler 步骤和每 chunk 的 video prefill 仍然执行。

当前 head 为 `H(h)=W_h h+b_h`，故 `R_v=W_h(h_L-h_0)`。不能用 `H(h_L-h_0)` 替代，因为后者多了 bias。当前线性结构下，hidden/velocity 两种缓存复用在实数运算下等价，但 BF16/FP16 的舍入不同，不保证逐 bit 一致。velocity 模式降低 residual 存储维度，不能据此宣称同倍数的显存或时延收益。

在已有 LIBERO 或 RoboTwin 命令后设置：

```bash
EVALUATION.c3cache_enabled=true \
EVALUATION.c3cache_residual_space=velocity \
EVALUATION.c3cache_start_step=0 \
EVALUATION.c3cache_end_step=6 \
EVALUATION.c3cache_refresh_interval=4
```

三组对照仅改变缓存开关/空间：baseline 为 `c3cache_enabled=false`；原 C3ache 为 `c3cache_enabled=true c3cache_residual_space=hidden`；输出空间缓存为 `c3cache_enabled=true c3cache_residual_space=velocity`。其余 checkpoint、编译设置、种子、去噪步数和 replan_steps 保持一致。

直接调用模型时传 `c3cache_residual_space="velocity"`。`get_c3cache_stats()` 和评测结果中的缓存配置增加 `residual_space` 字段。切换模式会从 chunk 0 重新刷新，避免将 hidden_dim 与 action_dim 的缓存混用。

新的研究方案见 [WAM 缓存研究方案（2026-10-07）](../reports/wam_cache_research_plan_20261007.md)。CUDA 总计时、分阶段诊断及其报告保存在 `experiment/inference-timing` 分支；main 只保留原先可选的简单 wall timing，不含后加的诊断插桩。

同一个 model 实例对应一条顺序 episode 流。不要把多个环境的 chunk 交错喂给同一个缓存。外部直接修改模型权重或更换语义上下文时，也应 reset。

## GPU 服务器上的评测命令（本次未执行）

以下命令假设服务器已有 README 所述的依赖、仿真环境、数据、权重。`sigma_shift=5.0` 沿用本仓库 release 权重评测示例，不代表论文明确给出了这个值。

LIBERO 单任务：

```bash
python experiments/libero/eval_libero_single.py \
  task=libero_uncond_2cam224_1e-4 \
  ckpt=./checkpoints/fastwam_release/libero_uncond_2cam224.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/libero_uncond_2cam224_dataset_stats.json \
  EVALUATION.task_suite_name=libero_spatial EVALUATION.task_id=0 \
  EVALUATION.action_horizon=32 EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5.0 EVALUATION.compile_action_infer=false \
  EVALUATION.c3cache_enabled=true \
  EVALUATION.c3cache_start_step=0 EVALUATION.c3cache_end_step=6 \
  EVALUATION.c3cache_refresh_interval=4 EVALUATION.timing_enabled=true
```

LIBERO 全套评测可把入口替换成 `experiments/libero/run_libero_manager.py`，去掉单任务选择，增加 `MULTIRUN.num_gpus=1`；该 manager 会转发缓存配置。每个任务默认 50 个 episode。

RoboTwin 单任务：

```bash
python experiments/robotwin/eval_robotwin_single.py \
  task=robotwin_uncond_3cam_384_1e-4 \
  ckpt=./checkpoints/fastwam_release/robotwin_uncond_3cam_384.pt \
  EVALUATION.dataset_stats_path=./checkpoints/fastwam_release/robotwin_uncond_3cam_384_dataset_stats.json \
  EVALUATION.task_name=click_alarmclock EVALUATION.task_config=demo_randomized \
  EVALUATION.eval_num_episodes=20 \
  EVALUATION.action_horizon=32 EVALUATION.num_inference_steps=10 \
  EVALUATION.sigma_shift=5.0 \
  EVALUATION.c3cache_enabled=true \
  EVALUATION.c3cache_start_step=0 EVALUATION.c3cache_end_step=6 \
  EVALUATION.c3cache_refresh_interval=4 EVALUATION.timing_enabled=true
```

全任务入口为 `experiments/robotwin/run_robotwin_manager.py`，去掉单任务选择，使用 `MULTIRUN.num_gpus=1 MULTIRUN.max_tasks_per_gpu=1` 做单卡时间对比，避免多个评测进程争用一张 GPU。

基线使用相同命令，只改 `EVALUATION.c3cache_enabled=false`。实验网格可以选择 `end_step=6,7` 与 `refresh_interval=0,4,8`；扩展缓存到末尾的消融为 `end_step=8,9`。总步数和 action horizon 保持 10、32。[论文 §4](https://arxiv.org/html/2606.08962v1#S4)

## 检查与后续验收

本地仅运行标准库检查：

```bash
python scripts/check_c3cache.py
git diff --check
```

不要把 `scripts/dryrun_fastwam.py` 当成无 GPU 检查脚本：它会加载完整模型并调用 CUDA。

上服务器后需要分别验证：

1. 固定输入与 seed，对比关闭缓存和 τ=1 的 action 数值，确认完整计算路径一致；FP16/BF16 的容差按实际 backend 设定。
2. 连续生成 chunk，检查缓存命中数、刷新位置和 episode reset；比较 eager 与 compile 路径的数值和持久缓存生命周期。
3. 用相同 checkpoint、种子、scheduler shift、replan_steps、编译设置、硬件和任务集比较成功率与推理时间。

当前评测代码每次 `infer_action()` 都用同一个 `seed` 重建 generator，因此非空 seed 会重复使用相同的初始 action noise。本实现保留这个行为。若研究 fresh noise 的影响，应给基线和缓存组采用相同的随机数策略。

LIBERO 默认开启 `torch.compile`，本说明的起始验收命令先用 eager。编译成本和 warmup 需要单独记录；开启编译后基线与缓存组必须使用相同设置。本次没有验证 CUDA Graph 或真实张量精度。

评测程序运行总时长包括环境交互、视频保存等，不能直接作为模型 inference speedup。应分别报告 `sum(baseline inference seconds) / sum(cache inference seconds)`、每 chunk 延迟和成功率；成功率或 episode 长度变化会影响总推理调用数。

设置 `EVALUATION.timing_enabled=true` 后，LIBERO 的任务结果 JSON 含 `inference_seconds`、`inference_chunks` 及逐 episode 数组；开启缓存时另有 `episode_c3cache_stats`。RoboTwin 在该任务的 `eval_output_dir` 下写 `fastwam_inference_timing_<task_config>.jsonl`，每行记录一个 chunk 的时间、episode/chunk 编号、缓存配置和统计。求总时间应累加每行 `infer_s`，不要累加已经累计过的 `cumulative_infer_s`；重跑时使用新的输出目录，避免把追加记录算入旧结果。计时覆盖模型调用直至 action 返回 CPU，不含外围图像预处理、action 反归一化或指标写盘。

如果 episode 有 C 个 chunk，缓存 M/N 个 step，τ>0 时刷新次数为 `ceil(C/τ)`；τ=0 时为 1。忽略提前失效时，完整 action DiT 调用次数为 `refresh_count*N + (C-refresh_count)*(N-M)`。这个计数用于验证调度，不能当成端到端加速比，因为编码、prefill、head 和 scheduler 仍有开销。
