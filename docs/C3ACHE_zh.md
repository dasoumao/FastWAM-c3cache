# C³ache 复现说明

这是在当前 FastWAM 仓库上实现的 C³ache 推理路径，依据 [C³ache v1](https://arxiv.org/html/2606.08962v1)。本次只做源码检查和标准库逻辑检查，没有加载权重、运行 GPU 推理、安装依赖或验证论文成功率与加速比。

## 方法与代码的对应关系

默认 `hidden` 方法缓存 **action expert 的整个 DiT 堆栈 residual**：`R = h_L - h_0`。每个去噪 step 有独立缓存，在同一 episode 的后续 action chunk 中复用。命中时使用当前 `h_0 + R`，继续执行输出 head 和 scheduler；只跳过 DiT blocks。图像编码和 video KV prefill 每个 chunk 仍然执行。新增 velocity 实验方法见后文。[论文 §3](https://arxiv.org/html/2606.08962v1#S3)

在本仓库中：

- `ActionDiT.prepare()` 中的 `action_encoder(latents_action)` 产生 `h_0`。
- `MoT.forward_action_with_video_cache_tensor()` 返回 `h_L`。
- `ActionDiT.post()` 把隐藏特征投影为 action velocity。
- `WanContinuousFlowMatchScheduler.step()` 更新 noisy action。

因此 residual 是隐藏特征张量，形状为 `[batch, action_horizon, hidden_dim]`，并非 action 空间里的 velocity 或相邻 sampler 状态之差。默认 action hidden dimension 为 1024。

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
| `c3cache_method` | `hidden` | 缓存方法，见下文 velocity 实验 |
| `c3cache_probe_depth` | `1` | `velocity_probe` 条件起点运行的 action block 数 |
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


## Velocity 实验实现

2026-10-09 已删除旧输出投影模式 `H(h_L)-H(h_0)` 及 `c3cache_residual_space` 接口。下面的新方法使用独立的 `c3cache_method` 参数，旧投影版本的结果不能归入这些新方案。数学定义和公平对照见 [实验计划](../reports/velocity_cache_experiment_plan_20261009.md)。

令 r 为完整刷新 chunk，c 为当前 chunk，k 为去噪循环序号。各模式都按同一 τ 调度，完整刷新时执行全部 N 次真实去噪。命中 chunk 的区别如下：

| `c3cache_method` | 命中时使用的预测/状态 | 区间要求 |
| --- | --- | --- |
| `hidden` | 原 `H(E(x_k)+R_k^r)` | 任意合法连续区间 |
| `velocity_delta` | 真实计算当前 `v_0^c`，之后 `v_0^c+(v_k^r-v_0^r)` | 起点必须为 1 |
| `velocity` | 直接使用 `v_k^r` | 起点必须为 0 |
| `prefix` | 直接从刷新时的 `x_{B+1}^r` 开始真实尾部 | 起点必须为 0 |
| `velocity_virtual` | `b^c+(v_k^r-b^r)`，`b=H(E(x_0))` | 起点必须为 0 |
| `velocity_probe` | `g^c+(v_k^r-g^r)` | 起点必须为 0 |

`velocity_delta` 缓存相对真实首步的累计差值，与同一刷新轨迹的相邻差分 `v_k-v_{k-1}` 连加在实数下等价。它没有第 −1 步，也不缓存旧版的 head 投影 residual。

条件起点 `g` 使用当前 step 0 的 noisy action、timestep、video K/V 与文本/proprio context，运行前 `c3cache_probe_depth` 个 action blocks，再接原 head。这是无需训练的浅层代理实验；原 head 未针对浅层特征训练，不能预设它准确。代理在刷新和复用 chunk 都有计算成本，其 block 数单独统计，不能只看完整 forward 次数判断预算。

`velocity` 和 `prefix` 只在可保证相同初始噪声时复用：固定非空 seed，并保持随机设备、形状、dtype 等设置一致。`seed=None` 或换用自定义 scheduler 时回退完整刷新，不把 fresh noise 当成等价前缀。`velocity_virtual` 在固定初始噪声下退化为直接 velocity 复用，仅作为等价对照；新噪声下其起点变化不代表当前观察校正。

`prefix` 是唯一跳过缓存前缀 scheduler 更新的模式；其他模式仍执行全部 scheduler 步。所有模式每个 chunk 仍计算当前观察编码与 video prefill，供真实尾部或浅层条件起点读取。缓存拥有独立存储，完整推理成功后才提交新轨迹。实数等价不保证 BF16 位级一致，真实张量精度与闭环效果需在服务器验证。

正常调用 FastWAM 的加载权重、设备转换或训练入口会清空缓存；若绕过这些入口直接修改子模块权重，需要显式 `model.reset_c3cache()`。运行时不会为了检测这种外部修改而逐 chunk 遍历全部模型参数。

任务 JSON 的 `episode_c3cache_stats` 新增 `refresh_chunks`、`reuse_chunks`、`fallback_chunks`、`scheduler_skipped_steps`、`probe_steps` 和 `probe_blocks`。`cached_from_chunk` 是当前整套缓存的来源 chunk 编号；`last_chunk_reason` 区分首次/定期刷新、命中、缺失缓存、非固定噪声和不支持的 scheduler。代理 block 次数不包含在 `full_steps` 中，比较预算时需另外计入。

例如 N=10、B=6、τ=4，在前面的评测命令后设置：

```bash
# A1：首步真实，复用 [1,6]，命中 chunk 共 4 次完整 DiT
EVALUATION.c3cache_enabled=true EVALUATION.c3cache_method=velocity_delta \
EVALUATION.c3cache_start_step=1 EVALUATION.c3cache_end_step=6 \
EVALUATION.c3cache_refresh_interval=4

# A2：首步真实，复用 [1,7]，命中 chunk 共 3 次完整 DiT
EVALUATION.c3cache_enabled=true EVALUATION.c3cache_method=velocity_delta \
EVALUATION.c3cache_start_step=1 EVALUATION.c3cache_end_step=7 \
EVALUATION.c3cache_refresh_interval=4
```

A1/A2 的同位置对照只需把 `c3cache_method` 改为 `hidden`。与原 hidden `[0,6]` 比较时，A1 多一次完整调用；A2 调用数相同但真实计算位置不同。

新增 CUDA 总计时和分阶段诊断仍保存在 `experiment/inference-timing` 分支，main 保留原有可选 wall timing。

## 批量 LIBERO 实验

`scripts/run_libero_cache_experiments.py` 使用标准库构造命令，逐组调用原 `experiments/libero/run_libero_manager.py`。默认权重、dataset stats、task 和 `sigma_shift=5.0` 与 README 的 release LIBERO 命令一致；显式固定 N=10、replan_steps=10、seed=42，默认扫描 B=3,5,6,7 和 τ=4，probe 深度 1、2。

```bash
# 预览全部命令，无需模型、Hydra 或 GPU
python scripts/run_libero_cache_experiments.py --num-gpus 1 --num-trials 10 --dry-run

# 全部三条实验线（在 GPU 服务器执行）
python scripts/run_libero_cache_experiments.py --num-gpus 1 --num-trials 10

# 只跑真实首步的质量对照，固定 B=6
python scripts/run_libero_cache_experiments.py --lines quality --ends 6 \
  --num-gpus 1 --num-trials 10 --output-root evaluate_results/cache_quality_b6

# 相同算法分别开/关 compile，并重复三个 seed
python scripts/run_libero_cache_experiments.py --ends 6 --compile both \
  --seeds 42,43,44 --num-gpus 1 --num-trials 10 \
  --output-root evaluate_results/cache_compile_b6
```

| 选项 | 用途 |
| --- | --- |
| `--lines all` | 三条实验线及对照，默认 |
| `--lines quality` | H0/H1/H2、A1/A2 与 baseline |
| `--lines simplify` | H0、直接 velocity、prefix 与 baseline |
| `--lines anchors` | H0、虚拟/浅层条件起点、A1/A2 与 baseline |
| `--ends 3,5,6,7` | 原 hidden 的前缀终点 B；A2/H2 自动使用 B+1 |
| `--taus 4` | 刷新间隔列表，例如 `0,1,4,8` |
| `--probe-depths 1,2` | 浅层条件起点深度列表 |
| `--compile false` | 默认 eager；可选 `true`、`both` |
| `--seed 42` / `--seeds 42,43` | 配对实验种子列表 |
| `--num-trials 50` | 每任务 episode 数，默认 50 |
| `--num-gpus 8` | manager 内部 worker 数，默认 8；各实验组顺序执行 |
| `--task-file PATH` | 用已有 `suite,task_id` 文件缩小任务集 |
| `--ckpt PATH`、`--dataset-stats PATH` | 覆盖 release 权重和归一化统计路径 |
| `--output-root PATH` | 独立实验目录；更换配置请使用新目录 |
| `--continue-on-error` | 某组失败后继续其他组，最终仍以非零状态退出 |

每个 seed/compile 设置只跑一次 baseline。B=6 的全部实验包含 11 组，每组默认四个 suite、40 个任务；每任务 10 次就是 4,400 个 episode。完整扫描会复用实际配置相同的对照，dry run 会给出实际组数和 episode 估算。

批量目录中 `manifest.json` 留存配置、精确 argv、运行状态与结果目录，`summary.csv` 汇总成功率、wall ms/chunk 和缓存计数；每组目录下有 `manager.log`、原 manager 配置和逐任务 JSON。失败重试写新 attempt，不覆盖旧结果。脚本核对任务清单、结果数量和 episode 数，不把缺少任务的运行标记为完成。

汇总的 wall ms/chunk 包含首次推理/编译成本，不能直接称为稳态速度，也不是 CUDA kernel 时间。此脚本未将计时分支的诊断插桩搬回 main；正式性能实验仍需一致预热与 GPU 验收。

本地检查：

```bash
python scripts/check_c3cache.py
python scripts/check_libero_cache_experiments.py
python scripts/check_velocity_cache.py
```
