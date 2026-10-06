# LIBERO C³ache 实验分析（2026-09-30）

数据来源：`/mnt/d/Downloads/c3cache/root/gpufree-data/FastWAM-c3cache/evaluate_results/libero/libero_uncond_2cam224_1e-4/`。

本次仅读取已有配置、任务 JSON、汇总 JSON 和 worker 日志，使用 Python 标准库核算，没有运行模型、安装依赖或修改原始结果。

## 结论

关闭 torch.compile 后，`[0,6]、τ=4` 的累计模型推理 wall time 加速为 **1.6824×**，CUDA Event 累计加速为 **1.6825×**，已接近论文相同配置的 **1.69×**。开启编译时，该缓存配置相对编译 baseline 的增益为 **1.3128×**（wall time）。两种编译设置下缓存调度都符合预期，不能将增益差别归因于缓存异常清空。

实际部署中，编译与缓存同时开启仍是四组中最快的配置，平均 CUDA 延迟 **85.844 ms/chunk**；不能为了得到更大的相对加速比而将关闭编译理解为更快。

论文比较依据：[C³ache v1，Table 1](https://arxiv.org/html/2606.08962v1#S4.T1)。该表最高 2.51× 对应的是 `[0,7]、τ=0`，并非本次配置。数值接近不能证明模型版本、硬件或完整评测协议与作者完全一致。

## 四组完整实验

每组均完成 40 个任务、每任务 10 个 episode，共 400 个 episode。

| Run ID | Compile | C³ache | 成功率 | CUDA ms/chunk | CUDA 总秒数 | Wall 总推理秒数 | Chunk 数 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| 20260929_172138 | 开 | 关 | 97.00%（388/400） | 112.277 | 716.215 | 716.359 | 6379 |
| 20260929_180958 | 开 | 开 | 97.25%（389/400） | 85.844 | 545.541 | 545.676 | 6355 |
| 20260929_190049 | 关 | 关 | 96.50%（386/400） | 289.380 | 1872.290 | 1872.428 | 6470 |
| 20260929_200705 | 关 | 开 | 96.75%（387/400） | 172.446 | 1112.794 | 1112.925 | 6453 |

逐行比较 manager_config.yaml：172138 与 190049 仅 compile 开关不同；180958 与 200705 仅 compile 开关不同；190049 与 200705 仅 c3cache_enabled 不同。

共同配置：相同 checkpoint 路径、BF16、seed=42、10 个去噪步、sigma_shift=5.0、replan_steps=10、num_trials=10。action_horizon=null 沿用数据 num_frames=33 对应的 horizon 32。缓存开启时范围 `[0,6]`、τ=4。配置一致不等于对 checkpoint 文件内容作了哈希验证。

## 加速比的正确比较

| 比较 | 每 chunk CUDA 加速 | 累计 CUDA 推理加速 | 累计 wall 推理加速 | 任务总耗时加速（包含仿真等） |
| --- | ---: | ---: | ---: | ---: |
| 编译 baseline → 编译+C³ache | 1.3079× | 1.3129× | 1.3128× | 1.0761× |
| 非编译 baseline → 非编译+C³ache | 1.6781× | 1.6825× | 1.6824× | 1.2407× |

非编译 baseline → 编译+C³ache 的平均 CUDA 延迟加速为 **3.3710×**。这是两项优化合计收益，不能作为 C³ache 单独的加速比。

本目录任务总耗时并非模型推理耗时：关闭编译两组分别为 3873.940 秒与 3122.271 秒。论文中的推理加速不应与包含仿真、视频保存的这一指标混用。

## 缓存调度审计

对两组缓存实验全部 **800 个 episode**，逐一检查：

- 实际 episode 推理次数等于 completed_chunks。
- full_steps + reused_steps = 10 × episode chunk 数。
- reused_steps = 7 × (C − ceil(C/4))。
- 缓存 step 列表等于 `[0,1,2,3,4,5,6]`。

以上检查全部通过。同时，四组实验逐 episode 时间、chunk 数之和与任务统计一致，逐任务 CUDA 时间之和与 summary.json 一致。

| Cache run | 完整 action 堆栈调用 | 跳过 action 堆栈调用 | 实际跳过比例 | 刷新 chunk 数 |
| --- | ---: | ---: | ---: | ---: |
| Compile 开（180958） | 31273 | 32277 | 50.7899% | 1744 |
| Compile 关（200705） | 31735 | 32795 | 50.8213% | 1768 |

实际跳过比例小于无限长序列的 52.5%，是每个 episode 必须从完整刷新开始、长度通常不是 4 的整数倍造成的。两组跳过比例几乎相同。

## 冷启动敏感性分析

所有 run 的首个任务是 libero_10_0。开启编译的 baseline/C³ache 首个 episode 均为 35 个 chunk，总 CUDA 时间分别为 15.461/19.225 秒，平均 441.729/549.280 ms；之后该任务的 episode 平均降至约 110/83 ms。这个现象与冷启动或首次编译影响相符，但没有逐调用 trace 证明全部额外时间来自编译。

仅剔除各 run 第一个 episode，作为敏感性分析（并非正式的稳态 benchmark）：

| 设置 | Baseline CUDA ms/chunk | C³ache CUDA ms/chunk | 加速比 |
| --- | ---: | ---: | ---: |
| Compile 开 | 110.459 | 83.278 | 1.3264× |
| Compile 关 | 289.341 | 172.418 | 1.6781× |

所以开启编译时较小的缓存相对收益并非主要由首个 episode 的额外耗时造成。四组对照表明，编译改变了缓存可节省部分的相对收益；具体是 kernel、CPU 调度、CUDA Graph 还是固定编码开销占比变化，需要分阶段 profiler 才能归因。

## 关闭编译后的分 suite 结果

| Suite | Baseline SR | C³ache SR | Baseline CUDA ms/chunk | C³ache CUDA ms/chunk | 延迟加速 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Spatial | 98% | 97% | 289.705 | 173.949 | 1.6655× |
| Object | 100% | 99% | 289.669 | 173.207 | 1.6724× |
| Goal | 97% | 97% | 289.143 | 173.830 | 1.6634× |
| Long（libero_10） | 91% | 94% | 289.219 | 170.763 | 1.6937× |

关闭编译时，按相同任务/episode 索引配对，有 7 个从失败变为成功、6 个从成功变为失败，净增加 1 次成功。开启编译时相应为 8 个改善、7 个退化。每组仅 400 个 episode、单次运行，不能据 +0.25 个百分点宣称成功率获得统计上可靠的提升。

## 不完整运行与证据边界

171652、171739、171855 没有完整 summary，未纳入比较。前两者日志包含 robosuite.macros_private 缺失相关异常；171855 未见完成标记。四组完整运行均有 worker completed=40 skipped=0，未发现 Traceback，也未发现显式 recompilation/graph-break/cudagraph-skip 日志；没有此类日志不等于排除所有编译行为。

日志未记录 GPU 型号，也没有文本编码、VAE、video prefill、action blocks 的分阶段时间。现有证据支持本配置的相对加速已接近论文，但不足以确认硬件一致或每个模块的真实耗时占比。

## 2026-10-06 补充：为什么 compile 后缓存加速比下降

### 已有数据能够确认的量

这里统一使用每 chunk CUDA 延迟，避免 chunk 数不同混入累计耗时比：

| 量 | Eager | Compile |
| --- | ---: | ---: |
| Baseline ms/chunk | 289.380 | 112.277 |
| C3ache ms/chunk | 172.446 | 85.844 |
| Cache 相对加速 | 1.6781× | 1.3079× |
| Cache 节省 ms/chunk | 116.934 | 26.432 |
| Cache 延迟降低比例 | 40.41% | 23.54% |

加速倍数下降约 **0.3702**。compile 对 baseline 的加速为 **2.5774×**，对 cache 版本为 **2.0088×**。如果简单假设 cache 版本也获得 2.5774× 的整体收益，会预测 66.908 ms/chunk，实际是 85.844 ms/chunk，相差 **18.937 ms/chunk**。这个差值值得调查，但该乘法假设不是必然成立的性能下界，不能直接称为实现损失。

### 为什么两项优化不必相乘

令 F 为缓存不跳过的开销，A 为一次完整 action core 开销，B 为一次复用 core 开销，N 为总步数，M 为实际复用步数，H 为缓存管理和额外拷贝等开销：

\[
T_{\mathrm{base}}=F+NA,\qquad
T_{\mathrm{cache}}=F+(N-M)A+MB+H.
\]

compile 会分别改变 F、A、B、H，并没有理由以相同倍数改变它们。当前 `torch.compile` 只包装 video prefill 和各类 action core；VAE 编码、Python 采样循环、scheduler、缓存身份检查和持久张量 clone 等仍在这些编译函数之外。

假如 compile 特别擅长降低 action core 的 Python 调度、小算子和 kernel launch 开销，那么 C3ache 随后跳过同一个 core 时，能再次省下的成本就变少。即使缓存版本每一次保留下来的完整 DiT 都获得了同样的编译收益，也不能推导出整个缓存版本获得与 baseline 相同的倍数。[PyTorch 文档](https://docs.pytorch.org/docs/2.7/generated/torch.compile.html) 明确说明 `reduce-overhead` 利用 CUDA Graph 降低 Python 开销，且不保证所有调用都适用。

### 一个符合现有数据、但尚未验证的解释

先作很强的简化：复用与管理成本近似为零，每个完整 action step 等价，两组闭环轨迹可比较。采用实际约 50.8% 的跳过比例 s，用 A_total=(T_base-T_cache)/s 反推：

| 反推量（并非 profile 实测） | Eager | Compile |
| --- | ---: | ---: |
| 可跳过 action 相关开销 | 约 230.09 ms | 约 52.04 ms |
| 其他开销 | 约 59.29 ms | 约 60.23 ms |

仅靠“action 相关开销显著压缩、其余约 60 ms 基本不变”就能解释所见加速比变化。这个拟合没有证明 action 真是这些时间，也没有排除 cache 分支额外开销；新增观测将检验而不是预设这个解释。

### 必须区分的候选原因

| 候选原因 | 需要看到的证据 |
| --- | --- |
| 两项优化消除了重叠成本 | baseline 的 action_full 编译收益显著高于非 action 阶段，缓存普通完整步也有相似收益 |
| 冷启动/首次编译 | 慢 chunk 集中在 worker 首调用、各 core 首调用或 compiler counters 增长处；剔除后差距明显缩小 |
| 刷新路径编译效率不同 | 同 step 的 action_refresh 明显慢于 baseline action_full，且不是 residual_clone 单独解释的 |
| 缓存命中路径未高效 replay | action_reuse 的 host/CUDA 时间偏高，配合 perf_hints/编译日志有对应异常 |
| 持久张量保存成本变突出 | residual_clone 或 video_kv_clone 占比明显增大 |
| 缓存身份校验、scheduler、Python 边界限制 | cache_identity、step_setup、scheduler_step 等贡献明显 |
| 路径/shape 变化造成重编译 | 后续 chunk 仍出现新图/guard failure 日志及计数器增长 |

当前 compiled cores 使用 `fullgraph=True`，因此不应默认解释成 core 内部静默 graph break；若不支持整图捕获通常会报错。CUDA Graph 不适用、不同路径的图记录与重编译则是另外的问题。日志或内部 counters 没有异常也不能单独证明高效 replay。[编译排查文档](https://docs.pytorch.org/docs/2.7/torch.compiler_troubleshooting.html)

### 新增代码观测与下一轮对照

已加入默认关闭的分阶段诊断，同时接入 LIBERO/RoboTwin；RoboTwin 另补充默认关闭的 `compile_action_infer` 透传。逐 chunk 保存阶段 host/CUDA 时间、每个去噪 step 的分支类型、缓存周期位置、worker/core 调用序号、编译计数器差值、GPU/PyTorch/CUDA 元数据。计数器为版本相关的 best-effort 信息，不是稳定公共 API。

使用 CUDA Event 的连续边界、chunk 结束一次等待，不在每个阶段之间强制同步。CUDA 指标包含 stream 空闲间隙，host 指标包含提交和阻塞，均不是纯 kernel busy time。诊断会扰动计时，不能替代关闭诊断时的正式加速数字。

具体配置、四组对照命令、过滤与分析脚本见 [推理诊断说明](../docs/INFERENCE_DIAGNOSTICS_zh.md)。先比较全部样本和剔除前若干 worker chunk 的样本，再比较 action_full/action_refresh/action_reuse 的逐 step 成本及非 action 成本。必要时在 GPU 机器进行同输入回放与 kernel trace，进一步区分 CPU launch gaps、融合、CUDA Graph 和设备执行；本地没有执行这些实验。

相关新研究方向记录在 [WAM 缓存研究报告](wam_cache_research_20261006.md)。本补充没有用假定阶段时间替代真实测量。
