# WAM 缓存加速研究方案（2026-10-07）

本文记录基于当前 FastWAM 实现的研究建议。代数推导、文献已有结论和待验证假设分别注明；没有运行模型或据此宣称新方法有效。


## 0. 本次分支整理与实现范围

- `experiment/inference-timing` 保存至 `7a49ed6` 的计时版本，包含 CUDA 总计时、分阶段诊断、分析脚本及旧报告；其两个计时提交为 `1486b90`、`7a49ed6`。
- `main` 通过撤销提交 `d3312fc` 移除这两次计时改动，不改写原历史。C3ache 原有可选 wall timing 保留；后加的 CUDA/Event 分阶段观测仅在计时分支。
- 本文件作为新的研究方案保留在 main，汇总之前讨论并明确新增 velocity residual 的定义。
- 本次实现输出空间 residual 缓存；前缀仿射合并、动态误差判别、视觉预测刷新仍属于后续研究，未宣称已经完成。

### 0.1 已实现的 velocity residual 缓存

定义 E 为 action_encoder，H 为输出 head，F 为完整 action DiT 堆栈。第 k 步完整运行时：

```text
h0 = E(x_k)
hL = F(h0, t_k, current_video_KV, current_context)
v_full = H(hL)
R_v[k] = v_full - H(h0)
```

复用 chunk 的对应步：

```text
v_cached = H(E(x_k_current)) + R_v[k]
x_next = scheduler.step(v_cached, delta_k, x_k_current)
```

这是相对于当前 noisy action 的直接线性输出 H(h0) 所缓存的增量，不是缓存整个 velocity，也不是 v_k−v_{k−1} 的相邻步差值。h0 和 residual 仍按原先同一去噪步索引对齐，在 chunk 之间复用。

对于当前仿射 head H(h)=W_h h+b_h，有 R_v=W_h(hL−h0)。必须扣除两次 head 输出以消除 bias，不能直接用 H(hL−h0)，否则会多加 b_h。hidden 模式复用 H(h0_current+R_h)，velocity 模式复用 H(h0_current)+R_v，因此在实数运算下等价；BF16/FP16 的运算位置、减法舍入和存储精度不同，可能产生输出偏差，尤其 cancellation 时。刷新 chunk 直接返回 v_full，不先分解再相加。

两种模式的缓存 shape 分别是 `[B,H,D_hidden]` 与 `[B,H,D_action]`。以 hidden_dim=1024、action_dim=7 为例，单缓存张量元素数减少约 146.3 倍；这是 residual 存储量，不是模型显存或速度改善倍数。velocity 复用仍需 encoder 和 head，刷新还多一次 head，因此实际时延不保证优于 hidden。

配置：`EVALUATION.c3cache_enabled=true EVALUATION.c3cache_residual_space=velocity`。原模式为 `hidden`，默认仍为 hidden。缓存区间、tau、episode reset、video prefill 和全部 scheduler 步骤沿用原规则；切换 residual_space 会清空不兼容的缓存并从刷新 chunk 开始。两种模式共用一份缓存状态，不能混合历史条目。

### 0.2 本阶段验收与后续顺序

1. 标准库假模型检查带 bias 的数学运算、当前 noisy action 依赖、hidden/velocity 多 chunk 等价、tau=0/1/4/8、非零区间起点及空间切换失效；不加载真实模型。
2. GPU 机器固定 checkpoint、观察、proprio、prompt、seed 与 schedule，对比 baseline、hidden、velocity。先 tau=1 验证完整预测，再 tau=4 验证复用。分别检查 FP32 与实际 BF16 误差，不把实数等价当成逐 bit 一致。
3. 使用相同 episode 初始状态比较成功率与 residual 缓存元素数；分阶段性能观测使用计时分支，若要测新机制需把本次机制提交合入计时实验分支，不能直接比较两个实现不一致的分支。
4. 先建立 projected residual 的误差与储存基线，再决定是否继续前缀仿射合并（下文第 3 节）。如速度收益很小，应作为结构基线，不单独主张新的有效加速方法。

## 1. 前提：residual 不是输入无关的常量

当前缓存的是 action expert 整个 DiT 堆栈的残差：

\[
R_k=F_\theta(h_0,t_k,\operatorname{videoKV}(o),\mathrm{context})-h_0.
\]

它依赖 noisy action、观察、任务条件和去噪时刻。`src/fastwam/models/wan22/mot.py` 的 `forward_action_with_video_cache_tensor` 在各层将 action K/V 与 video K/V 拼接，并读取 action context。因此缓存命中时不读取当前条件，是采用历史近似的结果，不是原函数具有条件独立性。

[C3ache](https://arxiv.org/html/2606.08962v1) 观察到平滑执行过程中，相邻 chunk 在对应去噪步骤的 residual 高度相关；这不能直接推成跨任务、跨场景不变性。输入无关也不自动意味着更好的泛化，它可能舍弃应对新观察必需的反馈。

输入相关的方法不一定需要在线动态 gate。可以离线确定条件敏感的步骤或子空间，部署时使用固定分工，避免逐次决策开销。

## 2. 首选科研主线：识别后续去噪能够修正的缓存误差

### 2.1 可检验的解释

早期缓存有效有两个不同解释：一是 residual 变化本来就小；二是 residual 可以变化，但后续真实去噪能修正其中一部分误差。第二种解释允许缓存隐藏特征并不相似、但最终执行动作不敏感的计算。

在第 k 步注入缓存 residual 误差 e_k，当前线性 action 输出头将其转换为一次状态扰动：

\[
\delta x_{k+1}=\Delta_k W_h e_k.
\]

保持后续推理规则固定，最终实际执行的动作扰动可作局部一阶近似：

\[
\delta a_{\mathrm{exec}}\approx S J_{\mathrm{tail},k}\Delta_kW_he_k.
\]

这里 J 是后续去噪的局部 Jacobian，S 选取真正执行的动作并纳入合理的动作尺度。平移、旋转和夹爪需要分别衡量，不能仅依赖混合量纲的一个 L2 值。尾部网络会混合 action tokens，也不能直接认为“未执行的 token 不重要”。

这是局部分析，不是全局误差保证。后续去噪也可能放大误差，尤其在接触和夹爪切换附近。

### 2.2 方法与实验

1. 离线记录少量完整推理轨迹，在不同步骤注入真实跨 chunk residual 差值及控制尺度的人工扰动。
2. 比较 hidden cosine/L2、输出 velocity 误差、最终执行动作误差三类度量的排序关系。
3. 分别分析自由运动、接近目标、接触和抓取阶段，检验是否存在可迁移的低敏感方向。
4. 如果假设成立，再训练或校准一个便宜的决策器，预测最终动作误差。在线只读当前轻量视觉特征、proprio、历史缓存统计，不能先算完整 fresh residual 再判断是否复用。
5. 用离线同输入回放验证误差机制，再用闭环评测验证成功率和外部扰动后的响应能力；前者不能替代后者。

潜在贡献是从特征一致性转向执行动作的一致性，并解释何时可以容忍较大的特征差异。若最终动作误差与 hidden cosine 已高度对应，或 gate 的成本抵消节省，则该方向的主要假设不成立。

误差传播本身已有研究：[ERTACache](https://proceedings.iclr.cc/paper_files/paper/2026/hash/84d395725a9b40cb4a49d84478ac24c7-Abstract-Conference.html) 分析特征偏移和步长放大误差。因此创新不能仅是增加误差阈值，而应落在执行动作敏感度、跨 chunk 条件变化和反馈修正机制。

## 3. 优先实现的结构机会：合并缓存前缀的状态转移

### 3.1 从当前代码可以推导的性质

`action_dit.py` 的输入投影和输出头均为线性层。`fastwam.py::_denoise_action_c3cache_reuse` 执行 `head(action_encoder(x) + residual)`。记：

\[
E(x)=W_ex+b_e,\qquad H(h)=W_hh+b_h.
\]

固定缓存 R_k 后：

\[
v_k=H(E(x_k)+R_k)=Ax_k+d_k,\quad A=W_hW_e,\quad d_k=W_h(b_e+R_k)+b_h.
\]

当前 Euler scheduler 更新为：

\[
x_{k+1}=(I+\Delta_kA)x_k+\Delta_kd_k.
\]

于是连续 m 个缓存命中步骤可以合成为：

\[
x_m=P_mx_0+q_m,
\]

其中可递推计算 P_0=I、q_0=0，P_{k+1}=(I+Δ_k A)P_k，q_{k+1}=(I+Δ_k A)q_k+Δ_k d_k。采用列向量记法，各 action token 使用同一小矩阵，偏置可随 token 和 step 改变。

- P_m 仅依赖模型权重、步长和合并范围，是严格的条件无关部分。
- q_m 含历史 residual，所以仍然依赖历史观察与任务。
- 后续完整去噪继续读取当前条件。

这能跳过缓存前缀内剩余的 embedding、head、scheduler 和 Python 循环，而不只是跳过 DiT。也可以缓存投影后的 W_h R_k，降低存储量；它并非直接缓存 velocity，因为 velocity 仍有依赖当前 x 的 Ax 项。

当前固定 seed 的 `infer_action` 会在每个 chunk 重建相同随机生成器，因而相同 shape 下初始噪声重复。相同缓存前缀的终点也会重复，可进一步研究直接复用前缀终点。注意刷新路径直接使用 h_L，而复用路径重构 h_0+(h_L-h_0)，BF16 舍入会使两者不完全一致。

### 3.2 边界与科研价值

上述等价只针对实数运算下的当前缓存采样路径，不等于原始完整 WAM。改变运算顺序会改变有限精度结果；非线性 head、clip、不同 scheduler 或夹杂完整步时不能直接套用同一个前缀公式。

该实现简洁，但单独的论文贡献可能有限：最重的 DiT 已经被 C3ache 跳过，剩余实际收益必须在 compile 开启时测量。[ActionCache](https://arxiv.org/abs/2607.06370) 已有缓存中间动作并 warm-start 的方法，不能仅把“缓存前缀终点”当作新意。

更有价值的组合是：用合并后的状态转移生成粗动作状态，再依据当前条件下的可修正误差决定真实去噪预算。需要与简单终点缓存、固定步数 warm-start、C3ache 作同条件对照。

## 4. 突破非 action 瓶颈：根据未被预测到的变化刷新视觉条件

当前 FastWAM 每个 chunk 都重新执行 video backbone 的观察 prefill。首先必须分阶段计时确认它的占比，不能从总时间反推成确定事实。

研究假设：已执行动作能够解释的视觉变化，可以通过历史记忆更新来处理；无法解释且影响控制的变化，应触发完整刷新。手臂按计划移动造成的大幅画面变化可能不要求重新规划；物体轻微滑动却可能立即改变动作。

使用轻量预测器：

\[
\hat z_{c+1}=G(z_c,a_{\mathrm{executed}},p_c),\qquad
\nu_c=z^{\mathrm{cheap}}_{c+1}-\hat z_{c+1}.
\]

利用预测误差 ν_c 决定 video prefill、action residual 是否刷新及后续真实去噪预算。需要保留一个读取当前观察的便宜通道，避免历史模型漏掉新物体、外部扰动或指令变化。

FastWAM 当前不在推理时生成未来视频，不能为了判断缓存再额外运行昂贵的视频生成。预测器的计算、数据传输与 gate 成本都必须计入端到端延迟。

已有 [VLA-Cache](https://arxiv.org/abs/2502.02175) 复用静态视觉 token 并关注任务相关信息，[学习式视觉缓存](https://arxiv.org/abs/2602.00686) 也已研究可学习选择。因此新意必须通过实验体现为“动作可解释变化”与“需要反馈的变化”的区分，而不是普通帧差或 token importance。

关键对照包括正常大幅运动、微小目标位移、遮挡、物体滑动和新障碍。若预测误差不能比帧差或周期刷新更好地区分这些情况，应放弃复杂预测器。

## 5. 实验优先级与贡献边界

| 实验 | 控制条件 | 回答的问题 |
| --- | --- | --- |
| 条件干预 | 固定 noisy action 和 step，分别改变观察、文本、proprio | 是否存在条件不敏感部分 |
| 扰动传播 | 固定扰动尺度，改变注入 step 和运动阶段 | 早期缓存有效是否因为后续修正 |
| 度量对比 | 同时记录 hidden、velocity、最终执行动作误差 | gate 应优化什么 |
| 噪声对照 | 固定种子重复噪声、独立噪声，保持其他设置一致 | 跨 chunk 相似性有多少来自噪声重复 |
| 阶段 profile | compile 开启的强 baseline，含 detector 和拷贝成本 | 应优化 action 还是视觉条件 |

优先验证可修正误差，用前缀仿射合并作为简洁实现基础；确认视觉瓶颈后再研究预测误差刷新。不要一开始同时堆叠所有模块，否则难以解释贡献。

数据驱动的模板、投影或阈值必须在独立校准集上确定，并做留任务/留场景验证。固定与动态策略均需报告同一预算下的成功率、平均与 P95 CUDA 延迟、外部扰动响应时间。正式性能比较使用关闭诊断的运行，采样诊断用于机制分析。

“动态 τ”“低秩分解”“跨 chunk 缓存”不能直接作为独立新意：分别已有自适应缓存、[SVD-Cache](https://arxiv.org/abs/2601.07396) 和 [X-Cache](https://arxiv.org/abs/2604.20289) 等相关研究。本文给出的是研究候选与可证伪实验，不是已经证明的新颖性或加速结果。
