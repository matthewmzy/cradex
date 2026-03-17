# CRA 项目完善指南：Idea 总结 + LLM Prompt + 实验规划

---

## 〇、代码审查 & 课题反思（Code Review & Critical Reflection）

> **本节由代码全面审查后生成，指出核心问题并给出改进方向。**

### A. 课题层面的反思

#### A1. 轴独立性假设的脆弱性（核心风险）

CRA 的理论优势建立在"参数轴可以被解耦适应"的假设上。但物理参数之间**天然存在耦合**：
- **gravity_dir 与 gravity_mag** 是同一个力的两个分量，不是独立参数
- **friction 与 mass** 耦合：摩擦力 = μ·N，法向力 N 取决于 mass 和 gravity
- **kp/kd 与 mass** 耦合：同一个 PD 控制器对不同质量物体的响应完全不同

**改进方向**：
1. 合并 gravity_dir + gravity_mag 为一个 stage（3D gravity vector adapter），减少不必要的分解
2. 在论文中明确讨论"弱耦合"假设条件，承认强耦合参数不应被分解
3. 增加实验验证：对比 gravity 合并 vs 分开 的性能差异
4. 考虑增加"耦合度分析"实验：测量各轴 adapter latent 之间的互信息

#### A2. 组合泛化的理论漏洞

原文声称"每个 adapter 独立响应 → 天然支持组合外推"，但这并不严格成立：
- Stage N 的 encoder 的**输入是 observation history**，而 observation 包含**所有轴变化的综合效应**
- 例如 friction adapter 看到的 obs 同时受到 gravity/mass 变化的影响
- 解耦是**训练信号层面**（frozen cascade 使得每个 stage 只能补偿新引入的变化），但**信息层面**并不解耦

**改进方向**：
1. 更严谨的表述：CRA 的组合泛化来源于**"每个 stage 学到的是条件残差补偿"**，而非信息层面的独立性
2. 需要用 linear probe 实验（已规划的实验 4b）来**验证**而非假设解耦，如果 R² 不理想需诚实报告
3. 增加理论分析：在什么条件下 additive residual 结构是最优的（如线性近似时）

#### A3. 公平性问题

| 方法 | 参数量 |
|------|--------|
| CRA | ~1.3M |
| Full DR | ~0.5M |
| RMA | ~0.8M |

CRA 参数量是 Full DR 的 2.6 倍。需要增加：
- **Equal-capacity Full DR baseline**：1.3M 参数的单一大网络，证明优势不来自容量
- **Equal-capacity RMA baseline**：1.3M 参数的 RMA，使用更大的 encoder
- **CRA-Small**：减小 adapter 使得总参数量与 Full DR 持平（~0.5M）

#### A4. 训练效率质疑

CRA 的 cascaded training 需要 5000 + 4×3000 = 17000 iterations（vs Full DR/RMA 的 10000 iterations）。
虽然总 timestep 预算可以对齐（通过调整每 iter 的步数），但 cascaded 天然就比单阶段慢：
- 每个 stage 不能并行训练
- 如果某个 stage 收敛慢会成为瓶颈

需要报告 wall-clock 时间对比，不要回避这个劣势。

### B. 代码层面的关键问题

#### B1. [CRITICAL] 逐环境重力 DR 实现是错误的

`shadow_hand_rotation.py:553-561` — IsaacGym 的 gravity 是**全局参数**，不能逐环境设置。当前实现让所有 env 使用第一个被 reset 的 env 的重力值，这意味着**gravity DR 实际上不工作**。

**修复方案**：将 gravity variation 实现为施加在物体上的**外力**（external force on rigid body）：
```
F_pseudo_gravity = (g_desired - g_global) * m_object
```
每 step 通过 `apply_rigid_body_force_tensors()` 施加。

#### B2. [BUG] RMAPolicy.evaluate_actions 重复调用 encoder

`baselines.py:139` 的 `self.forward()` 内部已经计算了 z 和 value，但 `baselines.py:143` 又重新计算了一次 encoder。修复：直接复用 forward 返回的 value。

#### B3. [PERF] HistoryBuffer.push() 的 clone() 是多余的

`rollout_buffer.py:55-56` — `.clone()` 浪费 GPU 内存。PyTorch 的赋值对重叠内存是安全的。更好的做法：使用 circular buffer 索引避免数据移动。

#### B4. [ISSUE] set_dof_state_tensor_indexed 使用了错误的索引

`shadow_hand_rotation.py:522-527` 使用 `hand_indices`（actor index），但 API 需要 DOF index。

#### B5. [MISSING] 缺少观测归一化

当前代码没有 observation normalization（running mean/std），对 RL 训练稳定性至关重要。

#### B6. [MISSING] 缺少 Curriculum DR baseline

实验对比表中列出了 "Curriculum DR" 但代码中没有实现。

### C. 实验设计改进

#### C1. 组合泛化实验需更公平

Full DR 和 RMA 应该用**连续分布的 DR**（标准做法）训练，不用 sparse grid。CRA 用 staged axis training。然后在 dense grid 上统一评估。

#### C2. 预期数值调整

原文预期过于乐观。调整为：
- CRA unseen 组合成功率：~55-60%（原 68%）
- CRA 泛化 gap：~15-18pp（原 8pp）
- 仍应显著优于 baseline

#### C3. 增加必要的新 baseline

1. **Large FullDR**：参数量对齐（~1.3M）
2. **Large RMA**：参数量对齐（~1.3M）
3. **Sequential DR Curriculum**：逐轴扩大 DR 范围但用单一网络
4. **CRA-MergedGravity**：合并 gravity_dir + gravity_mag 为一个 stage

---

## 一、Idea 最终总结

### 题目

**CRA: Cascaded Residual Adaptation for Robust Dexterous Manipulation via Axis-Decomposed Specialization**

### 一句话总结

将灵巧操作的 domain adaptation 沿物理参数轴分解为多个级联冻结的残差适应模块，每个模块仅从历史交互中学习辨识和适应**一个**参数轴的变化，实现解耦适应、组合泛化、并行推理。

### 动机链

1. **灵巧手的真正灵巧性需要 in-hand manipulation**（旋转、工具使用等），而非简单的末端抓取
2. **异构本体遥操作几乎不可能**为灵巧手采集高质量 demonstration（精度不足或 gap 太大）
3. **RL 在仿真中自主训练**是获取灵巧操作数据的唯一可规模化路径
4. **全域随机化（Full DR）的根本问题**：单一 adaptation module 必须从混合信号中同时辨识所有物理参数，这是一个指数复杂的联合辨识问题 → 导致策略保守、组合外推失败
5. **CRA 的核心洞察**：参数轴可以被解耦适应。强制每个模块只学一个轴，从"物体是否打滑"学摩擦，从"物体如何加速"学重力 —— 单轴辨识比多轴联合辨识简单得多

### 方法

```
Training（顺序）:
  Stage 0: 训练 base policy π₀，固定参数或极窄 DR
  Stage i = 1..N:
    1. Enable DR axis i（同时保持 axis 1..i-1 的 DR active）
    2. Freeze π₀ 和前 i-1 个 stage
    3. 新建 AdaptationEncoder Aᵢ (GRU, history→zᵢ) + ResidualHead ρᵢ (MLP, (obs,zᵢ)→δaᵢ)
    4. 用 PPO 训练 Aᵢ + ρᵢ 直到收敛
    5. Checkpoint

Inference（并行）:
  a = clip( π₀(o) + Σᵢ ρᵢ(o, Aᵢ(history)),  a_min, a_max )
  所有 Aᵢ 和 ρᵢ 并行计算（additive structure，无数据依赖）
```

### 理论优势

1. **解耦适应**：每个模块被结构性地迫使只学一个轴 → 可解释、可分析（t-SNE、linear probe）
2. **组合泛化**：训练时只覆盖 sparse 的参数组合，测试时 unseen 的 joint combination 也能处理（因为每个模块独立/条件独立地处理自己的轴）
3. **简化的学习问题**：每个 stage 只需单轴适应，收敛更快更稳定
4. **并行推理**：additive 结构，所有模块同时计算
5. **模块化复用**：gravity adapter 可能跨任务迁移（in-hand rotation → grasp transition 复用同一个 gravity module）

### 与已有工作的关键区分

| 工作 | CRA 的区别 |
|------|-----------|
| Residual Policy Learning (Silver 2018) | CRA 是多级 cascaded + 每级绑定特定参数轴；RPL 只有单层 residual |
| RMA (Kumar 2021) | CRA 分解成多个专用 encoder；RMA 用单一 encoder 辨识所有参数 |
| Progressive Networks (Rusu 2016) | PN 用于跨任务；CRA 用于同一任务的域适应，且是 additive 而非 lateral connections |
| DexNDM | 多专家蒸馏；CRA 是 cascaded freeze + residual，不需蒸馏 |
| Full DR + Curriculum | 单网络逐步扩大 DR 范围；CRA 是多网络 + 轴分解 + frozen cascade |

---

## 二、LLM Prompt（用于后续代码完善）

以下 prompt 可以直接交给 Claude / GPT-4 / 其他 coding agent，让它基于现有代码将项目完善为可投稿状态。

---

```markdown
# CRA Project Completion Prompt

## Context

You are working on a research project called **CRA (Cascaded Residual Adaptation)** for
dexterous manipulation. The complete prototype codebase is already written in `cra_dexterous/`.
Your job is to refine, complete, and harden it into a submission-ready research project
targeting a top robotics venue (CoRL / RSS / ICRA).

## Current Codebase (already implemented)

```
cra_dexterous/
├── cra/
│   ├── models/           # Neural network architectures
│   │   ├── base_policy.py          # ActorCritic MLP (Stage 0)
│   │   ├── adaptation_encoder.py   # GRU/Conv1D history→latent encoder
│   │   ├── residual_head.py        # Per-stage residual MLP (near-zero init)
│   │   ├── cra_policy.py           # Full CRA composite: staged freeze, additive inference
│   │   └── baselines.py            # FullDRPolicy, RMAPolicy
│   ├── algo/             # RL algorithm
│   │   ├── ppo.py                  # PPO (clipped surrogate, GAE, value clipping)
│   │   └── rollout_buffer.py       # GPU buffer + HistoryBuffer for adaptation window
│   ├── envs/             # IsaacGym environments
│   │   ├── base_env.py             # Abstract IsaacGym env with PhysX setup
│   │   ├── shadow_hand_rotation.py # Shadow Hand in-hand rotation (24DOF, 118-D obs)
│   │   ├── rewards.py              # Quaternion rotation reward, drop penalty, action penalty
│   │   └── axis_dr.py              # AxisDRManager with 8 configurable axes
│   ├── trainer/          # Training loops
│   │   ├── cra_trainer.py          # Multi-stage CRA pipeline (base→axis1→axis2→...)
│   │   └── baseline_trainer.py     # Single-stage for FullDR / RMA
│   └── utils/            # Logging, checkpointing, analysis
│       ├── logger.py, checkpoint.py
│       └── analysis.py             # t-SNE, linear probes, ablation study
├── scripts/              # Entry points (train.py, eval.py, download_assets.py, visualize_adaptation.py)
├── configs/              # YAML configs for task + method
├── README.md, requirements.txt, pyproject.toml
```

The codebase compiles and all model/algo imports pass functional tests.
The IsaacGym environment (`shadow_hand_rotation.py`) is structurally complete but has NOT
been end-to-end tested with an actual IsaacGym installation.

## What Needs To Be Done

Please complete the following tasks **in priority order**. Each task should be a separate
commit with clear description.

### Priority 1: Environment Correctness & End-to-End Training

1. **Fix and validate `shadow_hand_rotation.py`**:
   - Test with IsaacGym Preview 4, fix any API mismatches
   - Verify DOF indexing (the Shadow Hand MJCF may have different joint ordering)
   - Verify fingertip body name lookup works (provide fallback)
   - Validate that `_apply_dr_params` correctly modifies simulation parameters:
     - Gravity: IsaacGym gravity is global, but per-env gravity variation needs
       workaround (apply as an external force on the object instead)
     - Object mass/friction: verify per-actor property setting
   - Add a `--render` flag for visual debugging
   - Write a `test_env_standalone.py` that creates the env, runs 100 random steps,
     and prints obs/reward statistics

2. **End-to-end training smoke test**:
   - Run CRA training for 50 iterations (base) + 20 iterations (1 stage) on 256 envs
   - Verify: rewards increase, checkpoints save/load correctly, TensorBoard logs work
   - Fix any shape mismatches or training instabilities

3. **Per-environment gravity workaround**:
   IsaacGym gravity is a global sim parameter. For per-env gravity direction DR,
   implement one of:
   - (a) Apply gravity as an external force on the object rigid body (preferred)
   - (b) Use different sim instances (heavy, not recommended)
   - (c) Use IsaacLab which supports per-env gravity via scene.set_gravity()
   Document the approach in code comments and README.

### Priority 2: Second Task (Grasp Transition)

4. **Implement `shadow_hand_grasp_transition.py`**:
   - Task: transition grasp pose from one functional grasp to another on the same object
   - Start with object in grasp A (e.g., power grasp), goal is grasp B (e.g., precision grasp)
   - Reward: distance between current finger contact pattern and target contact pattern
   - This is the secondary task to demonstrate CRA's generality

### Priority 3: Experiment Infrastructure

5. **Compositional generalization evaluation script** (`scripts/eval_compositional.py`):
   - Train with a **sparse** grid of parameter combinations
   - Evaluate on a **dense** grid including unseen combinations
   - Report: success rate heatmap over the parameter grid
   - This is THE key experiment for the paper

6. **Multi-seed benchmark script** (`scripts/benchmark.py`):
   - Run CRA + FullDR + RMA with seeds {0,1,2,3,4}
   - Collect: reward curves, success rates, wall-clock time
   - Output: aggregated results CSV + learning curve plots with std bands

7. **Axis ordering ablation** (`scripts/ablation_ordering.py`):
   - Run CRA with all permutations of 4 axes (or a meaningful subset)
   - Compare final success rates
   - Output: table + bar chart

8. **Cross-task transfer experiment** (`scripts/eval_cross_task.py`):
   - Train gravity/friction adaptation modules on in-hand rotation
   - Freeze those modules, attach to a new base policy for grasp transition
   - Compare: full CRA from scratch vs. transferred modules + new base

### Priority 4: Robustness & Polish

9. **WandB integration**: Add `--logger wandb` option alongside TensorBoard

10. **Proper observation normalization**: Add running mean/std normalization
    (VecNormalize-style) to observations, critical for RL stability

11. **Asymmetric actor-critic**: The critic can receive privileged information
    (ground-truth DR parameters) during training. Add `--privileged-critic` flag.
    This is standard in sim2real dexterous manipulation.

12. **Object geometry axis**: Implement the `object_scale` and `object_geometry` DR axes.
    For geometry: load multiple YCB objects, randomly select one per env on reset.
    Add an `object_geometry` CRA stage that adapts to shape variation.

13. **Data export for IL**: Add `scripts/collect_demos.py` that runs a trained CRA policy
    and saves (obs, action) trajectories in a standard format (HDF5 or zarr) for
    downstream imitation learning.

### Priority 5: Paper-Ready Figures & Tables

14. Add `scripts/plot_results.py` that generates:
    - Learning curves (reward vs. timestep) with std bands for CRA/FullDR/RMA
    - Compositional generalization heatmap (success rate over param grid)
    - t-SNE of each stage's latent colored by DR param
    - Ablation bar charts (axis ordering, module disabling)
    - All figures should be publication-quality (LaTeX-friendly fonts, proper labels)

## Architecture Notes for the Developer

- **Additive residual structure**: `a = π₀(o) + Σᵢ ρᵢ(o, Aᵢ(history))`.
  All residuals computed in parallel. No sequential dependency at inference.
  Residuals are unbounded; only the final action is clipped to joint limits.

- **Frozen cascade**: At stage i, π₀ and stages 1..i-1 have `requires_grad=False`.
  Only stage i's encoder+residual+critic are trained. The frozen stages form part
  of the "environment dynamics" from stage i's perspective → stationary MDP.

- **History buffer**: `HistoryBuffer` maintains a sliding window of (obs, action) pairs.
  On env reset, the corresponding rows are zeroed. Window size = 50 steps (~1s at 50Hz).

- **Near-zero residual init**: ResidualHead output layer has weights ~ U(-0.01, 0.01).
  New stages initially contribute ~0 action, ensuring smooth stage addition.

- **Axis DR order**: The recommended order is by single-axis reward-drop.
  Stage 1 (trained with simplest context) should handle the most impactful axis.

## Code Quality Requirements

- Type hints on all public functions
- Docstrings (Google style) on all classes and public methods
- No dead code, no placeholder TODOs in final version
- All configs should be expressible in YAML (no hardcoded magic numbers)
- Use `@torch.no_grad()` properly during evaluation
- GPU memory: all tensors should be on the correct device; avoid CPU↔GPU transfers
```

---

## 三、实验规划与预期效果

### 实验 1: 训练分布内性能对比（Table 1）

**设置**：
- 任务：Shadow Hand in-hand cube rotation
- 4096 parallel envs，50Hz 控制
- DR 覆盖：gravity direction (全球面), gravity magnitude (7-12 m/s²), object mass (20-500g, log-uniform), friction (0.3-2.0)
- 训练预算统一为 ~2×10⁸ total timesteps

**方法对比**（含参数量对齐 baseline）：

| 方法 | 训练方式 | 网络参数量 | 备注 |
|------|---------|-----------|------|
| CRA (ours) | Stage 0: 5000 iter + 3 stages × 3000 iter | ~1.1M | gravity合并为一个stage |
| Full DR | 单阶段，全 DR | ~0.5M | 标准 |
| Full DR-Large | 单阶段，全 DR，大网络 | ~1.1M | 参数量对齐 |
| RMA | 单阶段，全 DR + GRU encoder | ~0.8M | 标准 |
| RMA-Large | 单阶段，全 DR + 大 encoder | ~1.1M | 参数量对齐 |
| Curriculum DR | 单阶段，逐轴扩大 DR 范围 | ~0.5M | 逐轴 curriculum 但单网络 |
| CRA-NoFreeze | 同 CRA 结构但不 freeze 前级 | ~1.1M | 消融：验证 freeze 的必要性 |

**注意**：gravity_dir + gravity_mag 合并为一个 gravity stage（3D vector adapter），因此实际为 3 个 adaptation stage 而非 4 个。

**预期结果**：

| 方法 | 成功率 (↑) | 平均旋转误差 (↓) | 连续成功次数 (↑) |
|------|-----------|-----------------|----------------|
| Full DR | ~65% | ~0.35 rad | ~12 |
| Full DR-Large | ~67% | ~0.33 rad | ~13 |
| RMA | ~72% | ~0.28 rad | ~18 |
| RMA-Large | ~73% | ~0.27 rad | ~19 |
| Curriculum DR | ~70% | ~0.30 rad | ~16 |
| CRA-NoFreeze | ~71% | ~0.29 rad | ~17 |
| **CRA (ours)** | **~76%** | **~0.24 rad** | **~22** |

**预期**：CRA 在训练分布内领先约 3-8pp，Large baseline 应缩小但不消除差距，证明结构优势而非容量优势。核心优势在实验 2。

---

### 实验 2: 组合泛化（Table 2 + Figure 3, **核心实验**）

**设置**：
- 训练时：所有方法使用**连续分布 DR**（标准做法），CRA 按 staged axis training
- 测试时：预定义 dense grid — gravity 8 个方向 × mass 5 个值 × friction 5 个值 = 200 种固定组合
- 区分 "seen-like"（参数值落在训练分布高密度区域）和 "unseen-like"（参数值在训练分布边缘或插值区域）
- **额外对比**：sparse grid training → dense grid（原设计），作为附加实验

**预期结果**（保守估计）：

| 方法 | Seen-like 组合成功率 | Unseen-like 组合成功率 | 泛化 gap |
|------|---------------------|----------------------|---------|
| Full DR | ~62% | ~35% | 27pp |
| Full DR-Large | ~64% | ~38% | 26pp |
| RMA | ~70% | ~48% | 22pp |
| RMA-Large | ~71% | ~50% | 21pp |
| **CRA (ours)** | **~74%** | **~58%** | **16pp** |

**关键图表**：2D heatmap，x 轴 = friction，y 轴 = mass，颜色 = 成功率。CRA 的 heatmap 应该均匀绿色，Full DR 在 unseen 区域大片红色。

**为什么 CRA 在这里显著领先**：CRA 的每个 adapter 独立处理一个轴且在训练时各轴 DR 已经活跃，所以新的 joint 组合只是各 adapter 的独立响应的叠加 — 天然支持组合外推。Full DR/RMA 在 joint distribution 上训练，unseen joint 组合是 OOD。

---

### 实验 3: 轴排序消融（Table 3）

**设置**：测试 4! = 24 种排列中的 6 种代表性排列

| 排列 | Stage 1→2→3→4 | 最终成功率 |
|------|---------------|-----------|
| Ours (推荐) | grav_dir → grav_mag → mass → friction | ~78% |
| 反序 | friction → mass → grav_mag → grav_dir | ~72% |
| 随机序 1 | mass → grav_dir → friction → grav_mag | ~75% |
| 随机序 2 | friction → grav_dir → mass → grav_mag | ~74% |

**预期**：按 reward-drop 排序（影响最大的轴先适应）最优，但差异在 6pp 以内 — CRA 对排序有合理的鲁棒性。

---

### 实验 4: 模块解耦分析（Figure 4）

**子实验 4a: t-SNE**

对每个 stage 的 latent zᵢ 做 t-SNE，用对应的 DR 参数值着色：
- gravity module 的 latent 按 gravity 方向聚类 ✓
- friction module 的 latent 按 friction 值聚类 ✓
- 交叉检查：gravity module 的 latent 按 friction 值着色应**不聚类** → 证明解耦

**子实验 4b: Linear probe R²**

| Adapter | 预测自己轴的 R² | 预测其他轴的 R² |
|---------|----------------|----------------|
| Gravity adapter → gravity param | **0.92** | 0.05-0.12 |
| Mass adapter → mass param | **0.88** | 0.08-0.15 |
| Friction adapter → friction param | **0.85** | 0.06-0.10 |

对角线 R² 高、非对角线低 → 解耦适应的定量证据。

**子实验 4c: 消融（disable 单个 stage）**

| 配置 | 成功率 |
|------|--------|
| All active | 78% |
| Disable gravity | 35% (↓43pp) |
| Disable mass | 68% (↓10pp) |
| Disable friction | 60% (↓18pp) |

→ 每个 module 的贡献和该轴的实际影响成正比。

---

### 实验 5: 跨任务迁移（Table 4, Bonus）

**设置**：
- 在 in-hand rotation 上训练完整 CRA (4 stages)
- 新任务：grasp transition
- Transfer：冻结 gravity + friction adapter，只训练新 base policy + mass adapter

| Approach | Grasp transition 成功率 | 训练时间 (相对) |
|----------|----------------------|---------------|
| CRA from scratch | ~65% | 1.0× |
| Full DR | ~58% | 1.0× |
| **CRA with transfer** | **~63%** | **0.4×** |

→ 迁移后性能接近从头训练，但训练成本大幅降低。物理适应模块确实是 task-agnostic 的。

---

### 实验 6: 数据采集质量（Table 5）

**设置**：
- 用 CRA / FullDR / RMA 策略各采集 100K 步 demonstration
- 用这些 demo 训练视觉策略（ACT 或 Diffusion Policy）
- 评估下游策略在仿真中的成功率

| Demo 来源 | 下游 IL 策略成功率 |
|----------|------------------|
| Full DR demos | ~42% |
| RMA demos | ~48% |
| **CRA demos** | **~55%** |

→ CRA 策略更稳定、更一致 → 生成的 demo 质量更高 → 下游 IL 表现更好。

---

### 实验 7: Sim-to-Real（如果条件允许）

- 硬件：Allegro Hand 或 LEAP Hand + 物体追踪（AprilTag 或 RGB-D 6D 估计）
- 任务：in-hand cube rotation
- 方法：直接部署 CRA 策略（privileged state → teacher） → 蒸馏到 vision-based student
- 对比：CRA teacher vs. Full DR teacher → student 的成功率

这个实验为 optional，如果有硬件则做，没有则标注 future work。

---

### 论文结构预览

```
1. Introduction
   - 灵巧手操作的数据获取瓶颈
   - 仿真 RL 为什么是唯一可规模化路径
   - Full DR 的联合辨识问题 → CRA 的动机

2. Related Work
   - In-hand manipulation RL (DeXtreme, AnyRotate, HORA, DexNDM)
   - Domain randomization & adaptation (RMA, RAPP)
   - Residual policy learning (Silver 2018, Johannink 2019)
   - Modular / compositional policy learning

3. Method
   3.1 Problem formulation
   3.2 Axis-decomposed domain randomization
   3.3 Cascaded frozen residual architecture
   3.4 Staged training algorithm
   3.5 Parallel inference

4. Experiments
   4.1 Setup (hand, objects, DR axes, baselines)
   4.2 In-distribution performance (Exp 1)
   4.3 Compositional generalization (Exp 2, CORE)
   4.4 Axis ordering ablation (Exp 3)
   4.5 Disentanglement analysis (Exp 4)
   4.6 Cross-task transfer (Exp 5)
   4.7 Data generation for IL (Exp 6)

5. Discussion & Limitations
   - 轴的选择和划分仍然是手工的
   - 当参数轴强耦合时 CRA 可能退化
   - 总训练时间略多于 Full DR

6. Conclusion
```

投稿目标：**CoRL 2026** 或 **RSS 2026**（8 pages + references）

---

## 四、已实现的改进（Implementation Log）

> **本节记录基于上述反思和 Prompt 已完成的代码改进。**

### Phase 1: 核心 Bug 修复 & 架构改进

#### 1. 合并 gravity 轴（A1 改进）
- **文件**: `cra/envs/axis_dr.py`
- 将 `gravity_dir` + `gravity_mag` 合并为统一的 `gravity` 轴
- 采样方式: `direction = normalize(randn) × uniform(7, 12)` → 3D gravity vector
- `get_gravity_vectors()` 支持统一和 legacy 两种路径
- `configs/method/cra.yaml` 更新为 3 个 stage: `["gravity", "object_mass", "friction"]`

#### 2. 修复 per-env 重力 DR（B1 修复）
- **文件**: `cra/envs/shadow_hand_rotation.py`, `cra/envs/base_env.py`
- IsaacGym 重力是全局参数，不能逐 env 设置
- 实现外力 workaround: `F_pseudo = (g_desired - g_global) * m_object`
- 通过 `apply_rigid_body_force_tensors()` 每 step 施加
- `base_env.py` 添加 `_apply_external_forces()` hook

#### 3. 修复 RMA 双重 encoder 调用（B2 修复）
- **文件**: `cra/models/baselines.py`
- `evaluate_actions()` 不再重复调用 encoder，直接复用 `forward()` 返回的 value

#### 4. 移除 HistoryBuffer 多余 clone（B3 修复）
- **文件**: `cra/algo/rollout_buffer.py`
- 移除 `push()` 和 `get()` 中的 `.clone()`

#### 5. 修复 DOF 索引 bug（B4 修复）
- **文件**: `cra/envs/shadow_hand_rotation.py`
- `set_dof_state_tensor_indexed` 需要 DOF index 而非 actor index
- 新增 `hand_dof_indices = hand_indices * num_hand_dofs`
- 同时计算 `object_body_indices` 用于外力施加

#### 6. 添加观测归一化（B5 修复）
- **新文件**: `cra/utils/obs_normalizer.py`
- `ObsNormalizer(nn.Module)` 实现 Welford 在线均值/方差估计
- 支持 `clip_range` 截断、`train()`/`eval()` 模式切换
- 集成到 `cra_trainer.py` 和 `baseline_trainer.py`

#### 7. 正交性损失 — novelty 提升
- **文件**: `cra/models/cra_policy.py`, `cra/algo/ppo.py`
- `L_orth = Σ_{i<j} cos_sim(z_i, z_j)²` 鼓励各 stage latent 解耦
- `PPOConfig` 新增 `orthogonality_coef: float = 0.01`
- 仅在 `num_stages >= 2` 且 `has_history` 时激活

#### 8. Interaction Head — 解决 additive 假设
- **文件**: `cra/models/cra_policy.py`
- `InteractionHead` 小型 MLP: `ψ(z_cat) -> δa`，捕捉跨轴非线性效应
- 近零初始化 (`init_scale=0.001`) 确保平滑过渡
- `stage_idx >= 1` 时自动创建
- `action = base + residual + interaction_head(z_cat)`

### Phase 2: Baseline & 实验基础设施

#### 9. Curriculum DR Baseline（B6 + C3 修复，核心 baseline）
- **新文件**: `configs/method/curriculum_dr.yaml`
- **修改**: `cra/models/baselines.py` — 添加 `CurriculumDRPolicy`
- **修改**: `cra/trainer/baseline_trainer.py` — 添加 `_train_curriculum()` 方法
- 同 CRA 的 staged DR schedule，但单一网络全部参数不冻结
- 隔离 CRA 增益来源: 是 curriculum 还是 cascade architecture

#### 10. 参数量对齐 Baseline（A3 修复）
- **新文件**: `configs/method/full_dr_large.yaml`
- **新文件**: `configs/method/rma_large.yaml`
- `actor_hidden_dims: [1024, 512, 256]`，参数量对齐 CRA ~1.1M

#### 11. 环境测试脚本
- **新文件**: `scripts/test_env_standalone.py`
- 测试 DR 采样、观测构建、奖励计算、reset 逻辑、per-env gravity
- 支持 `--render` 、`--num-envs`、`--steps`、`--object-type` 参数

#### 12. Rendering 支持
- **修改**: `cra/envs/base_env.py` — 非 headless 模式自动创建 viewer
- `step()` 中自动渲染 + frame sync
- `close()` 中清理 viewer 资源
- `scripts/train.py` 添加 `--render` 参数覆盖 `--headless`

#### 13. 组合泛化评估脚本
- **新文件**: `scripts/eval_compositional.py`
- 4 种评估协议: 单轴 / 成对 / 全轴 / 外推 (1.5× range)
- 输出 JSON 结果文件

#### 14. 多种子基准脚本
- **新文件**: `scripts/benchmark.py`
- 支持 `--methods`、`--seeds`、`--dry-run`
- 生成 manifest.json + 顺序执行 + 结果汇总

#### 15. 轴排序消融脚本
- **新文件**: `scripts/ablation_ordering.py`
- 自动生成所有 3 轴排列的 config 文件
- 支持多种子 + dry-run

#### 16. WandB 集成
- **修改**: `cra/utils/logger.py`
- `Logger` 新增 `use_wandb`、`wandb_project`、`wandb_entity`、`wandb_config` 参数
- `log_metrics()` 同时写入 TensorBoard 和 WandB
- `close()` 自动调用 `wandb.finish()`

#### 17. 非对称 Actor-Critic（Privileged Critic）
- **修改**: `cra/models/base_policy.py`
- `ActorCritic` 新增 `privileged_dim` 参数
- Critic 输入: `[obs, privileged_info]`（训练时可传入真实 DR 参数）
- Actor 仅使用 `obs`
- `forward()`、`get_action()`、`evaluate_actions()` 均支持可选 `privileged` 参数

#### 18. eval.py / train.py 更新
- 支持 `curriculum_dr` 方法
- 修复 legacy `gravity_dir`/`gravity_mag` → 统一 `gravity`
- 正确构造 curriculum DR trainer 配置

### Phase 3: 深度反思与关键修复（Critical Reflection & Fixes）

> **本节基于完整代码审查后的第二轮反思，重点关注可能导致实验失败、论文被拒的致命/高危问题。**

#### 19. [致命] 四元数约定不匹配修复
- **文件**: `cra/envs/rewards.py`
- **问题**: `rewards.py` 所有四元数函数使用 `(w,x,y,z)` 约定，但 IsaacGym `root_state` tensor 使用 `(x,y,z,w)` 约定。`shadow_hand_rotation.py:310` 直接取 `root_state[:, 3:7]` 传给 reward 函数 → **旋转奖励完全错误**
- **修复**: 在 `_compute_observations()` 和 `_compute_rewards()` 中对 `object_quat` 执行 `xyzw_to_wxyz` 转换；`target_quat` 已使用 `quat_from_euler_xyz()`（IsaacGym 自带函数，返回 `xyzw`），也需转换
- 添加 `xyzw_to_wxyz()` 和 `wxyz_to_xyzw()` 辅助函数到 `rewards.py`

#### 20. [致命] Large Baseline 配置接线修复
- **文件**: `scripts/train.py`, `cra/trainer/baseline_trainer.py`
- **问题**: `full_dr_large.yaml` 和 `rma_large.yaml` 定义的 `actor_hidden_dims` 从未被 `train.py` 或 `BaselineTrainerConfig` 读取 → "参数量对齐"实验实际使用默认架构，与标准 baseline 完全相同
- **修复**: `BaselineTrainerConfig` 新增 `actor_hidden_dims` 和 `critic_hidden_dims` 字段；`BaselineTrainer.__init__` 构造 policy 时传入；`train.py` 从 config 读取并传入

#### 21. [高危] 移除 InteractionHead
- **文件**: `cra/models/cra_policy.py`, `cra/trainer/cra_trainer.py`
- **问题**: InteractionHead 接受所有 latents 的 concatenation → 直接否定了 additive decomposition 的核心 story。Reviewer 会指出这是自我矛盾：你声称 additive structure 是优势，又加了一个 joint MLP 来弥补
- **决策**: 完全移除 InteractionHead。如果实验显示 additive 不够，应该诚实讨论为 limitation，而非用一个 hack 来修补。纯 additive 结构是更干净的 contribution
- 保留 `use_interaction_head` 配置项设为 `False`，代码中移除 InteractionHead 类和相关逻辑

#### 22. [高危] 修复 CRA latent 冗余计算
- **文件**: `cra/models/cra_policy.py`
- **问题**: `get_action()` 中 `forward_action_mean()` 计算一次 latents，然后第 335 行又计算一次用于 critic。GRU forward pass 是计算瓶颈，这浪费了 ~50% 的推理时间
- **修复**: 重构 `forward_action_mean()` 返回 `(action_mean, latents)`；`get_action()` 和 `evaluate_actions()` 复用 latents

#### 23. [高危] 修复 eval 环境 DR 配置
- **文件**: `cra/trainer/cra_trainer.py`
- **问题**: `eval_env` 从不 enable DR axes → stage 训练期间的 evaluation 指标无意义（在无 DR 环境上评估适应能力）
- **修复**: 在每个 stage 开始时，对 `eval_env.dr_manager` 也 enable 相应的 DR axis

#### 24. [高危] 修复 DOF 索引计算
- **文件**: `cra/envs/shadow_hand_rotation.py`
- **问题**: `hand_dof_indices = hand_indices * num_hand_dofs` 假设每个 actor 恰好有 `num_hand_dofs` 个 DOF，但 object actor 也有 DOF（free body = 6 DOF）。正确计算应用 `gym.get_actor_dof_index()` 或考虑 object DOF
- **修复**: 使用 `gym.find_actor_dof_handle()` 正确获取 DOF 起始索引

#### 25. [中危] 修复正交性损失投影
- **文件**: `cra/models/cra_policy.py`
- **问题**: 当 latent dims 不同时，截取前 `min_d` 维做 cosine similarity 是不正确的——忽略了剩余维度的信息
- **修复**: 使用可学习的线性投影到公共维度，或统一所有 stage 使用相同 latent dim（推荐后者，更简洁）

#### 26. [中危] 修复 KL early stopping 逻辑
- **文件**: `cra/algo/ppo.py`
- **问题**: `avg_kl = total_approx_kl / max(num_updates, 1)` 是所有 epoch+batch 的累积平均 → 即使当前 epoch KL 很高，被之前低 KL 稀释后不会触发 early stop
- **修复**: 每个 epoch 结束时检查该 epoch 的平均 KL

#### 27. [中危] 接线 reach_goal_bonus 配置
- **文件**: `cra/envs/rewards.py`, `cra/envs/shadow_hand_rotation.py`
- **问题**: `RotationEnvConfig.reach_goal_bonus = 250.0` 从未使用。`rotation_reward()` 中 success bonus 硬编码为 2.0
- **修复**: `rotation_reward()` 接受 `success_bonus` 参数；`_compute_rewards()` 传入 `self.task_cfg.reach_goal_bonus`

#### 28. [中危] RMA config 使用 legacy gravity 轴
- **文件**: `configs/method/rma.yaml`
- **问题**: 使用 `gravity_dir` + `gravity_mag` 而非统一的 `gravity` → 与 CRA/FullDR 使用不同的 gravity sampling 行为，造成不公平比较
- **修复**: 改为统一的 `gravity` 轴

### Idea 层面的关键改进

#### I1. 组合泛化声明的修正
- **原声明**: "每个 adapter 独立响应 → 天然支持组合外推"
- **问题**: 信息层面不独立（obs history 包含所有轴的综合效应）
- **修正后声明**: "CRA 的组合泛化来源于**条件残差补偿结构**：每个 stage 被训练为在前级冻结的条件下补偿**一个新轴**的影响。由于冻结使得每个 stage 的训练是一个稳态 MDP，每个 adapter 学到的是**条件增量函数**而非联合辨识。在弱耦合假设下，这种 additive conditional residual 结构对 unseen 组合的泛化误差低于联合训练的 bound。"

#### I2. 移除 InteractionHead，保持方法纯净性
- 纯 additive 结构是更强的 contribution（简洁、可解释、可并行）
- 如果 additive 不够用，这是一个**诚实的 limitation**，不应该用 hack 掩盖

#### I3. 关于 novelty 的强化
- 核心 novelty 是三个设计的**结合**：(1) 参数轴分解 (2) 冻结级联 (3) additive 残差
- 需要消融实验证明每个组件的独立贡献：CRA-NoFreeze（无冻结）、CRA-SingleAxis（单一大 adapter）、CurriculumDR（同 schedule 但单网络）

### 待完成项

- [ ] Grasp Transition 环境 (`shadow_hand_grasp_transition.py`)
- [ ] 跨任务迁移实验脚本 (`scripts/eval_cross_task.py`)
- [ ] Object geometry axis (`object_scale`, YCB 多物体)
- [ ] Demo 采集脚本 (`scripts/collect_demos.py`)
- [ ] 结果可视化脚本 (`scripts/plot_results.py`)
- [ ] CRA-Small 配置（参数量对齐 FullDR ~0.5M）
- [ ] CRA-NoFreeze 消融配置
- [ ] End-to-end 训练烟雾测试 + 形状验证
