# Geodesic Loss 训练诊断实验

## 实验目的
排查 `geodesic` Grassmann loss 为何在训练中不收敛：weight=0.05 时 geodesic_loss 卡在 ~24.5,
weight=1.0 时发到 ~31.8, 而 control (w=0.0) 组 hami 训练收敛正常。

## 实验流水线

### 1. StableEigh backward 符号验证
- 文件: `verify_losses/verify_stableeigh_sign.py`
- 方法: 对比 `StableEigh` 和 `torch.linalg.eigh` 的梯度（特征值/特征向量/联合测试）
- 结论: ✅ 符号正确，简并情况下 StableEigh 给出有限梯度（~1e-16），原生 eigh 给 0

### 2. SVD backward NaN 排查
- 文件: `verify_losses/verify_svd_backward.py`
- 方法: 模拟 bad prediction（所有奇异值≈0）时 geodesic loss 的 SVD backward 是否 NaN
- 结论: ✅ 100 组测试全部有限梯度，无 NaN/Inf

### 3. 全链梯度流通性
- 文件: `verify_losses/diagnose_geodesic_loss.py`
- 方法: 模拟完整链 H_pred → eigh → occ → SVD → geodesic, 检查梯度是否流到 H_pred
- 结论: ✅ 梯度正常流通，grad_H norm=0.41, 有限，VV truncation 无影响

### 4. 离线加载 ckpt 计算 loss
- 文件: `verify_losses/diagnose_ckpt_losses.py`
- 方法: 加载三组的 epoch-00 checkpoint, 在 10 个乙醇分子上计算 hami_mae, geodesic, projection, densityS loss
- 结论: ✅ 本地值与训练日志完美吻合

### 5. MINAO 基线 geodesic
- 文件: `verify_losses/diagnose_ckpt_losses.py` 中的额外分析
- 方法: 计算 H_init (MINAO Fock 矩阵) 的 occupied subspace 与真实 subspace 的 geodesic 距离
- 结论: **MINAO 基线 geodesic ≈ 0.03（子空间几乎完美对齐）**

## 核心发现

| 状态 | hami_mae | geodesic_loss | projection | 说明 |
|------|----------|---------------|------------|------|
| **H_init (MINAO)** | N/A | **0.03** | 0.03 | 初始 Fock 矩阵的子空间已极准 |
| w=0.0 epoch 0 | 8.46e-04 | **0.96** | 0.47 | 纯 hami 训练 → 子空间轻微退化 |
| w=0.05 epoch 0 | 6.04e-03 | **24.50** | 10.0 | geodesic 梯度推歪子空间 |
| w=1.0 epoch 0 | 5.07e-02 | **31.75** | 13.0 | 几乎完全随机 |
| max possible | — | **32.07** | 13.0 | nocc=13, (π/2)²×13 |

## 根本原因（优化动力学悖论）

1. 模型架构 `remove_init=true` → 预测输出 `f(r)` 是 H_init 的修正: `H_pred = H_init + f(r)`
2. 随机初始化时 `f(r) ≈ 0` → `H_pred ≈ H_init` → 子空间极准 (geodesic≈0.03)
3. hami_loss 梯度方向与子空间保持方向不一致 — **改善矩阵元 → 破坏特征向量对齐**
4. 即使少量更新 (w=0.0), 子空间从 0.03 退化到 0.96 (30× worse)
5. geodesic 梯度 (SVD backward 中 `1/(σ_i²-σ_j²)` 大噪声) 进一步放大这种退化
6. w=0.05 时 geodesic 梯度占比小但已有害, w=1.0 时完全主导 → 发散

## 可能修复方案

### A. Grassmann weight warmup

**思路**: 训练开始时 grassmann_weight=0（纯 hami 训练），在前 warmup_steps 步中线性增长到目标值。

**原理**: 在第 0 步时 H_pred ≈ H_init, 子空间极准 (geodesic≈0.03)。随着 hami 梯度更新，子空间可能轻微退化，但因为 geodesic weight 极小 (≈0)，这种退化不会被放大。当 weight 逐渐增大时，子空间已经稳定在 hami 收敛后的状态（此时 σ 值分布合理，SVD backward 稳定），geodesic 梯度只会微调子空间方向。

**实现方式**:

修改 `GrassmannError` 类，添加 warmup 逻辑：

```python
class GrassmannError(_OrbitalEnergyErrorBase):
    def __init__(self, ..., grassmann_warmup_steps=0):
        ...
        self.warmup_steps = grassmann_warmup_steps
        self._call_count = 0          # 计数每次 cal_loss 调用

    def cal_loss(self, batch_data, error_dict={}, metric=None):
        ...
        if compute_grass and grass_losses:
            lg = torch.stack(grass_losses).mean()
            error_dict['grassmann_loss'] = lg.detach()

            # 线性 warmup: weight 从 0 ramp 到 target
            weight = self.grassmann_weight
            if self.warmup_steps > 0:
                ratio = min(1.0, self._call_count / self.warmup_steps)
                weight = self.grassmann_weight * ratio
            self._call_count += 1

            error_dict['loss'] += weight * lg
```

**实际实现**: 使用外挂包装类 `GrassmannWarmupWrapper`，不修改 `GrassmannError` 本身（文件: `src/training/losses.py:972-1041`）：

```python
class GrassmannWarmupWrapper:
    def __init__(self, grassmann_error, warmup_steps=5000):
        self._inner = grassmann_error
        self.warmup_steps = warmup_steps
        self._call_count = 0

    def cal_loss(self, batch_data, error_dict=None, metric=None):
        self._call_count += 1
        ratio = min(1.0, float(self._call_count) / self.warmup_steps)

        orig_gw = self._inner.grassmann_weight
        self._inner.grassmann_weight = 0.0               # 内部不累计
        self._inner.cal_loss(batch_data, error_dict, metric)
        self._inner.grassmann_weight = orig_gw           # 恢复

        lg = error_dict.get('grassmann_loss')
        if lg is not None and orig_gw > 0:
            error_dict['loss'] = error_dict['loss'] + (orig_gw * ratio) * lg
        return error_dict
```

**接线方式** (`src/training/module.py:192-203`):
```python
grass_err = GrassmannError(...)
warmup_steps = self.hparams.get("grassmann_warmup_steps", 0)
if warmup_steps > 0:
    grass_err = GrassmannWarmupWrapper(grass_err, warmup_steps=warmup_steps)
self.loss_func_list_train.append(grass_err)
```

使用时在 Hydra config 中增加: `+grassmann_warmup_steps=5000`，验证侧不启用 warmup（始终报告全量 loss）。

---

### B. 换 projection metric

**思路**: 用 `||P_pred - P_gt||_F² / nocc` 替代 `sum(arccos²(σ))`。

**对比**:
| 属性 | geodesic | projection |
|------|----------|------------|
| 公式 | `sum(arccos²(σ))` | `\|P_pred - P_gt\|² / nocc` |
| 梯度路径 | SVD + acos | 直接矩阵运算 |
| 最大梯度噪声源 | `1/(σ_i²-σ_j²)` | 无 |
| 数值范围 | 0 ~ 32 (nocc=13) | 0 ~ 2 |
| 与 hami loss 尺度匹配 | ❌ 差 100-1000× | ✅ 相近 |

**实现方式**: 修改 shell 脚本中的 `+grassmann_metric=projection` 即可（无需改代码）。

**优点**: 无 SVD，梯度稳定，天然与 hami loss 尺度匹配（旧代码已验证 `w=0.05 projection` 能正常收敛）。
**缺点**: 不等价于 Riemannian geodesic distance，物理含义不如 geodesic 精确。

---

### C. 实现 StableSVD

**思路**: 仿照 `StableEigh`，给 `torch.linalg.svd` 写 custom autograd Function，在 backward 中 clamp 住 `1/(σ_i² - σ_j²)` 项。

**SVD backward 通用公式**:
```
dL/dM = U [ (F ⊙ (U^T dU)) + diag(dL/dΣ) ] V^T
```
其中 `F_ij = 1/(σ_j² - σ_i²)`，当 σ 集群退化时 `F → ∞`。

**StableSVD backward**:
```python
diff_sq = sigma.unsqueeze(-1)**2 - sigma.unsqueeze(-2)**2  # σ_j² - σ_i²
mask = (diff_sq.abs() > eps)
safe_diff_sq = torch.where(mask, diff_sq, torch.ones_like(diff_sq))
F = torch.where(mask, 1.0 / safe_diff_sq, torch.zeros_like(diff_sq))
```

整个 backward 需要额外处理 U 和 V 的梯度（非对称矩阵），比 eigh 的 backward 复杂。但与 StableEigh 概念一致。

**优点**: 从根因修复 geodesic loss 的梯度不稳定性，可保留 geodesic metric。
**缺点**: 实现复杂（SVD backward 涉及 U、Σ、V 三个部分的链式法则），需要仔细数值验证。

---

### D. 两阶段训练

**思路**: 将训练拆为两个独立的阶段:
- **Phase 1**: hami-only (w=0), 训练到 hami_mae 收敛（~60000 steps）
- **Phase 2**: load Phase 1 的最优 checkpoint，添加 geodesic loss (w>0)，fine-tune ~10000 steps

**原因**: 在 Phase 2 开始时，H_pred 已经很接近 H_gt（hami_mae≈1e-5），子空间几乎完美（geodesic≈0.001，σ≈1 且区分度好）。此时 SVD backward 中的 `1/(σ_i²-σ_j²)` 各项都是稳定的有限值，geodesic 梯度是有意义的方向。

**实现方式**: 不需要改代码，只需:
1. 先跑 hami-only (w=0) 训练
2. 拿到最优 ckpt 后，用 `trainer.fit(ckpt_path=...)` resume 训练，并覆盖 `grassmann_weight` 为非零值

**优点**: 绝对安全 — Phase 1 已经收敛到局部最优，Phase 2 只做微调。H_init 子空间退化的风险被限制在最小范围。
**缺点**: 需要跑两次训练（总计算量翻倍），两个阶段之间需要手动串流程。

---

## 推荐优先级

| 优先级 | 方案 | 理由 |
|--------|------|------|
| 1 | **B. projection** + **D. 两阶段** | 已提交 projection 对照任务在跑，等结果出来后直接比。如果 projection 收敛好，直接用；如果 geodesic 想保留，用两阶段 fine-tune。 |
| 2 | **A. warmup** | 实现代价最小，可直接验证 gradient scale 假说 |
| 3 | **C. StableSVD** | 根因修复但实现复杂度高

## 训练任务 & Checkpoint 对照

| 组别 | weight | TaskName | task_id | epoch-00 ckpt |
|------|--------|----------|---------|---------------|
| geodesic w=1.0 | 1.0 | v2_geodesic_w1_0_20260715 | z77rl | first10-epoch00-31.834799.ckpt |
| geodesic w=0.05 | 0.05 | v2_geodesic_w0_05_20260715 | t4nlr | first10-epoch00-1.232646.ckpt |
| geodesic w=0.0 (control) | 0.0 | v2_geodesic_w0_0_20260715 | tk2kt | first10-epoch00-0.000858.ckpt |
| projection w=1.0 | 1.0 | v2_projection_w1_0_20260715 | lf6zh | (pending) |
| projection w=0.05 | 0.05 | v2_projection_w0_05_20260715 | lt2s2 | (pending) |
| projection w=0.0 | 0.0 | v2_projection_w0_0_20260715 | h7kng | (pending) |

输出根目录: /home/pepe/workbench/grassd_outputs/

## 使用说明

```bash
# 验证 StableEigh 符号
python verify_losses/verify_stableeigh_sign.py

# 验证 SVD backward 稳定性
python verify_losses/verify_svd_backward.py

# 诊断全链梯度
python verify_losses/diagnose_geodesic_loss.py

# 离线 ckpt loss 计算
python verify_losses/diagnose_ckpt_losses.py
```
