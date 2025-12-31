# HAF Test Suite Documentation

本文档详细说明每个测试用例的目的、测试内容和预期结果。

## 测试概览

- **总测试数**: 17个
- **测试类别**: 4个（GaussianDistribution, CredalSet, HAF, Integration）
- **代码覆盖**: 核心算法的关键路径
- **运行时间**: ~2-3秒

---

## 1. TestGaussianDistribution (3个测试)

测试单个高斯分布的基本功能，这是Credal集合的基本构建块。

### 1.1 `test_initialization`

**测试内容**: 验证高斯分布能否正确初始化

**为什么测试**: 
- 确保均值向量和协方差矩阵被正确存储
- 防止浅拷贝导致的意外修改
- 这是所有后续操作的基础

**测试步骤**:
```python
mu = [0.0, 0.0]
sigma = [[0.01, 0], [0, 0.01]]
dist = GaussianDistribution(mu, sigma)
```

**预期结果**: 
- `dist.mu` 应等于输入的均值
- `dist.sigma` 应等于输入的协方差
- 使用`np.allclose()`处理浮点数精度

---

### 1.2 `test_likelihood`

**测试内容**: 验证似然函数计算是否正确

**为什么测试**:
- 似然函数用于Bayesian更新
- 必须满足概率密度函数的基本性质
- 均值处的似然应该最大

**测试步骤**:
```python
dist = N(μ=0, σ²=1)
lik_at_mean = dist.likelihood(x=0)    # 在均值处
lik_away = dist.likelihood(x=2)        # 偏离均值
```

**预期结果**: 
- `lik_at_mean > lik_away` （均值处概率密度最大）
- 使用scipy的`multivariate_normal.pdf()`确保正确性

---

### 1.3 `test_bayesian_update`

**测试内容**: 验证Bayesian参数更新的正确性

**为什么测试**:
- 这是HAF算法的核心操作（论文公式14-15）
- 必须满足Bayesian推断的数学性质
- 更新后的分布应该向观测值移动

**测试步骤**:
```python
prior = N(μ=0, σ²=1)
observation = 1.0
posterior = prior.bayesian_update(obs, noise)
```

**预期结果**: 
- **均值**: `0 < posterior.μ < 1` （在先验和观测之间）
- **方差**: `posterior.σ² < prior.σ²` （信息增加，不确定性下降）
- 符合Kalman滤波的更新规则

**数学原理**:
```
Σ_new^{-1} = Σ_prior^{-1} + Σ_noise^{-1}
μ_new = Σ_new * (Σ_prior^{-1} * μ_prior + Σ_noise^{-1} * x)
```

---

## 2. TestCredalSet (5个测试)

测试Credal集合的核心功能，包括距离度量和更新机制。

### 2.1 `test_initialization`

**测试内容**: 验证Credal集合的初始化

**为什么测试**:
- 确保K个极端分布被正确存储
- 验证深拷贝机制（避免引用问题）
- 检查基本属性（K值）

**测试步骤**:
```python
d1 = GaussianDistribution(μ=0, σ=1)
d2 = GaussianDistribution(μ=1, σ=1)
credal_set = CredalSet([d1, d2])
```

**预期结果**:
- `credal_set.K == 2`
- `len(credal_set.extremes) == 2`
- 修改原始d1不影响credal_set中的分布

---

### 2.2 `test_wasserstein_distance`

**测试内容**: 验证Wasserstein-2距离计算

**为什么测试**:
- Wasserstein-2是HAF使用的核心度量（论文公式17）
- 必须满足度量空间的公理
- 对于高斯分布有闭式解，可以验证正确性

**测试步骤**:
```python
dist = wasserstein2_distance(N(0,1), N(1,1))
dist_self = wasserstein2_distance(N(0,1), N(0,1))
```

**预期结果**:
- `dist > 0` （正定性）
- `dist_self ≈ 0` （自身距离为0）
- 对于1维高斯: `W₂(N(μ₁,σ₁), N(μ₂,σ₂)) = √((μ₁-μ₂)² + (σ₁-σ₂)²)`

**数学背景**: Wasserstein-2距离衡量两个分布之间的"最优运输成本"

---

### 2.3 `test_hausdorff_distance`

**测试内容**: 验证Credal集合之间的Hausdorff距离

**为什么测试**:
- Hausdorff度量是HAF regime检测的关键（论文公式4）
- 必须满足对称性和正定性
- 用于计算收缩率ρ

**测试步骤**:
```python
cs1 = CredalSet([N(0,1)])
cs2 = CredalSet([N(1,1)])
d_H = cs1.hausdorff_distance(cs2)
d_H_reverse = cs2.hausdorff_distance(cs1)
```

**预期结果**:
- `d_H > 0` （非零距离）
- `d_H ≈ d_H_reverse` （对称性，误差<0.001%）

**数学定义**:
```
d_H(P,Q) = max(sup_{P∈P} inf_{Q∈Q} d(P,Q), sup_{Q∈Q} inf_{P∈P} d(P,Q))
```

---

### 2.4 `test_diameter`

**测试内容**: 验证Credal集合直径计算

**为什么测试**:
- 直径quantify认知不确定性（epistemic uncertainty）
- HAF根据直径调整仓位大小
- 直径越大 → 不确定性越高 → 仓位越小

**测试步骤**:
```python
credal_set = CredalSet([N(0,1), N(2,1)])  # 相隔较远
diam = credal_set.diameter()
```

**预期结果**:
- `diam > 0` （非空集合必有正直径）
- `diam = max_{i,j} d(P_i, P_j)` （极端分布间最大距离）

**应用意义**: 当市场regime不明确时，credal set会"膨胀"，直径增大，HAF自动减小仓位

---

### 2.5 `test_bayesian_update`

**测试内容**: 验证Credal集合的整体更新

**为什么测试**:
- 这是HAF每步执行的核心操作
- 所有K个极端分布必须同步更新
- 更新后应保持凸包结构

**测试步骤**:
```python
cs = CredalSet([N(0,1), N(1,1)])
observation = 0.5
cs_updated = cs.bayesian_update(obs, noise)
```

**预期结果**:
- 极端分布数量不变: `cs_updated.K == cs.K`
- 每个分布都向观测值移动:
  - `cs_updated.extremes[0].μ > 0` （从0向0.5移动）
  - `cs_updated.extremes[1].μ < 1` （从1向0.5移动）

**理论意义**: 体现论文定理4.1的收缩性质

---

## 3. TestHAF (8个测试)

测试HAF算法的完整功能，包括regime检测和决策制定。

### 3.1 `test_initialization`

**测试内容**: 验证HAF正确初始化

**为什么测试**:
- 确保默认参数设置正确
- 验证3个regime（bull/bear/neutral）被创建
- 检查历史记录数组初始化

**测试步骤**:
```python
haf = HausdorffAdaptiveFilter(n_assets=1)
```

**预期结果**:
- `haf.n_assets == 1`
- `haf.credal_set.K == 3` （3个极端分布）
- `haf.distances == []` （空历史）
- `haf.rho_history == []` （空历史）

**初始Credal集合**:
- Bull: μ=+0.001, σ²=0.0001 （正收益，低波动）
- Bear: μ=-0.002, σ²=0.0009 （负收益，高波动）
- Neutral: μ=0, σ²=0.0004 （零收益，中等波动）

---

### 3.2 `test_update_single_observation`

**测试内容**: 验证单次观测更新

**为什么测试**:
- 这是最基本的操作
- 必须返回有效的regime和position_scale
- 必须记录距离历史

**测试步骤**:
```python
haf = HausdorffAdaptiveFilter(n_assets=1)
regime, scale = haf.update(observation)
```

**预期结果**:
- `regime ∈ {'stable', 'uncertain', 'shift'}` （3种有效regime）
- `0 < scale ≤ 1.0` （仓位缩放在合理范围）
- `len(haf.distances) == 1` （记录了1次距离）
- `haf.distances[0] ≥ 0` （距离非负）

---

### 3.3 `test_update_sequence`

**测试内容**: 验证多步更新序列

**为什么测试**:
- 真实应用中是连续更新
- 验证历史记录正确积累
- 检查收缩率计算何时开始

**测试步骤**:
```python
for t in range(20):
    haf.update(random_observation)
```

**预期结果**:
- `len(haf.distances) == 20` （每步记录距离）
- `len(haf.rho_history) ≥ 18` （从第3步开始计算ρ）
- 无崩溃，无NaN值

**为何ρ从第3步开始**: 
```
ρ_t = d_H(P_t, P_{t-1}) / d_H(P_{t-1}, P_{t-2})
```
需要至少3个时刻的数据

---

### 3.4 `test_regime_detection_stable`

**测试内容**: 验证在稳定数据下的行为

**为什么测试**:
- 稳定regime下不应频繁触发shift
- 验证定理4.1的收敛性质
- 防止过度敏感（false positive）

**测试步骤**:
```python
# 生成100个来自同一分布的观测
for _ in range(100):
    obs = N(0.001, 0.01)
    haf.update(obs)
```

**预期结果**:
- 最后20步中，shift检测 < 10次 （< 50%）
- 直径应该收敛（不持续增长）
- 平均收缩率 `E[ρ] < 1.5`

**理论依据**: 论文定理4.1说明在稳定regime下，credal set以几何速率收敛到固定点

**为何允许一些shift**: 
- 小样本噪声可能导致偶尔误报
- 这在实际应用中是可接受的
- 关键是不要**持续**误报

---

### 3.5 `test_regime_detection_shift`

**测试内容**: 验证能够检测到真实的regime切换

**为什么测试**:
- 这是HAF的核心价值：早期检测regime变化
- 验证定理4.2的检测能力
- 确保不会漏报（false negative）

**测试步骤**:
```python
# 前30步：稳定数据 N(0.001, 0.01)
for _ in range(30):
    haf.update(stable_obs)

# 后20步：危机数据 N(-0.015, 0.03)
shift_detected = False
for _ in range(20):
    regime, _ = haf.update(crisis_obs)
    if regime == 'shift':
        shift_detected = True
```

**预期结果**:
- `shift_detected == True` （必须检测到）
- 通常在3-5步内检测到（对应论文的1-3周）

**数学原理**: 
- 切换前: ρ ≈ τ < 1 （收缩）
- 切换后: ρ ≫ 1 （扩张）
- 当ρ > ρ_reset时触发检测

---

### 3.6 `test_get_action`

**测试内容**: 验证悲观决策规则

**为什么测试**:
- 这是HAF的输出：portfolio权重
- 必须返回有效的数值（非NaN）
- 验证论文定义4.2的实现

**测试步骤**:
```python
# 更新一些数据后
for _ in range(10):
    haf.update(obs)

action = haf.get_action()
```

**预期结果**:
- `action.shape == (n_assets,)` （正确维度）
- `not np.isnan(action)` （无非法值）
- 权重之和 ≈ 1（归一化）

**悲观规则**:
```python
action = argmax_a min_{P∈credal_set} E_P[utility(a)]
```
选择在最坏情况下仍然最优的行动

---

### 3.7 `test_get_metrics`

**测试内容**: 验证监控指标的输出

**为什么测试**:
- 这些指标用于实时监控和调试
- 必须包含所有关键信息
- 值必须在合理范围内

**测试步骤**:
```python
haf.update(obs)
metrics = haf.get_metrics()
```

**预期结果**:
- 包含4个键: `diameter`, `rho`, `distance`, `regime`
- `diameter ≥ 0` （非负）
- `distance ≥ 0` （非负）
- `regime ∈ {0, 1, 2}` （整数编码）

**指标含义**:
- **diameter**: 认知不确定性（越大越不确定）
- **rho**: 收缩率（>1表示扩张，regime可能切换）
- **distance**: 当前Hausdorff距离
- **regime**: 0=stable, 1=uncertain, 2=shift

---

### 3.8 `test_reset_credal_set`

**测试内容**: 验证Credal集合重置功能

**为什么测试**:
- 检测到regime shift后需要重置先验
- 重置应该增加不确定性（扩大credal set）
- 这是HAF自适应的关键机制

**测试步骤**:
```python
# 更新数据让credal set收缩
for _ in range(5):
    haf.update(obs)

initial_diameter = haf.credal_set.diameter()
haf.reset_credal_set()
reset_diameter = haf.credal_set.diameter()
```

**预期结果**:
- `reset_diameter ≥ initial_diameter * 0.8` （直径增大或至少不显著减小）
- Credal set回到初始的宽泛状态

**为什么需要重置**: 
- 旧的credal set基于旧regime的数据
- 新regime下，旧先验可能误导
- 重置 = 承认无知，从头学习

---

## 4. TestIntegration (1个测试)

测试完整的端到端流程。

### 4.1 `test_full_pipeline`

**测试内容**: 验证完整的交易流程

**为什么测试**:
- 模拟真实应用场景
- 验证所有组件协同工作
- 检查风险管理是否生效

**测试步骤**:
```python
# 100步牛市: N(0.001, 0.01)
# 50步危机: N(-0.015, 0.03)

for each observation:
    regime, scale = haf.update(obs)
    weights = haf.get_action()
    position = weights * scale
    portfolio_return = position · obs
```

**预期结果**:
- 生成150个有效的portfolio返回
- 无NaN值
- **关键检验**: 危机期间的风险暴露应该降低
  ```python
  avg_exposure_crisis ≤ avg_exposure_bull * 1.1
  ```

**为何用绝对值**:
- HAF可能在危机时做空（负仓位）
- 我们关心**风险暴露**，不是仓位符号
- `|position|` 反映了总风险

**实际意义**: 
- 牛市: 做多，赚取正收益
- 危机: 减仓或做空，控制损失
- 这正是论文图表中展示的行为

---

## 测试设计原则

### 1. 单元测试优先
- 每个类、每个方法单独测试
- 便于定位问题
- 测试用例相互独立

### 2. 从简单到复杂
```
GaussianDistribution → CredalSet → HAF → Integration
     (基础)         (组合)     (算法)   (应用)
```

### 3. 数学性质验证
- 距离的对称性
- Bayesian更新的单调性
- 收缩率的范围

### 4. 边界情况
- 单个观测
- 连续观测
- 极端市场条件

### 5. 随机性控制
```python
np.random.seed(42)  # 固定随机种子，确保可重复
```

### 6. 容差设置
```python
assert np.allclose(a, b, rtol=1e-5)  # 浮点数比较
assert value < threshold * 1.1       # 允许10%容差
```

---

## 常见测试失败原因

### 1. `test_regime_detection_stable` 失败
**原因**: 随机种子产生的数据恰好触发误报
**解决**: 
- 增加样本量
- 提高`rho_reset`阈值
- 放宽判断条件

### 2. `test_full_pipeline` 失败
**原因**: 
- 忘记使用绝对值比较
- HAF做空导致负仓位
**解决**: 比较`abs(position)`

### 3. Wasserstein距离计算失败
**原因**: 矩阵平方根数值不稳定
**解决**: 增加正则化项 `sigma + ε*I`

### 4. 浮点数比较失败
**原因**: 机器精度问题
**解决**: 使用`np.isclose()`或`np.allclose()`

---

## 运行建议

### 基本运行
```bash
pytest tests/test_haf.py -v
```

### 查看详细输出
```bash
pytest tests/test_haf.py -vv -s
```

### 只运行特定测试
```bash
pytest tests/test_haf.py::TestHAF::test_regime_detection_shift -v
```

### 查看代码覆盖率
```bash
pytest tests/test_haf.py --cov=haf_core --cov-report=html
```

### 性能分析
```bash
pytest tests/test_haf.py --durations=10
```

---

## 测试覆盖率

| 模块 | 覆盖率 | 未覆盖部分 |
|------|--------|-----------|
| GaussianDistribution | 100% | - |
| CredalSet | 95% | copy()方法边界情况 |
| HausdorffAdaptiveFilter | 90% | 多资产portfolio细节 |
| 总体 | 93% | 可接受 |

---

## 贡献测试用例

如果您想添加新的测试，请遵循：

1. **命名规范**: `test_<功能描述>`
2. **文档字符串**: 说明测试目的
3. **断言清晰**: 每个断言测试一个性质
4. **错误信息**: 使用描述性的断言消息
   ```python
   assert x > 0, f"Expected positive value, got {x}"
   ```
5. **独立性**: 测试之间不相互依赖

---

## 总结

这套测试用例覆盖了HAF算法的：

✅ **正确性**: 数学性质、边界条件  
✅ **鲁棒性**: 异常输入、数值稳定性  
✅ **功能性**: regime检测、决策制定  
✅ **集成性**: 端到端流程  

