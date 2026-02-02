# MP-SHM-X2: Multi-Physics Stochastic Hybrid Model

MP-SHM-X2 (Multi-Physics Stochastic Hybrid Model - X2) 是一个用于智能手机电池寿命预测的**随机混合系统 (Stochastic Hybrid System, SHS)** 实现。该项目旨在通过将离散的用户行为随机过程与连续的多物理场动力学方程相结合，精确模拟电池在真实使用场景下的消耗与老化。

## 0. 快速使用指南

### 如何运行仿真
直接运行详细仿真脚本，生成 SOC 曲线和状态分布图：
```bash
uv run run_detailed_simulation.py
```
运行结束后，会在根目录生成 `detailed_simulation_report.png`。

### 如何修改参数
您可以通过修改以下文件来调整仿真行为：

1.  **修改物理参数与用户配置 (首选)**: `mp_shm_x2/config.py`
    *   **物理参数**: 修改 `SimulationParams` 类中的字段（如电池容量 `Q_design`, 环境温度 `T_amb`, 截止电压 `V_cut`）。
    *   **用户状态参数**: 修改 `user_params` 字典。
        *   `mu_dwell`: 状态持续时间参数（对于 Idle 状态，这是夜间最长待机时间的对数值）。
        *   `f_base`, `L_base`: 该状态下的 CPU 频率和屏幕亮度基准值。
        *   `lambda_net`: 该状态下的网络数据包到达率。

2.  **修改状态转移逻辑**: `mp_shm_x2/user_config.py`
    *   **状态列表**: 修改 `STATES` 列表添加新状态。
    *   **转移概率**: 修改 `TRANSITIONS` 字典调整状态切换概率。
    *   **持续时间策略**: 修改 `DwellingTimePolicy` 类以自定义特殊状态的持续时间逻辑。

---

## 1. 架构概览

本项目采用**离散时间步进 (Discrete Time-Stepping)** 方法进行数值模拟，核心架构遵循**分层解耦**设计：

1.  **随机激励层 (Stochastic Layer)**: 模拟用户行为切换（半马尔可夫链）及系统负载的随机波动（OU 过程、泊松过程）。
2.  **功率计算层 (Power Layer)**: 将随机激励转换为具体的功率需求 $P_{\text{sys}}$，包含 CPU、屏幕和网络模块。
3.  **电化学层 (Electrochemical Layer)**: 核心求解器。基于功率需求解代数方程得到电流 $I$，并积分更新 SOC 和极化电压。
4.  **热力学层 (Thermal Layer)**: 双节点热模型，计算核心温度 $T_c$ 和表面温度 $T_s$，并反馈影响电化学参数。

---

## 2. 核心方程与实现

所有连续状态的更新均采用 **显式欧拉法 (Explicit Euler Method)**，时间步长默认为 $\Delta t = 1s$。

### 2.1 随机激励模块 (`mp_shm_x2.stochastic`, `mp_shm_x2.power`, `mp_shm_x2.user_config`)

*   **用户行为 (Advanced Semi-Markov Model)**:
    *   **状态空间**: $S \in \{\text{Idle, Video, Game, Call, Camera}\}$，支持扩展。
    *   **状态转移**: 使用可配置的转移矩阵 $P(S_{next} | S_{curr})$ 驱动状态切换。注意：由于驻留时间显式建模，转移矩阵中通常移除自转移（Self-transition）。
    *   **驻留时间 (Dwelling Time)**:
        *   **策略模式 (Strategy Pattern)**: 核心逻辑封装在 `user_config.py` 的 `DwellingTimePolicy` 类中，不同状态采用不同采样策略。
        *   **多峰分布**: Idle 状态采用混合分布，区分“短时查看”（Short Idle）和“长时待机”（Long Idle）。
        *   **历史依赖 (History Dependency)**: 驻留时间分布参数依赖于前序状态（如 $S_{prev} = \text{Camera} \implies$ Idle 倾向于短时）。
        *   **昼夜节律 (Circadian Rhythm)**: 引入时钟变量 $T_{\text{clock}}$，分布参数（如 `base_median`, `sigma`）随时间呈周期性变化（模拟日间活跃/夜间休眠）。
        *   **配置优先**: 支持通过 `SimulationParams.user_params` 注入参数（如 `mu_dwell`），策略类会自动处理参数的物理含义（如将 Idle 的 `mu_dwell` 解释为夜间最大时长的对数值）。
*   **负载噪声**: 使用 Ornstein-Uhlenbeck (OU) 过程模拟 CPU 频率和屏幕亮度的波动。
    *   **方程**: $d\xi = -\frac{1}{\tau}\xi dt + \sigma dW_t$
    *   **实现**: `OUProcess` 类使用 Euler-Maruyama 方法更新：
        `x_new = x + (-1/tau * x * dt) + sigma * sqrt(dt) * normal(0,1)`
*   **网络尾耗**: 模拟无线电的高能耗“尾巴”状态。
    *   **方程**: $\frac{dx_{\text{net}}}{dt} = -\frac{1}{\tau_{\text{tail}}} x_{\text{net}} + \text{impulses}$
    *   **实现**: `NetworkPower.step()` 中，无数据包时指数衰减 `x *= exp(-dt/tau)`，有数据包时重置为 1.0。

### 2.2 电化学内核模块 (`mp_shm_x2.battery`)

这是模型的核心，包含一个代数环（Algebraic Loop）。

*   **电流求解 (代数约束)**:
    *   **方程**: $R_0 I^2 - (U_{\text{ocv}} - V_p)I + P_{\text{sys}} = 0$
    *   **实现**: `BatteryModel.solve_current()` 解此二次方程。
    *   **失效判定**: 若判别式 $\Delta < 0$，抛出 `PowerCollapseError`（功率崩溃），代表负载超过电池物理极限。
*   **SOC 演化**:
    *   **方程**: $\frac{dz}{dt} = -\frac{I}{3600 \cdot Q_{\text{design}}}$
    *   **实现**: `get_soc_derivative()` 计算导数，主循环中 `z += dz * dt`。
*   **极化电压**:
    *   **方程**: $\frac{dV_p}{dt} = -\frac{V_p}{\tau_p} + \frac{I}{C_p}$
    *   **实现**: `get_vp_derivative()`。

### 2.3 热力学模块 (`mp_shm_x2.thermal`)

采用双节点（核心 Core + 表面 Surface）热阻容网络。

*   **核心温度 $T_c$**:
    *   **方程**: $C_c \frac{dT_c}{dt} = I^2 R_0 + I T_c \frac{\partial U}{\partial T} - \frac{T_c - T_s}{R_{\text{th,in}}}$
    *   **实现**: `get_core_temp_derivative()`。包含焦耳热、可逆熵热（Peltier效应）和内部传导。
*   **表面温度 $T_s$**:
    *   **方程**: $C_s \frac{dT_s}{dt} = \frac{T_c - T_s}{R_{\text{th,in}}} - h_{\text{eff}}A(T_s - T_{\text{amb}}) + \eta P_{\text{sys}}$
    *   **实现**: `get_surface_temp_derivative()`。包含内部传导、对流散热和系统热耦合。

## 3. 参数配置 (`mp_shm_x2.config`)

项目支持高度可配置的参数系统，符合 `Parameters.md` 规范。
*   **[C] 常量参数**: 如 $Q_{\text{design}}$, $E_a$。
*   **[F] 函数参数**: 如 $R_0(z, T_c)$, $U_{\text{ocv}}(z, T_c)$，支持动态计算。
*   **用户状态参数**: 在 `SimulationParams.user_params` 中定义不同模式（Game, Video等）下的基准频率、亮度及驻留时间分布参数。

## 4. 快速开始

### 运行仿真
执行 `run_detailed_simulation.py` 即可运行一个标准的 "Idle/Video" 混合场景仿真，并生成 SOC 曲线图。

```bash
uv run run_detailed_simulation.py
```

### 运行测试
项目包含完整的单元测试，覆盖各个物理子模块。

## 5. 高级特性实现说明

### 5.1 半马尔可夫链增强 (Semi-Markov Enhancements)

为满足复杂场景模拟需求，我们在 `UserBehaviorModel` 中实现了以下机制：

1.  **灵活的状态转移**:
    使用字典结构 `self.transitions` 存储稀疏转移矩阵，允许动态修改状态空间和转移概率。
    ```python
    self.transitions = {
        "Idle": {"Video": 0.3, "Game": 0.2, ...},
        ...
    }
    ```

2.  **条件驻留时间生成**:
    `_sample_dwelling_time(state)` 方法集成了多重逻辑：
    *   **历史回溯**: 检查 `self.history`。例如，若上一个状态是 `Camera`，则强制进入短时 Idle 模式（模拟拍完照后锁屏）。
    *   **混合分布 (Mixture Model)**: 对于 Idle 状态，根据概率（受时间影响）选择“短模式”（Mean~60s）或“长模式”（Mean~1h）。
    *   **时变参数**: 引入 `_get_circadian_factor()` 计算昼夜因子（0.0~1.0），动态调整分布的 $\mu$ 和 $\sigma$。例如，夜间 Video 的观看时长期望值更长。

3.  **时钟同步**:
    引入 `self.t_clock` 变量，在 `step()` 中与仿真时间同步推进（`dt`），并按 24小时（86400s）循环，为昼夜节律提供时间基准。

---
