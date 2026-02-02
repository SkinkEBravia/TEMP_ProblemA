# **MP-SHM-X2: 智能手机电池随机混合模型完整说明文档**

## **1. 模型概述**

MP-SHM-X2（Multi-Physics Stochastic Hybrid Model - X2）是一个用于智能手机电池寿命预测的**随机混合系统（Stochastic Hybrid System, SHS）**。模型核心思想是将**离散的用户行为**（跳跃过程）与**连续的多物理场动力学**（微分代数方程）进行数学耦合，从而捕捉电池消耗的"不可预测性"。

### **模型特性**
- **分层解耦架构**：分为激励层、电化学层、热力学层
- **随机-确定性混合**：半马尔可夫链 + OU过程 + 确定性DAE
- **多物理场耦合**：电化学-热力学双向耦合
- **可解性分析**：显式处理"功率崩溃"边界条件

---

## **2. 系统状态定义**

### **2.1 连续状态向量** $\mathbf{x}(t) \in \mathbb{R}^5$
$$
\mathbf{x}(t) = \begin{bmatrix}
z(t) & \text{SOC (State of Charge)} \\
V_p(t) & \text{极化电压 (Polarization)} \\
T_c(t) & \text{核心温度 (Core Temperature)} \\
T_s(t) & \text{表面温度 (Skin Temperature)} \\
x_{\text{net}}(t) & \text{网络尾耗状态 (Network Tail State)}
\end{bmatrix}
$$

### **2.2 离散随机状态**
- $S(t) \in \{\text{Idle, Video, Game, Call, ...}\}$：用户行为模式
- 由**半马尔可夫链**驱动，驻留时间服从对数正态分布

### **2.3 激励噪声状态**
- $\xi_f(t)$：CPU频率OU噪声过程
- $\xi_L(t)$：屏幕亮度OU噪声过程

---

## **3. 随机激励模块 (Stochastic Excitation)**

### **3.1 用户行为半马尔可夫链**
状态$S_k$的驻留时间：
$$
\tau_k \sim \text{LogNormal}(\mu[S_k], \sigma^2[S_k]) \quad \text{[TEXT描述]}
$$
- $\mu[S_k], \sigma[S_k]$：状态依赖的对数正态分布参数
- 物理意义：模拟用户行为的"惯性"，避免无记忆的指数分布

### **3.2 Ornstein-Uhlenbeck (OU) 噪声过程**
$$
d\xi_f = -\frac{1}{\tau_{\text{ou},f}} \xi_f dt + \sigma_{\text{ou},f} dW_f(t) \quad \text{[推导自TEXT描述]}
$$
$$
d\xi_L = -\frac{1}{\tau_{\text{ou},L}} \xi_L dt + \sigma_{\text{ou},L} dW_L(t)
$$

**离散形式（Euler-Maruyama）**：
$$
\xi_f(t+\Delta t) = \xi_f(t) + \frac{1}{\tau_{\text{ou},f}}(\mu_f - \xi_f(t))\Delta t + \sigma_{\text{ou},f}\sqrt{\Delta t}\cdot N(0,1)
$$
（$\xi_L$同理，$\mu_f = \mu_L = 0$，围绕基准值波动）

### **3.3 物理控制参数**
$$
f_{\text{cpu}}(t) = \bar{f}_{S(t)} + \xi_f(t) \quad \text{[TEXT Eq.5.1]}
$$
$$
L_{\text{scr}}(t) = \bar{L}_{S(t)} + \xi_L(t) \quad \text{[TEXT Eq.5.1]}
$$
- $\bar{f}_{S}, \bar{L}_{S}$：状态$S$的基准频率和亮度（查找表）

### **3.4 网络尾耗动力学**
$$
\frac{dx_{\text{net}}}{dt} = -\frac{1}{\tau_{\text{tail}}} x_{\text{net}}(t) + \sum_{k=1}^\infty (1-x_{\text{net}}(t^-))\cdot\delta(t-t_k) \quad \text{[TEXT Eq.5.2]}
$$

**物理意义**：
- 空闲期：指数衰减 $x_{\text{net}}(t) = x_{\text{net}}(t_0)e^{-(t-t_0)/\tau_{\text{tail}}}$
- 数据包到达时刻$t_k$：瞬时重置 $x_{\text{net}}(t_k^+) = 1$
- 到达过程：泊松点过程，强度$\lambda_{\text{net}}(S(t))$

**离散实现**：
```python
x_net_new = x_net_prev * exp(-dt/tau_tail)
if packet_arrived_in_dt:
    x_net_new = 1.0
```

### **3.5 功率组件分解**

#### **CPU功耗（动态频率缩放DVFS）**
$$
P_{\text{cpu}}(t) = \alpha \cdot [f_{\text{cpu}}(t)]^3 \quad \text{[TEXT Eq.5.4部分]}
$$
- $\alpha$：CPU功耗系数（W/Hz³）
- 立方关系反映动态电压频率缩放特性

#### **OLED屏幕功耗**
$$
P_{\text{screen}}(t) = P_{\text{driver}} + C_{\text{oled}} \cdot L_{\text{scr}}(t) \cdot [\text{APL}(t)]^\gamma \quad \text{[TEXT Eq.5.3]}
$$
- $P_{\text{driver}}$：屏幕驱动电路基础功耗
- $C_{\text{oled}}$：OLED效率系数
- $\text{APL}(t) = \overline{\text{APL}}_{S(t)}$：平均图像电平（状态依赖）
- $\gamma \approx 2.2$：OLED非线性指数

#### **网络功耗**
$$
P_{\text{net}}(t) = P_{\text{net,idle}} + (P_{\text{net,max}} - P_{\text{net,idle}}) \cdot x_{\text{net}}(t) \quad \text{[推导自TEXT描述]}
$$

#### **系统总功率需求**
$$
\boxed{P_{\text{sys}}(t) = P_{\text{cpu}}(t) + P_{\text{screen}}(t) + P_{\text{net}}(t) + P_{\text{base}}} \quad \text{[更改方案]}
$$
- $P_{\text{base}}$：系统基底功耗（内存、传感器等）

---

## **4. 电化学内核模块 (Electrochemical Core)**

### **4.1 代数约束：电流求解**
**功率平衡方程**：
$$
P_{\text{sys}}(t) = V_{\text{term}}(t) \cdot I(t) \quad \text{(功率守恒)}
$$

**端电压方程（Thevenin等效）**：
$$
V_{\text{term}}(t) = U_{\text{ocv}}(z, T_c) - V_p(t) - I(t) \cdot R_0(z, T_c) \quad \text{[修正自TEXT Eq.5.7]}
$$

**合并得二次本构方程**：
$$
\boxed{R_0 \cdot I^2 - (U_{\text{ocv}} - V_p) \cdot I + P_{\text{sys}} = 0} \quad \text{[TEXT Eq.5.10]}
$$

**判别式与可解性**：
$$
\Delta_{\text{disc}}(t) = (U_{\text{ocv}} - V_p)^2 - 4R_0 P_{\text{sys}} \quad \text{[TEXT Eq.5.11]}
$$

**电流解（物理可行支）**：
$$
I(t) = \frac{(U_{\text{ocv}} - V_p) - \sqrt{\Delta_{\text{disc}}}}{2R_0}, \quad \Delta_{\text{disc}} \geq 0 \quad \text{[TEXT Eq.5.11]}
$$

### **4.2 最大功率极限**
$$
P_{\text{max}}^{\text{limit}}(t) = \frac{(U_{\text{ocv}} - V_p)^2}{4R_0} \quad \text{[TEXT Eq.5.12]}
$$

**物理意义**：当$P_{\text{sys}} > P_{\text{max}}^{\text{limit}}$时，$\Delta_{\text{disc}} < 0$，系统无实数解，触发"电压塌陷"。

### **4.3 SOC演化（库仑积分）**
$$
\frac{dz}{dt} = -\frac{I(t)}{3600 \cdot Q_{\text{design}} \cdot \text{SOH}(t)} \quad \text{[TEXT Eq.5.5]}
$$
- $Q_{\text{design}}$：电池设计容量（Ah）
- $\text{SOH}(t)$：健康状态（State of Health）
- 默认：$\text{SOH}(t) = 1$（若不建模老化）

### **4.4 极化电压演化（浓差极化）**
$$
\frac{dV_p}{dt} = -\frac{1}{\tau_p} V_p(t) + \frac{R_p}{\tau_p} I(t) \quad \text{[TEXT Eq.5.6]}
$$
- $\tau_p = R_p C_p$：极化时间常数（可温度依赖）
- $R_p, C_p$：极化电阻与电容

---

## **5. 热力学模块 (Thermal Dynamics)**

### **5.1 双节点热模型架构**
```
环境(T_amb) ← 对流(h_eff*A_surf) ← 表面(T_s) ← 传导(1/R_th,in) ← 核心(T_c)
                                         ↑
                                     外部热源(η_therm*P_sys)
```

### **5.2 核心温度演化**
$$
C_c \frac{dT_c}{dt} = \underbrace{I^2 R_0}_{\text{焦耳热}} + \underbrace{I T_c \frac{\partial U_{\text{ocv}}}{\partial T}(z, T_c)}_{\text{可逆熵热}} - \underbrace{\frac{T_c - T_s}{R_{\text{th,in}}}}_{\text{内部传导}} \quad \text{[TEXT Eq.5.8]}
$$

**各项物理意义**：
1. **焦耳热**：$I^2R_0$，不可逆热损耗
2. **可逆熵热**：$I T_c (\partial U_{\text{ocv}}/\partial T)$，充放电时的吸放热（Peltier效应）
3. **内部传导**：核心与表面间的热传导

### **5.3 表面温度演化**
$$
C_s \frac{dT_s}{dt} = \underbrace{\frac{T_c - T_s}{R_{\text{th,in}}}}_{\text{来自核心}} - \underbrace{h_{\text{eff}} A_{\text{surf}}(T_s - T_{\text{amb}})}_{\text{对流散热}} + \underbrace{\eta_{\text{therm}} P_{\text{sys}}(t)}_{\text{外部热源}} \quad \text{[TEXT Eq.5.9]}
$$

- $\eta_{\text{therm}}$：系统功耗传导至电池表面的热耦合系数
- $h_{\text{eff}}$：有效对流换热系数
- $A_{\text{surf}}$：电池表面积

---

## **6. 参数温度与SOC依赖**

### **6.1 开路电压 $U_{\text{ocv}}(z, T_c)$**
- 典型形式：$U_{\text{ocv}} = U_{\text{ref}} + k_T(T_c - T_{\text{ref}}) + \text{非线性}(z)$
- 实现方式：查找表或经验公式（如Shepherd模型）

### **6.2 内阻 $R_0(z, T_c)$**
**Arrhenius温度依赖**：
$$
R_0(T_c) = R_{0,\text{ref}} \cdot \exp\left[\frac{E_a}{R}\left(\frac{1}{T_c} - \frac{1}{T_{\text{ref}}}\right)\right]
$$
- $E_a$：活化能
- $R$：气体常数

**SOC依赖**：通常U型曲线，低SOC和高SOC时内阻增大

### **6.3 熵热系数 $\partial U_{\text{ocv}}/\partial T$**
- 实现方式：数值差分或经验公式
- 典型值：-0.1 ~ 0.3 mV/K，随SOC变化

### **6.4 极化参数 $R_p(T_c), C_p(T_c)$**
- 温度升高 → $R_p$减小，$C_p$增大
- $\tau_p = R_p C_p$ 可能相对稳定

---

## **7. 数值求解策略**

### **7.1 时间步进算法（Δt = 1s）**
每个时间步按固定顺序执行：

```python
def time_step(t, state, params):
    # 1. 更新离散随机状态
    state.S = update_user_state(state.S, params)
  
    # 2. 更新OU噪声（Euler-Maruyama）
    state.xi_f = step_ou(state.xi_f, params.tau_ou_f, params.sigma_ou_f, dt)
    state.xi_L = step_ou(state.xi_L, params.tau_ou_L, params.sigma_ou_L, dt)
  
    # 3. 更新网络状态
    state.x_net = step_network(state.x_net, dt, params.tau_tail, packet_arrived)
  
    # 4. 计算系统功率需求
    P_sys = calculate_power(state, params)
  
    # 5. 代数求解电流（关键步骤）
    I, Δ = solve_current(P_sys, state.z, state.Vp, state.Tc, params)
  
    # 6. 可解性检查（失效判据1）
    if Δ < 0:
        raise PowerCollapseError("Δ < 0: 功率超过电池极限")
  
    # 7. 计算端电压
    V_term = calculate_terminal_voltage(I, state, params)
  
    # 8. 边界检查（失效判据2,3）
    if state.z <= 0:
        raise CapacityDepletionError("SOC ≤ 0")
    if V_term <= params.V_cut:
        raise UndervoltageError(f"V_term ≤ {params.V_cut}V")
  
    # 9. 微分状态更新（显式欧拉）
    state.z += soc_rhs(I, params) * dt
    state.Vp += vp_rhs(state.Vp, I, params) * dt
    state.Tc += tc_rhs(I, state, params) * dt
    state.Ts += ts_rhs(state, P_sys, params) * dt
  
    return I, V_term, Δ
```

### **7.2 失效模式定义**
| 失效模式 | 触发条件 | 物理意义 |
|---------|---------|---------|
| **功率崩溃** | $\Delta_{\text{disc}} < 0$ | 负载功率超过电池最大输出能力，电压瞬间塌陷 |
| **容量耗尽** | $z \leq 0$ | SOC降至零，电量完全耗尽 |
| **低压切断** | $V_{\text{term}} \leq V_{\text{cut}}$ | 端电压低于保护阈值，触发UVLO |

### **7.3 首达时间（First Hitting Time）**
仿真持续至首次触发任一失效条件，记录：
- 失效时间 $T_{\text{fail}}$
- 失效模式
- 最终状态 $\mathbf{x}(T_{\text{fail}})$

---

## **8. 模型参数表（最小集）**

### **8.1 电化学参数**
| 参数 | 符号 | 单位 | 典型值/范围 | 说明 |
|------|------|------|-------------|------|
| 设计容量 | $Q_{\text{design}}$ | Ah | 3.0-5.0 | 电池标称容量 |
| 健康状态 | $\text{SOH}(t)$ | - | 1.0 | 容量衰减因子 |
| 开路电压 | $U_{\text{ocv}}(z,T)$ | V | 查表函数 | SOC、温度依赖 |
| 欧姆内阻 | $R_0(z,T)$ | Ω | 0.05-0.15 | 温度、SOC依赖 |
| 极化电阻 | $R_p$ | Ω | 0.02-0.08 | RC电路参数 |
| 极化电容 | $C_p$ | F | 1000-5000 | RC电路参数 |
| 截止电压 | $V_{\text{cut}}$ | V | 3.0-3.3 | 欠压保护阈值 |

### **8.2 热力学参数**
| 参数 | 符号 | 单位 | 典型值/范围 | 说明 |
|------|------|------|-------------|------|
| 核心热容 | $C_c$ | J/K | 60-100 | 电池芯热容 |
| 表面热容 | $C_s$ | J/K | 10-20 | 电池外壳热容 |
| 内部热阻 | $R_{\text{th,in}}$ | K/W | 1-5 | 核心到表面的热阻 |
| 对流系数 | $h_{\text{eff}}$ | W/(m²·K) | 5-15 | 有效换热系数 |
| 表面积 | $A_{\text{surf}}$ | m² | 0.005-0.01 | 电池表面积 |
| 热耦合系数 | $\eta_{\text{therm}}$ | - | 0.1-0.3 | 系统热传导比例 |

### **8.3 激励参数**
| 参数 | 符号 | 单位 | 说明 |
|------|------|------|------|
| CPU系数 | $\alpha$ | W/Hz³ | CPU功耗比例系数 |
| 屏幕驱动功耗 | $P_{\text{driver}}$ | W | OLED驱动电路基础功耗 |
| OLED系数 | $C_{\text{oled}}$ | W/(nits) | 屏幕亮度效率系数 |
| 非线性指数 | $\gamma$ | - | OLED亮度-功耗非线性指数 |
| 网络空闲功耗 | $P_{\text{net,idle}}$ | W | 无线电空闲状态功耗 |
| 网络最大功耗 | $P_{\text{net,max}}$ | W | 无线电激活峰值功耗 |
| 基底功耗 | $P_{\text{base}}$ | W | 系统其他组件基础功耗 |
| 网络尾耗时 | $\tau_{\text{tail}}$ | s | 网络尾耗衰减时间常数 |

### **8.4 随机过程参数**
| 参数 | 符号 | 单位 | 说明 |
|------|------|------|------|
| OU时间常数(CPU) | $\tau_{\text{ou},f}$ | s | CPU频率噪声的均值回归时间 |
| OU强度(CPU) | $\sigma_{\text{ou},f}$ | Hz/√s | CPU频率噪声强度 |
| OU时间常数(屏幕) | $\tau_{\text{ou},L}$ | s | 屏幕亮度噪声的均值回归时间 |
| OU强度(屏幕) | $\sigma_{\text{ou},L}$ | nits/√s | 屏幕亮度噪声强度 |
| 泊松强度 | $\lambda_{\text{net}}(S)$ | 1/s | 数据包到达率（状态依赖） |

---

## **9. 模型验证指标**

### **9.1 归一化均方根误差（NRMSE）**
$$
\text{NRMSE} = \frac{\sqrt{\frac{1}{N}\sum_{i=1}^N (V_{\text{sim},i} - V_{\text{exp},i})^2}}{\max(V_{\text{exp}}) - \min(V_{\text{exp}})} \quad \text{[TEXT Eq.5.13]}
$$

### **9.2 失效时间分布**
- 蒙特卡洛仿真获得$T_{\text{fail}}$的统计分布
- 分析不同失效模式的比例

### **9.3 关键现象复现**
1. **电压纹波**：由网络数据包突发引起的小幅震荡
2. **温度-容量耦合**：低温下容量衰减加速
3. **功率崩溃**：高负载下的突然关机

---

## **10. 模型洞察与应用**

### **10.1 热反馈回路（正反馈机制）**
$$
T_c \downarrow \xrightarrow{\text{Arrhenius}} R_0 \uparrow \xrightarrow{\text{Eq.5.12}} P_{\text{max}}^{\text{limit}} \downarrow \xrightarrow{\Delta<0} \text{关机}
$$
**工程意义**：低温环境下，热管理比电量管理更关键。

### **10.2 网络尾耗的数学本质**
Eq.5.2证明：稀疏的数据包（脉冲）通过**重置机制**，能强制无线电长期维持高能耗状态，解释了"间歇性刷消息比连续下载更耗电"的现象。

### **10.3 可解性边界与用户感知**
$P_{\text{max}}^{\text{limit}}$定义了电池的瞬时功率能力，当用户行为（如启动大型游戏）超过此边界时，触发"突然关机"，而非平滑降频。

---

## **附录：符号索引**

| 符号 | 含义 | 首次出现 |
|------|------|----------|
| $z(t)$ | 电池荷电状态（SOC） | Sec 2.1 |
| $V_p(t)$ | 极化电压 | Sec 2.1 |
| $T_c(t), T_s(t)$ | 核心/表面温度 | Sec 2.1 |
| $x_{\text{net}}(t)$ | 网络尾耗状态 | Sec 2.1 |
| $S(t)$ | 用户行为模式 | Sec 2.2 |
| $\xi_f(t), \xi_L(t)$ | OU噪声过程 | Sec 2.3 |
| $P_{\text{sys}}(t)$ | 系统总功率需求 | Sec 3.5 |
| $I(t)$ | 电池放电电流 | Sec 4.1 |
| $V_{\text{term}}(t)$ | 电池端电压 | Sec 4.1 |
| $\Delta_{\text{disc}}(t)$ | 电流求解判别式 | Sec 4.1 |
| $P_{\text{max}}^{\text{limit}}$ | 最大功率极限 | Sec 4.2 |

---

**文档状态**：完整数学定义，可直接用于实现。

**下一步**：基于此文档进行代码实现，遵循第7节的数值求解策略。