import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

class BatteryModel:
    """
    电池物理模型类，封装电池的电压、内阻和容量特性。
    以后可以通过继承此类或替换实例来更换不同的电池模型。
    """
    def __init__(self, phi=1.0):
        self.phi = phi  # 【未定义变量】暂时当作老化因子/容量衰减系数

    def V_ocv(self, z):
        """【未定义】开路电压 V_ocv(z)"""
        return 3.0 + 1.2 * z

    def R_int(self, T, z):
        """【未定义】内阻 R_int(T, z)"""
        return 0.1 * (1.0 + 0.001 * (300.0 - T))

    def V_terminal(self, z, I, T):
        """端电压 V_batt = V_ocv(z) - I * R_int(T, z)"""
        return self.V_ocv(z) - I * self.R_int(T, z)

    def calculate_current(self, P_total, T, z):
        """根据所需总功率 P_total 计算负载电流 I_load (解 P = (V_ocv - I*R)*I)"""
        Vocv = self.V_ocv(z)
        Rint = self.R_int(T, z)
        
        # 判别式 D = Vocv^2 - 4 * Rint * P_total
        delta = Vocv**2 - 4 * Rint * P_total
        if delta < 0:
            # 如果功率过大，电池无法提供，则限制在最大功率点电流
            return Vocv / (2 * Rint)
        else:
            # 取较小的根（正常工作区间）
            return (Vocv - np.sqrt(delta)) / (2 * Rint)

    def Q_max(self, T):
        """【未定义】最大可用容量 Q_max(T)仅标注了和T与Phi有关，这里当作和Phi线性关系"""
        return 3600.0 * self.phi



class ScreenComponent:
    """屏幕组件"""
    def __init__(self, k_s=1.0, gamma=1.0):
        self.k_s = k_s
        """【参数】屏幕功耗系数"""
        self.gamma = gamma
        """【参数】屏幕亮度非线性指数"""

    def S_on(self, t):
        """【未定义】屏幕开关状态
        暂时设定为在600s和3000s之间关闭，其他时间开启
        暂时通过写死的表达式
        """
        return 1.0 if (t < 600 or t > 3000) else 0.0

    def b_screen(self, t):
        """【未定义】屏幕亮度"""
        return 0.8

    def power(self, t):
        """屏幕功耗 P_scr(t) = k_s * S_on(t) * b(t)^gamma"""
        return self.k_s * self.S_on(t) * (self.b_screen(t) ** self.gamma)

class CPUComponent:
    """CPU 组件"""
    def __init__(self, k_c=1.0, P0_leak=1.0, beta_leak=0.01, T_ref=300.0):
        self.k_c = k_c
        """【参数】CPU 功耗系数，是u(t)f^3(t)的缩放系数"""
        self.P0_leak = P0_leak
        """【参数】漏电功率？"""
        self.beta_leak = beta_leak
        """【参数】控制温差影响漏电功率的系数（位于非线性部分）"""
        self.T_ref = T_ref
        """【参数】漏电功率参考温度"""

    def u_cpu(self, t):
        """【未定义】CPU 利用率"""
        return 0.2 + 0.1 * np.sin(2 * np.pi * t / 600)

    def f_cpu(self, t):
        """【未定义】CPU 频率 这里当作是利用率的仿射变换"""
        util = self.u_cpu(t)
        return 0.5 + 0.5 * util

    def P_leak(self, T):
        """CPU 漏电功耗"""
        return self.P0_leak * np.exp(self.beta_leak * (T - self.T_ref))

    def power(self, t, T):
        """CPU 功耗 P_cpu(t) = k_c u(t) f(t)^3 + P_leak(T)"""
        return self.k_c * self.u_cpu(t) * (self.f_cpu(t) ** 3) + self.P_leak(T)

class NetworkComponent:
    """网络组件"""
    def __init__(self, k_tx=1.0, k_rx=1.0, k_tail=1.0, tau_tail=1.0, alpha_tail=1.0):
        self.k_tx = k_tx
        """【参数】短时功耗系数"""
        self.k_rx = k_rx
        """【参数】长时功耗系数"""
        self.k_tail = k_tail
        """【参数】网络尾耗（Tail Power）功率系数"""
        self.tau_tail = tau_tail
        """【参数】网络尾耗状态的时间常数，控制尾耗衰减速度"""
        self.alpha_tail = alpha_tail
        """【参数】网络尾耗状态的触发系数，控制进入尾耗状态的速度"""

    def r_tx(self, t):
        """【未定义】短时功耗部分"""
        return 1.0 if (t % 300 < 10) else 0.0

    def r_rx(self, t):
        """【未定义】长时功耗部分"""
        return 0.1

    def a_net(self, t):
        """【未定义】网络触发脉冲"""
        return 1.0 if (self.r_tx(t) > 0 or self.r_rx(t) > 0) else 0.0

    def power(self, t, y_tail):
        """网络功耗 P_net(t) = k_tx r_tx + k_rx r_rx + k_tail y(t)"""
        return self.k_tx * self.r_tx(t) + self.k_rx * self.r_rx(t) + self.k_tail * y_tail

class GPSComponent:
    """GPS 组件"""
    def power(self, t):
        """【未定义】GPS 功耗"""
        return 0.1

class LogicComponent:
    """逻辑/SoC芯片组件"""
    def power(self, t):
        """【未定义】逻辑功耗"""
        return 0.05



class TotalPowerModel:
    """
    总功率模型，聚合各组件并计算总功率。
    """
    def __init__(self, screen, cpu, network, gps, logic, P_base=0.1):
        self.screen = screen
        self.cpu = cpu
        self.network = network
        self.gps = gps
        self.logic = logic
        self.P_base = P_base
        '''【参数】基础功耗'''

    def calculate(self, t, T, y_tail):
        """
        计算总功率 P_tot
        """
        p_scr = self.screen.power(t)
        p_cpu = self.cpu.power(t, T)
        p_net = self.network.power(t, y_tail)
        p_gps = self.gps.power(t)
        p_logic = self.logic.power(t)
        
        return self.P_base + p_scr + p_cpu + p_net + p_gps + p_logic



class BatteryThermalModel:
    """
    电池热仿真引擎类，负责 ODE 方程的组装与求解。
    """
    def __init__(self, battery: BatteryModel, power_model: TotalPowerModel):
        self.battery = battery
        """电池物理模型实例"""
        self.power_model = power_model
        """总功率模型实例"""
        
        # 热模型参数
        self.C_th = 1.0
        """【参数】热容 (Thermal Capacity)"""
        self.hA = 1.0
        """【参数】综合散热系数 (Heat Transfer Coefficient * Area)"""
        self.eta_heat = 1.0
        """【参数】逻辑功耗转化为热能的比例系数"""

    def T_env(self, t):
        """【参数】环境温度"""
        return 298.0

    def ode_system(self, t, x):
        """
        主微分方程组: x = [z, T, y]
        """
        z, T, y = x

        # 1计算总功率
        P_total = self.power_model.calculate(t, T, y)
        
        # 2计算电流和端电压 (由电池模型内部处理)
        I_load = self.battery.calculate_current(P_total, T, z)
        V_batt = self.battery.V_terminal(z, I_load, T)

        # 1.SOC 演化方程
        dzt = - I_load / self.battery.Q_max(T)

        # 2. 热力学演化方程
        # 产热项 P_heat = I^2 * R_int + eta * P_logic
        P_heat = (I_load ** 2) * self.battery.R_int(T, z) + self.eta_heat * self.power_model.logic.power(t)
        dTt = (P_heat - self.hA * (T - self.T_env(t))) / self.C_th

        # 3. 网络尾耗演化方程 (使用 network 组件的参数)
        net = self.power_model.network
        dyt = - (1.0 / net.tau_tail) * y + net.alpha_tail * net.a_net(t) * (1.0 - y)

        return [dzt, dTt, dyt]

    def simulate(self, z0=1.0, T0=300.0, y0=0.0, t_span=(0.0, 3600.0), num_points=1000):
        x0 = [z0, T0, y0]
        t_eval = np.linspace(t_span[0], t_span[1], num_points)
        
        sol = solve_ivp(
            fun=self.ode_system,
            t_span=t_span,
            y0=x0,
            t_eval=t_eval,
            vectorized=False
        )
        return sol

def main():
    # 1. 初始化各原子组件
    battery = BatteryModel(phi=1.0)
    screen = ScreenComponent()
    cpu = CPUComponent()
    network = NetworkComponent()
    gps = GPSComponent()
    logic = LogicComponent()
    
    # 2. 组装总功率模型
    power_model = TotalPowerModel(screen, cpu, network, gps, logic)
    
    # 3. 初始化仿真引擎
    engine = BatteryThermalModel(battery, power_model)
    
    # 4. 运行仿真
    sol = engine.simulate()
    t = sol.t
    z, T, y = sol.y
    
    # 计算 V_batt 序列用于绘图
    v_batt_list = []
    for i in range(len(t)):
        zi, Ti, yi = z[i], T[i], y[i]
        p_tot = power_model.calculate(t[i], Ti, yi)
        cur_i = battery.calculate_current(p_tot, Ti, zi)
        v_batt_list.append(battery.V_terminal(zi, cur_i, Ti))
    
    # 5. 绘图
    plt.figure(figsize=(12, 12))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, z, label='SOC', color='green')
    plt.title('Battery System Simulation (Component-Based with V_batt)')
    plt.ylabel('SOC')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(t, T, label='Temperature', color='red')
    plt.ylabel('Temp (K)')
    plt.grid(True)
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(t, y, label='Network Tail State', color='blue')
    plt.ylabel('Tail State')
    plt.grid(True)
    plt.legend()

    plt.subplot(4, 1, 4)
    plt.plot(t, v_batt_list, label='Terminal Voltage (V_batt)', color='purple')
    plt.xlabel('Time (s)')
    plt.ylabel('Voltage (V)')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    
    output_path = "ProblemA/battery_simulation_components.png"
    plt.savefig(output_path)
    print(f"Simulation plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    main()
