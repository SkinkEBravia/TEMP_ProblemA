"""
Power consumption module.

Calculates power for CPU, Screen, and Network components, and aggregates system power.
"""

import numpy as np
from .config import SimulationParams
from .stochastic import OUProcess, UserBehaviorModel

class CPUPower:
    """
    CPU Power Model.
    P_cpu = alpha * f_cpu^3
    f_cpu = f_base + xi_f
    [TEXT Eq.5.4]
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.ou_process = OUProcess(params.tau_ou_f, params.sigma_ou_f, params.dt)

    def get_power(self, base_freq: float) -> float:
        """
        Calculate CPU power.
        
        Args:
            base_freq: [arg] Base frequency for current state in Hz.
            
        Returns:
            Power in Watts.
        [TEXT Eq.5.4]
        """
        noise = self.ou_process.step()
        freq = base_freq + noise
        freq = max(0.0, freq) # Frequency cannot be negative
        
        p_cpu = self.params.alpha_cpu * (freq ** 3)
        return p_cpu

class ScreenPower:
    """
    Screen Power Model.
    P_screen = P_driver + C_oled * L_scr * APL^gamma
    L_scr = L_base + xi_L
    [TEXT Eq.5.3]
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.ou_process = OUProcess(params.tau_ou_L, params.sigma_ou_L, params.dt) 

    def get_power(self, base_brightness: float, apl: float) -> float:
        """
        Calculate Screen power.
        
        Args:
            base_brightness: [arg] Base brightness in nits.
            apl: [arg] Average Picture Level (0.0 to 1.0).
            
        Returns:
            Power in Watts.
        [TEXT Eq.5.3]
        """
        noise = self.ou_process.step()
        brightness = base_brightness + noise
        brightness = max(0.0, brightness)
        
        p_screen = self.params.P_driver + \
                   self.params.C_oled * brightness * (apl ** self.params.gamma_oled)
        return p_screen

class NetworkPower:
    """
    Network Power Model with Tail State.
    
    dx_net/dt = -1/tau * x_net + impulses
    P_net = P_idle + (P_max - P_idle) * x_net
    [TEXT Eq.5.2] and Derived Solution
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.x_net = 0.0
    
    def step(self, packet_arrival: bool) -> float:
        """
        Update network state and calculate power.
        
        Args:
            packet_arrival: [arg] True if a packet arrived in this time step.
            
        Returns:
            Power in Watts.
        """
        # Decay
        # x(t+dt) = x(t) * exp(-dt/tau)
        decay = np.exp(-self.params.dt / self.params.tau_tail)
        self.x_net = self.x_net * decay
        
        # Impulse reset
        if packet_arrival:
            self.x_net = 1.0
            
        # Calculate Power
        p_net = self.params.P_net_idle + \
                (self.params.P_net_max - self.params.P_net_idle) * self.x_net
        
        return p_net

class SystemPower:
    """
    Aggregates all power components.
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        self.cpu = CPUPower(params)
        self.screen = ScreenPower(params)
        self.net = NetworkPower(params)
        self.base_power = params.P_base
        
    def get_total_power(self, 
                        cpu_freq: float, 
                        screen_brightness: float, 
                        screen_apl: float, 
                        packet_arrival: bool) -> float:
        """
        Calculate total system power.
        
        Args:
            cpu_freq: [arg] Base CPU frequency.
            screen_brightness: [arg] Base screen brightness.
            screen_apl: [arg] Screen APL.
            packet_arrival: [arg] Network packet arrival flag.
            
        Returns:
            Total power in Watts.
        """
        p_cpu = self.cpu.get_power(cpu_freq)
        p_screen = self.screen.get_power(screen_brightness, screen_apl)
        p_net = self.net.step(packet_arrival)
        
        p_total = p_cpu + p_screen + p_net + self.base_power
        return p_total
