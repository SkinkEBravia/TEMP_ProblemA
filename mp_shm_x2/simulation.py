"""
Main Simulation Engine.

Integrates all modules to perform time-domain simulation.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from .config import SimulationParams, get_dudt
from .stochastic import UserBehaviorModel
from .power import SystemPower
from .battery import BatteryModel, PowerCollapseError
from .thermal import ThermalModel

@dataclass
class SimulationResult:
    """Stores time-series results."""
    time: List[float] = field(default_factory=list)
    z: List[float] = field(default_factory=list)
    V_term: List[float] = field(default_factory=list)
    I: List[float] = field(default_factory=list)
    Tc: List[float] = field(default_factory=list)
    Ts: List[float] = field(default_factory=list)
    P_sys: List[float] = field(default_factory=list)
    P_cpu: List[float] = field(default_factory=list) # Added
    P_screen: List[float] = field(default_factory=list) # Added
    P_net: List[float] = field(default_factory=list) # Added
    Vp: List[float] = field(default_factory=list) # Added Vp tracking
    State: List[str] = field(default_factory=list)
    Failure: Optional[str] = None

class SimulationEngine:
    """
    Orchestrates the Multi-Physics Stochastic Hybrid Model simulation.
    """
    
    def __init__(self, params: SimulationParams = None):
        self.params = params if params else SimulationParams()
        
        # Modules
        # Pass params to UserBehaviorModel now
        self.user_model = UserBehaviorModel(self.params)
        self.power_model = SystemPower(self.params)
        self.battery_model = BatteryModel(self.params)
        self.thermal_model = ThermalModel(self.params)
        
        # State Variables
        self.t = 0.0
        self.z = 1.0 # Initial SOC
        self.Vp = 0.0
        self.Tc = self.params.T_amb
        self.Ts = self.params.T_amb
        
        # Inputs (from user model / stochastic)
        # Placeholders for now, would come from S(t) -> Lookup Table
        self.base_freq = 1.0 # Hz (Normalized?)
        self.base_brightness = 500.0 # nits
        self.screen_apl = 0.5
        
        # Result Storage
        self.results = SimulationResult()

    def _get_inputs_for_state(self, state: str):
        """
        Map user state to physical inputs. 
        In a real app, this would be a config lookup.
        """
        if state == "Idle":
            return 0.2, 0.0, 0.0 # Low freq, screen off
        elif state == "Video":
            return 1.0, 400.0, 0.6
        elif state == "Game":
            return 2.5, 800.0, 0.8
        elif state == "Call":
            return 0.5, 0.0, 0.0 # Screen off, some CPU
        return 0.5, 200.0, 0.5

    def step(self):
        """
        Execute one time step of the simulation.
        """
        dt = self.params.dt
        
        # 1. Update Discrete State
        current_user_state = self.user_model.step(dt)
        
        # Map state to inputs
        # (This is a simplification; in full model, these come from lookups)
        f_cpu, L_scr, APL = self._get_inputs_for_state(current_user_state)
        
        # Network packet arrival (Stochastic Poisson process)
        # Simplified: Random check based on state?
        # For now: No packets unless in specific states
        packet_arrival = False
        if current_user_state in ["Game", "Video"]:
             if np.random.random() < 0.1: # 10% chance per second
                 packet_arrival = True

        # 2. Calculate System Power
        # Note: Stochastic updates happen inside power_model components
        p_cpu = self.power_model.cpu.get_power(f_cpu)
        p_screen = self.power_model.screen.get_power(L_scr, APL)
        p_net = self.power_model.net.step(packet_arrival)
        P_sys = p_cpu + p_screen + p_net + self.power_model.base_power
        
        # 3. Solve Current (Algebraic Loop)
        try:
            I, V_term, delta = self.battery_model.solve_current(P_sys, self.z, self.Vp, self.Tc)
        except PowerCollapseError as e:
            return "Power Collapse"
        
        # 4. Check Limits
        if self.z <= 0:
            return "Capacity Depletion"
        
        if V_term <= self.params.V_cut:
            return "Undervoltage"
        
        # 5. Calculate Derivatives
        dz_dt = self.battery_model.get_soc_derivative(I)
        dVp_dt = self.battery_model.get_vp_derivative(self.Vp, I, self.Tc)
        
        # dU/dT can be functional now
        # Note: In config.py we implemented get_dudt, but we need to check how to access it
        # Actually it's imported in simulation? No, we need to use it.
        # But wait, config.py defines get_dudt but BatteryModel doesn't wrap it yet or we use it directly?
        # Let's use the one from config if imported, or better, access via params if we linked it?
        # Currently get_dudt is a standalone function in config.py
        from .config import get_dudt
        dudt = get_dudt(self.z, self.Tc, self.params)
        
        dTc_dt = self.thermal_model.get_core_temp_derivative(self.Tc, self.Ts, I, 
                                                             self.battery_model.get_r0(self.z, self.Tc), 
                                                             dudt)
        
        dTs_dt = self.thermal_model.get_surface_temp_derivative(self.Tc, self.Ts, P_sys)
        
        # 6. Update States (Explicit Euler)
        self.z += dz_dt * dt
        self.Vp += dVp_dt * dt
        self.Tc += dTc_dt * dt
        self.Ts += dTs_dt * dt
        self.t += dt
        
        # 7. Record Results
        self.results.time.append(self.t)
        self.results.z.append(self.z)
        self.results.V_term.append(V_term)
        self.results.I.append(I)
        self.results.Tc.append(self.Tc)
        self.results.Ts.append(self.Ts)
        self.results.P_sys.append(P_sys)
        self.results.P_cpu.append(p_cpu)
        self.results.P_screen.append(p_screen)
        self.results.P_net.append(p_net)
        self.results.Vp.append(self.Vp) # Record Vp
        self.results.State.append(current_user_state)
        
        return None # No failure

    def run(self, duration: float) -> None:
        """
        Run simulation for a fixed duration or until failure.
        """
        steps = int(duration / self.params.dt)
        
        # Record initial state
        self.results.time.append(self.t)
        self.results.z.append(self.z)
        # Initial V_term guess (OCV)
        self.results.V_term.append(self.battery_model.get_ocv(self.z, self.Tc)) 
        self.results.I.append(0.0)
        self.results.Tc.append(self.Tc)
        self.results.Ts.append(self.Ts)
        self.results.P_sys.append(0.0)
        self.results.P_cpu.append(0.0)
        self.results.P_screen.append(0.0)
        self.results.P_net.append(0.0)
        self.results.Vp.append(self.Vp) # Record initial Vp
        self.results.State.append(self.user_model.current_state)
        
        for _ in range(steps):
            failure = self.step()
            if failure:
                self.results.Failure = failure
                print(f"Simulation stopped: {failure} at t={self.t:.2f}s")
                break
