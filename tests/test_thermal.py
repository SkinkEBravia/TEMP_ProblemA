import pytest
from mp_shm_x2.config import SimulationParams
from mp_shm_x2.thermal import ThermalModel

def test_thermal_derivatives():
    params = SimulationParams()
    # Simplify parameters for calculation check
    params.C_c = 100.0
    params.C_s = 10.0
    params.R_th_in = 1.0
    params.h_eff = 10.0
    params.A_surf = 0.1
    params.T_amb = 300.0
    params.eta_therm = 0.5
    
    thermal = ThermalModel(params)
    
    Tc = 310.0
    Ts = 305.0
    I = 2.0
    R0 = 0.1
    dudt = 0.0 # Ignore reversible heat for simple check
    P_sys = 10.0
    
    # Core Derivative Check
    # q_joule = 2^2 * 0.1 = 0.4
    # q_rev = 0
    # q_cond = (310 - 305) / 1.0 = 5.0
    # dTc = (0.4 + 0 - 5.0) / 100.0 = -4.6 / 100 = -0.046
    dTc = thermal.get_core_temp_derivative(Tc, Ts, I, R0, dudt)
    assert abs(dTc - (-0.046)) < 1e-6
    
    # Surface Derivative Check
    # q_cond = 5.0
    # q_conv = 10 * 0.1 * (305 - 300) = 1.0 * 5 = 5.0
    # q_ext = 0.5 * 10.0 = 5.0
    # dTs = (5.0 - 5.0 + 5.0) / 10.0 = 0.5
    dTs = thermal.get_surface_temp_derivative(Tc, Ts, P_sys)
    assert abs(dTs - 0.5) < 1e-6

def test_thermal_equilibrium():
    """Test if temps stay constant when everything is in equilibrium."""
    params = SimulationParams()
    params.T_amb = 300.0
    thermal = ThermalModel(params)
    
    Tc = 300.0
    Ts = 300.0
    I = 0.0
    R0 = 0.1
    dudt = 0.0
    P_sys = 0.0
    
    dTc = thermal.get_core_temp_derivative(Tc, Ts, I, R0, dudt)
    dTs = thermal.get_surface_temp_derivative(Tc, Ts, P_sys)
    
    assert dTc == 0.0
    assert dTs == 0.0
