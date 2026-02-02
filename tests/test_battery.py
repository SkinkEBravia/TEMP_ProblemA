import pytest
import numpy as np
from mp_shm_x2.config import SimulationParams
from mp_shm_x2.battery import BatteryModel, PowerCollapseError

def test_battery_r0_arrhenius():
    params = SimulationParams()
    params.R0_ref = 0.1
    params.E_a = 0.0 # No temp dependence
    
    bat = BatteryModel(params)
    
    # With Ea=0, R0 should be constant
    r0 = bat.get_r0(0.5, 300.0)
    assert abs(r0 - 0.1) < 1e-6
    
    # With Ea > 0
    params.E_a = 20000.0
    params.T_ref = 300.0
    bat = BatteryModel(params)
    
    # At T_ref, R0 = R0_ref
    r0_ref = bat.get_r0(0.5, 300.0)
    assert abs(r0_ref - 0.1) < 1e-6
    
    # At lower temp, R0 should increase
    r0_cold = bat.get_r0(0.5, 280.0)
    assert r0_cold > 0.1

def test_current_solver_normal():
    params = SimulationParams()
    params.R0_ref = 0.1
    params.E_a = 0.0
    bat = BatteryModel(params)
    
    # U_ocv approx 3.5V for z=0.5 in our simple mock
    # Vp = 0
    # P_sys = 3.5W
    # V_eff = 3.5
    # Delta = 3.5^2 - 4 * 0.1 * 3.5 = 12.25 - 1.4 = 10.85 > 0
    
    z = 0.5
    Tc = 300.0
    Vp = 0.0
    P_sys = 3.5
    
    I, V_term, delta = bat.solve_current(P_sys, z, Vp, Tc)
    
    # Check power balance
    # P = V * I
    assert abs(V_term * I - P_sys) < 1e-6
    assert delta > 0

def test_current_solver_collapse():
    params = SimulationParams()
    params.R0_ref = 1.0 # High resistance
    params.E_a = 0.0
    bat = BatteryModel(params)
    
    z = 0.5 # U_ocv = 3.5V
    Tc = 300.0
    Vp = 0.0
    
    # Max power = U^2 / 4R = 3.5^2 / 4 = 12.25 / 4 = 3.0625 W
    
    # Request 4.0 W -> Should fail
    with pytest.raises(PowerCollapseError):
        bat.solve_current(4.0, z, Vp, Tc)

def test_soc_derivative():
    params = SimulationParams()
    params.Q_design = 1.0 # 1 Ah
    params.SOH = 1.0
    bat = BatteryModel(params)
    
    I = 3600.0 # 1C rate in Coulombs? No, I is Amps. 
    # If I = 1A, it takes 1 hour to discharge 1Ah.
    # dz/dt should be -1/3600
    
    dz = bat.get_soc_derivative(1.0)
    assert abs(dz - (-1.0/3600.0)) < 1e-6

def test_vp_derivative():
    params = SimulationParams()
    params.tau_p = 10.0
    params.Rp_ref = 0.1
    params.Cp_ref = 100.0 # tau = 10
    # Actually Rp and Cp are functional now, so we need to ensure tau is what we expect
    # Rp = Rp_ref * exp(...), Cp = Cp_ref
    # Let's just mock params to make arrhenius factor 1
    params.E_a = 0.0
    
    bat = BatteryModel(params)
    
    # Steady state check: dVp/dt = 0 => Vp = Rp * I
    # Rp = 0.1, I = 10.0 -> Vp = 1.0
    I = 10.0
    Vp_steady = 1.0 # 0.1 * 10
    Tc = 300.0
    
    dv = bat.get_vp_derivative(Vp_steady, I, Tc)
    assert abs(dv) < 1e-6
