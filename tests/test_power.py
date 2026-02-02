import pytest
import numpy as np
from mp_shm_x2.config import SimulationParams
from mp_shm_x2.stochastic import OUProcess, UserBehaviorModel
from mp_shm_x2.power import CPUPower, ScreenPower, NetworkPower, SystemPower

def test_ou_process():
    params = SimulationParams()
    ou = OUProcess(tau=10.0, sigma=1.0, dt=1.0)
    
    # Check if state updates
    initial_val = ou.value
    new_val = ou.step()
    
    # It's random, but should be float
    assert isinstance(new_val, float)
    
    # Statistical properties check would require many steps, skipping for unit test
    
def test_user_behavior_model():
    ub = UserBehaviorModel()
    assert ub.current_state == "Idle"
    
    ub.set_state("Game")
    assert ub.current_state == "Game"
    
    with pytest.raises(ValueError):
        ub.set_state("InvalidState")

def test_cpu_power():
    params = SimulationParams()
    # Mock alpha_cpu to 1.0 for easy calc
    params.alpha_cpu = 1.0
    params.sigma_ou_f = 0.0 # No noise
    
    cpu = CPUPower(params)
    p = cpu.get_power(base_freq=2.0)
    
    # 1.0 * 2.0^3 = 8.0
    assert abs(p - 8.0) < 1e-6

def test_screen_power():
    params = SimulationParams()
    params.P_driver = 0.1
    params.C_oled = 1.0
    params.gamma_oled = 2.0
    params.sigma_ou_L = 0.0 # No noise
    
    screen = ScreenPower(params)
    p = screen.get_power(base_brightness=10.0, apl=0.5)
    
    # P = 0.1 + 1.0 * 10.0 * (0.5^2) = 0.1 + 10 * 0.25 = 2.6
    assert abs(p - 2.6) < 1e-6

def test_network_power():
    params = SimulationParams()
    params.P_net_idle = 0.1
    params.P_net_max = 1.1
    params.tau_tail = 1.0
    params.dt = 1.0 # exp(-1) decay
    
    net = NetworkPower(params)
    
    # Initial state 0 -> Power should be idle
    p0 = net.step(packet_arrival=False)
    assert abs(p0 - 0.1) < 1e-6
    
    # Packet arrival -> x_net becomes 1.0
    p1 = net.step(packet_arrival=True)
    # P = 0.1 + (1.1 - 0.1) * 1.0 = 1.1
    assert abs(p1 - 1.1) < 1e-6
    assert net.x_net == 1.0
    
    # Decay step
    p2 = net.step(packet_arrival=False)
    # x_net = 1.0 * exp(-1) approx 0.3678
    # P = 0.1 + 1.0 * 0.3678...
    expected_x = np.exp(-1)
    expected_p = 0.1 + 1.0 * expected_x
    assert abs(p2 - expected_p) < 1e-6

def test_system_power():
    params = SimulationParams()
    sys_power = SystemPower(params)
    
    # Just check if it runs without error and returns positive power
    p = sys_power.get_total_power(1.0, 1.0, 0.5, False)
    assert p > 0.0
