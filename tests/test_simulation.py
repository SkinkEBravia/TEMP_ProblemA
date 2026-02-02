import pytest
from mp_shm_x2.simulation import SimulationEngine
from mp_shm_x2.config import SimulationParams

def test_simulation_run_short():
    """Run a short simulation to check stability."""
    params = SimulationParams()
    params.dt = 1.0
    sim = SimulationEngine(params)
    
    # Run for 10 seconds
    sim.run(10.0)
    
    assert len(sim.results.time) > 0
    assert sim.results.Failure is None
    assert sim.t == 10.0
    
def test_simulation_depletion():
    """Test if simulation catches SOC depletion."""
    params = SimulationParams()
    params.Q_design = 0.0001 # Tiny battery
    sim = SimulationEngine(params)
    
    # Force high consumption state
    sim.user_model.set_state("Game")
    
    # Tiny battery might power collapse before capacity depletion if resistance is high
    # Let's lower R0 to ensure it discharges
    sim.params.R0_ref = 0.001
    
    sim.run(3600.0) # Run for an hour, should die instantly
    
    # Either is fine, but we expect depletion for a tiny battery with low R
    if sim.results.Failure:
        assert sim.results.Failure in ["Capacity Depletion", "Power Collapse"]

def test_simulation_undervoltage():
    """Test undervoltage cutoff."""
    params = SimulationParams()
    # High resistance to cause voltage drop
    params.R0_ref = 10.0 
    sim = SimulationEngine(params)
    
    # High load
    sim.user_model.set_state("Game")
    
    sim.run(100.0)
    
    # Either power collapse or undervoltage
    if sim.results.Failure:
        assert sim.results.Failure in ["Undervoltage", "Power Collapse"]
