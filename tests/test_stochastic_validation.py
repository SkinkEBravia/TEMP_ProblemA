import pytest
import numpy as np
from mp_shm_x2.stochastic import UserBehaviorModel

def test_transition_validation_success():
    """Test that a valid transition matrix passes validation."""
    model = UserBehaviorModel()
    # The default matrix should be valid
    # No error raised means success
    
def test_transition_validation_sum_error():
    """Test that probabilities not summing to 1 raises ValueError."""
    model = UserBehaviorModel()
    # Manually break the matrix
    model.transitions["Idle"] = {"Video": 0.5} # Sums to 0.5
    
    with pytest.raises(ValueError, match="sum to 0.5"):
        model._validate_transitions()

def test_transition_validation_negative_error():
    """Test that negative probabilities raise ValueError."""
    model = UserBehaviorModel()
    model.transitions["Idle"] = {"Video": 1.1, "Game": -0.1} # Sums to 1.0 but has negative
    
    with pytest.raises(ValueError, match="Negative probability"):
        model._validate_transitions()

def test_transition_validation_unknown_state():
    """Test that transitioning to an unknown state raises ValueError."""
    model = UserBehaviorModel()
    model.transitions["Idle"] = {"Video": 0.5, "UnknownState": 0.5}
    
    with pytest.raises(ValueError, match="not in known states"):
        model._validate_transitions()
