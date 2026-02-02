"""
Stochastic processes module.

Handles Ornstein-Uhlenbeck (OU) noise processes and discrete state transitions (Semi-Markov Chain).
"""

import numpy as np
from typing import Dict, Optional, List, Any, Callable
from .user_config import STATES, TRANSITIONS, sample_dwelling_time

class OUProcess:
    """
    Ornstein-Uhlenbeck process generator.
    
    Equation:
    dxi = -1/tau * xi * dt + sigma * dW
    """
    
    def __init__(self, tau: float, sigma: float, dt: float):
        """
        Initialize the OU process.
        
        Args:
            tau: [arg] Mean reversion time constant.
            sigma: [arg] Volatility parameter.
            dt: [arg] Time step.
        """
        self.tau = tau
        self.sigma = sigma
        self.dt = dt
        self.value = 0.0

    def step(self) -> float:
        """
        Perform one time step update using Euler-Maruyama method.
        
        Returns:
            New value of the process.
        """
        # dW ~ N(0, dt) => sqrt(dt) * N(0, 1)
        dw = np.sqrt(self.dt) * np.random.normal(0, 1)
        
        # xi(t+dt) = xi(t) - (1/tau)*xi(t)*dt + sigma*dW
        drift = -(1.0 / self.tau) * self.value * self.dt
        diffusion = self.sigma * dw
        
        self.value += drift + diffusion
        return self.value

    def reset(self):
        """Reset the process to zero."""
        self.value = 0.0

class UserBehaviorModel:
    """
    Simulates user behavior states using an Advanced Semi-Markov chain.
    
    Features:
    1. Flexible State Transition Matrix.
    2. History-dependent dwelling times (e.g., Idle after Camera is short).
    3. Time-dependent dwelling times (Circadian Rhythm via T_clock).
    4. Multi-modal distributions (e.g., Bimodal Idle).
    """
    
    def __init__(self, 
                 params: Optional[Any] = None, 
                 initial_state: str = "Idle", 
                 t_clock_init: float = 0.0,
                 states: List[str] = None,
                 transitions: Dict[str, Dict[str, float]] = None,
                 dwelling_time_fn: Callable = None):
        
        self.params = params
        self.current_state = initial_state
        self.time_in_state = 0.0
        
        # Time of day in seconds (0 to 86400). Rolls over.
        self.t_clock = t_clock_init 
        
        # History of states [S_{k-N}, ..., S_{k-1}]
        self.history: List[str] = []
        self.max_history = 5
        
        # --- Configuration Section ---
        self.states = states if states is not None else STATES
        self.transitions = transitions if transitions is not None else TRANSITIONS
        self.dwelling_time_fn = dwelling_time_fn if dwelling_time_fn is not None else sample_dwelling_time
        
        # Initial sample
        self._validate_transitions()
        self.target_dwelling_time = self._sample_dwelling_time(self.current_state)

    def _validate_transitions(self):
        """
        Validate the transition matrix.
        Checks:
        1. All probabilities are non-negative.
        2. Probabilities for each state sum to 1.0 (with tolerance).
        3. All target states are defined in self.states.
        """
        for src_state, targets in self.transitions.items():
            if src_state not in self.states:
                # Optionally warn or auto-add, but for strict validation we might warn
                pass # self.states is just a list, transition keys define the graph
                
            total_prob = 0.0
            for dst_state, prob in targets.items():
                if prob < 0:
                    raise ValueError(f"Negative probability {prob} for transition {src_state}->{dst_state}")
                if dst_state not in self.states:
                    raise ValueError(f"Target state '{dst_state}' in transition {src_state}->{dst_state} is not in known states list.")
                total_prob += prob
                
            if not np.isclose(total_prob, 1.0):
                raise ValueError(f"Transition probabilities for '{src_state}' sum to {total_prob}, expected 1.0")

    def _sample_dwelling_time(self, state: str) -> float:
        """
        Sample dwelling time based on State, History, and T_clock.
        Delegates to injected dwelling_time_fn.
        """
        return self.dwelling_time_fn(state, self.history, self.t_clock, self.params)

    def _get_next_state(self, current_state: str) -> str:
        """Select next state based on Transition Matrix."""
        if current_state not in self.transitions:
            return "Idle" # Fallback
            
        trans_probs = self.transitions[current_state]
        candidates = list(trans_probs.keys())
        probs = list(trans_probs.values())
        
        # Normalize probs if needed
        total = sum(probs)
        probs = [p/total for p in probs]
        
        return np.random.choice(candidates, p=probs)
        
    def step(self, dt: float) -> str:
        """
        Update user state and clock.
        
        Args:
            dt: [arg] Time step.
            
        Returns:
            Current state name.
        """
        self.time_in_state += dt
        self.t_clock = (self.t_clock + dt) % 86400.0 # Roll over 24h
        
        if self.time_in_state >= self.target_dwelling_time:
            # 1. Record History
            self.history.append(self.current_state)
            if len(self.history) > self.max_history:
                self.history.pop(0)
            
            # 2. Transition
            next_state = self._get_next_state(self.current_state)
            
            # Handle self-transition if needed (reset timer)
            # In Semi-Markov, self-transition usually means "renew" dwell time
            # or we can treat it as staying. Here we treat as new phase.
            self.current_state = next_state
            self.time_in_state = 0.0
            
            # 3. Sample new dwelling time
            self.target_dwelling_time = self._sample_dwelling_time(self.current_state)
            
        return self.current_state

    def set_state(self, state: str):
        """Force a state change."""
        if state in self.states:
            self.current_state = state
            self.time_in_state = 0.0
            self.target_dwelling_time = self._sample_dwelling_time(state)
        else:
            # Allow setting unknown states for testing but warn/add to list
            # Wait, the test expects a ValueError for unknown states.
            # Let's revert to strict checking to satisfy the test contract.
            raise ValueError(f"Unknown state: {state}")
