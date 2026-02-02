"""
Stochastic processes module.

Handles Ornstein-Uhlenbeck (OU) noise processes and discrete state transitions (Semi-Markov Chain).
"""

import numpy as np
from typing import Dict, Optional, List, Any

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
    
    def __init__(self, params: Optional[Any] = None, initial_state: str = "Idle", t_clock_init: float = 0.0):
        self.params = params
        self.current_state = initial_state
        self.time_in_state = 0.0
        
        # Time of day in seconds (0 to 86400). Rolls over.
        self.t_clock = t_clock_init 
        
        # History of states [S_{k-N}, ..., S_{k-1}]
        self.history: List[str] = []
        self.max_history = 5
        
        # --- Configuration Section (Could be moved to config.py) ---
        self.states = ["Idle", "Video", "Game", "Call", "Camera"]
        
        # Transition Matrix P(Next | Current)
        # Rows must sum to 1.0
        self.transitions = {
            "Idle":   {"Video": 0.3, "Game": 0.2, "Call": 0.1, "Camera": 0.1, "Idle": 0.3},
            "Video":  {"Idle": 0.8, "Game": 0.1, "Call": 0.1},
            "Game":   {"Idle": 0.9, "Video": 0.1},
            "Call":   {"Idle": 1.0},
            "Camera": {"Idle": 0.8, "Video": 0.2}
        }
        
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

    def _get_circadian_factor(self) -> float:
        """
        Calculate a factor based on time of day.
        Returns 1.0 at peak activity (e.g., 20:00), lower at night.
        """
        # Simple sinusoidal model
        # Peak at 14:00 (50400s) and 20:00? Let's assume single peak at 18:00 (64800s)
        # T = 86400
        # factor = 0.5 + 0.5 * cos(2pi * (t - t_peak) / T)
        t_peak = 18.0 * 3600.0
        T = 86400.0
        factor = 0.5 + 0.5 * np.cos(2 * np.pi * (self.t_clock - t_peak) / T)
        return factor

    def _sample_dwelling_time(self, state: str) -> float:
        """
        Sample dwelling time based on State, History, and T_clock.
        """
        prev_state = self.history[-1] if self.history else None
        circadian_factor = self._get_circadian_factor() # 0.0 (Night) to 1.0 (Day)
        
        # Base Parameters (can be extracted to config)
        mu = 0.0
        sigma = 0.5
        
        # --- Logic 1: State & History Dependence ---
        if state == "Idle":
            # Requirement: Bimodal Distribution for Idle
            # Mode 1: Short Idle (checking notifications) -> Mean ~30s
            # Mode 2: Long Idle (pocket/sleep) -> Mean ~1h (3600s)
            
            # If previous state was Camera, high chance of Short Idle (checking photo)
            is_short_idle = False
            if prev_state == "Camera":
                is_short_idle = True # Always short after camera (simplified)
            else:
                # Probabilistic mix based on Circadian Rhythm
                # Active day -> more short idles. Night -> more long idles.
                prob_long = 0.8 * (1.0 - circadian_factor) + 0.2
                is_short_idle = np.random.random() > prob_long
            
            if is_short_idle:
                median = 60.0
                sigma = 0.5
                val = np.random.lognormal(np.log(median), sigma)
            else:
                # Long idle: Night is longer than Day
                # Tuned: Reduce day-time long idle significantly
                # Day (factor~1.0): median = base_long * 1.0
                # Night (factor~0.0): median = base_long * 4.0
                
                base_long = 900.0 # 15 mins base for active day
                
                # Penalty for long idle during day:
                # If circadian_factor > 0.5 (Day), use smaller sigma to avoid heavy tail
                # If Night, allow longer tail
                
                time_multiplier = 1.0 + 5.0 * (1.0 - circadian_factor) # 15m -> 1.5h at night
                median = base_long * time_multiplier
                
                if circadian_factor > 0.6:
                    sigma = 0.6 # Tighter distribution during day
                else:
                    sigma = 0.8 # Variable at night
                
                # Penalty for excessive long idle during active hours
                # If it's day time (factor > 0.5), cap/penalize long durations
                val = np.random.lognormal(np.log(median), sigma)
                
                if circadian_factor > 0.6: # Active day time
                    # Soft cap: if val > 1 hour (3600), resample or truncate?
                    # Let's truncate to max 45 mins (2700s) + some noise
                    if val > 2700.0:
                         val = 2700.0 + np.random.exponential(300.0)

            return max(1.0, val) # Ensure at least 1s
                
        elif state == "Video":
            # Video duration depends on time of day (longer at night/evening)
            base_median = 600.0 # 10 mins
            median = base_median * (1.0 + 2.0 * circadian_factor) # Up to 30 mins
            sigma = 0.6
            
        elif state == "Game":
            median = 1200.0
            sigma = 0.4
            
        elif state == "Call":
            median = 180.0
            sigma = 0.7
            
        elif state == "Camera":
            median = 45.0
            sigma = 0.3
            
        else:
            median = 60.0
            sigma = 0.5
            
        mu = np.log(median)
        val = np.random.lognormal(mu, sigma)
        return max(1.0, val) # Ensure at least 1s

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
