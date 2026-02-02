
import numpy as np
from typing import List, Dict, Any, Optional

# --- Configuration for User Behavior Model ---

# 1. State Space Definition
STATES = ["Idle", "Video", "Game", "Call", "Camera"]

# 2. Transition Matrix Definition
# P(Next | Current)
# Note: Self-transitions are removed as dwelling time is modeled explicitly.
TRANSITIONS = {
    "Idle":   {"Video": 0.45, "Game": 0.3, "Call": 0.15, "Camera": 0.1},
    "Video":  {"Idle": 0.8, "Game": 0.1, "Call": 0.1},
    "Game":   {"Idle": 0.9, "Video": 0.1},
    "Call":   {"Idle": 1.0},
    "Camera": {"Idle": 0.8, "Video": 0.2}
}

class DwellingTimePolicy:
    """
    Encapsulates the logic for sampling dwelling times for different user states.
    Uses a strategy pattern to handle state-specific logic.
    """
    
    def __init__(self):
        # Default fallback configuration for states without specific overrides
        self.defaults = {
            "Game": {"median": 1200.0, "sigma": 0.4},
            "Call": {"median": 180.0, "sigma": 0.7},
            "Camera": {"median": 45.0, "sigma": 0.3},
            "Video": {"median": 600.0, "sigma": 0.6},
            "Idle": {"median": 900.0, "sigma": 0.5}, # Base default
            "default": {"median": 60.0, "sigma": 0.5}
        }

    def get_circadian_factor(self, t_clock: float) -> float:
        """
        Calculate a factor based on time of day.
        Returns 1.0 at peak activity (e.g., 18:00), lower at night (near 0.0).
        """
        t_peak = 18.0 * 3600.0
        T = 86400.0
        # cos goes from -1 to 1. 0.5 + 0.5*cos goes from 0 to 1.
        factor = 0.5 + 0.5 * np.cos(2 * np.pi * (t_clock - t_peak) / T)
        return factor

    def _get_user_param(self, params: Any, state: str, key: str) -> Optional[float]:
        """Safely retrieve a parameter from SimulationParams."""
        if params and hasattr(params, 'user_params') and state in params.user_params:
            return params.user_params[state].get(key)
        return None

    def _sample_lognormal(self, median: float, sigma: float) -> float:
        """Helper to sample from LogNormal given median and sigma."""
        return max(1.0, np.random.lognormal(np.log(median), sigma))

    def _strategy_idle(self, state: str, history: List[str], t_clock: float, params: Any, circadian_factor: float) -> float:
        """
        Bimodal strategy for Idle state.
        Mode 1: Short Idle (checking notifications).
        Mode 2: Long Idle (pocket/sleep).
        """
        prev_state = history[-1] if history else None
        
        # 1. Determine if Short or Long Idle
        is_short_idle = False
        if prev_state == "Camera":
            is_short_idle = True # Always short after camera
        else:
            # Active day -> more short idles. Night -> more long idles.
            prob_long = 0.8 * (1.0 - circadian_factor) + 0.2
            is_short_idle = np.random.random() > prob_long
        
        if is_short_idle:
            return self._sample_lognormal(median=60.0, sigma=0.5)
        
        # 2. Long Idle Logic
        # Check for config override
        mu_dwell = self._get_user_param(params, state, "mu_dwell")
        
        if mu_dwell is not None:
            # Treat config value as the 'Night' (maximum) median
            night_median = np.exp(mu_dwell)
            base_long = night_median / 6.0
        else:
            base_long = self.defaults["Idle"]["median"]

        # Penalty for long idle during day:
        # Time multiplier scales from 1x (Day) to 6x (Night)
        time_multiplier = 1.0 + 5.0 * (1.0 - circadian_factor) 
        median = base_long * time_multiplier
        
        # Sigma adjustment
        if circadian_factor > 0.6:
            sigma = 0.6 # Tighter distribution during day
        else:
            sigma = 0.8 # Variable at night
            
        val = self._sample_lognormal(median, sigma)
        
        # Soft cap for day time long idles
        if circadian_factor > 0.6 and val > 2700.0:
             val = 2700.0 + np.random.exponential(300.0)
             
        return val

    def _strategy_video(self, state: str, history: List[str], t_clock: float, params: Any, circadian_factor: float) -> float:
        """
        Video strategy: Duration scales with circadian factor (longer at night/active times).
        """
        mu_dwell = self._get_user_param(params, state, "mu_dwell")
        
        if mu_dwell is not None:
            base_median = np.exp(mu_dwell)
        else:
            base_median = self.defaults["Video"]["median"]
            
        # Scale up to 3x based on activity? 
        # Original logic: median = base * (1 + 2 * factor)
        # Note: factor is 1.0 at Peak (Evening), 0.0 at Night/Morning.
        # This implies Video is longer during Peak hours.
        median = base_median * (1.0 + 2.0 * circadian_factor)
        
        sigma = self._get_user_param(params, state, "sigma_dwell") or self.defaults["Video"]["sigma"]
        return self._sample_lognormal(median, sigma)

    def _strategy_standard(self, state: str, history: List[str], t_clock: float, params: Any, circadian_factor: float) -> float:
        """
        Standard strategy: Simple LogNormal based on config or defaults.
        Used for Game, Call, Camera.
        """
        mu_dwell = self._get_user_param(params, state, "mu_dwell")
        
        if mu_dwell is not None:
            median = np.exp(mu_dwell)
        else:
            # Fallback to defaults if available, else generic default
            default = self.defaults.get(state, self.defaults["default"])
            median = default["median"]
            
        sigma = self._get_user_param(params, state, "sigma_dwell")
        if sigma is None:
            default = self.defaults.get(state, self.defaults["default"])
            sigma = default["sigma"]
            
        return self._sample_lognormal(median, sigma)

    def sample(self, state: str, history: List[str], t_clock: float, params: Any = None) -> float:
        """
        Main entry point to sample dwelling time.
        """
        circadian_factor = self.get_circadian_factor(t_clock)
        
        # Dispatch based on state
        if state == "Idle":
            return self._strategy_idle(state, history, t_clock, params, circadian_factor)
        elif state == "Video":
            return self._strategy_video(state, history, t_clock, params, circadian_factor)
        else:
            # Use standard strategy for others (Game, Call, Camera, Unknown)
            return self._strategy_standard(state, history, t_clock, params, circadian_factor)

# Singleton instance for easy access
_policy = DwellingTimePolicy()

# Public Interface
def sample_dwelling_time(state: str, history: List[str], t_clock: float, params: Any = None) -> float:
    return _policy.sample(state, history, t_clock, params)
