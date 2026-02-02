"""
Battery electrochemical core module.

Handles SOC tracking, polarization dynamics, and current solving (Algebraic loop).
"""

import numpy as np
from .config import SimulationParams, get_ocv, get_r0, get_rp, get_cp

class PowerCollapseError(Exception):
    """Raised when the requested power exceeds battery capabilities."""
    pass

class BatteryModel:
    """
    Electrochemical battery model with Thevenin equivalent circuit.
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        
    def get_ocv(self, z: float, Tc: float) -> float:
        """Wrapper for OCV lookup."""
        return get_ocv(z, Tc, self.params)
        
    def get_r0(self, z: float, Tc: float) -> float:
        """Wrapper for R0 lookup."""
        return get_r0(z, Tc, self.params)

    def solve_current(self, 
                      P_sys: float, 
                      z: float, 
                      Vp: float, 
                      Tc: float) -> tuple[float, float, float]:
        """
        Solve for current I given system power demand.
        [TEXT Eq.5.11]
        """
        U_ocv = self.get_ocv(z, Tc)
        R0 = self.get_r0(z, Tc)
        
        # Effective voltage available to drive load before R0 drop
        V_eff = U_ocv - Vp
        
        discriminant = V_eff**2 - 4.0 * R0 * P_sys
        '''[TEXT Eq.5.11]'''
        
        if discriminant < 0:
            raise PowerCollapseError(f"Power collapse: P_sys={P_sys:.2f}W exceeds limit.")
            
        sqrt_delta = np.sqrt(discriminant)
        I = (V_eff - sqrt_delta) / (2.0 * R0)
        
        # Terminal Voltage
        V_term = U_ocv - Vp - I * R0
        
        return I, V_term, discriminant

    def get_soc_derivative(self, I: float) -> float:
        """
        Calculate dz/dt (Coulomb counting).
        [TEXT Eq.5.5]
        """
        # SOH can be functional
        soh = self.params.SOH if not callable(self.params.SOH) else self.params.SOH()
        
        capacity_Ah = self.params.Q_design * soh
        capacity_C = capacity_Ah * 3600.0
        
        return -I / capacity_C

    def get_vp_derivative(self, Vp: float, I: float, Tc: float) -> float:
        """
        Calculate dVp/dt.
        dVp/dt = -1/tau_p * Vp + I/Cp
        tau_p = Rp * Cp
        [TEXT Eq.5.6] Equivalent
        """
        # Get functional parameters
        Rp = get_rp(Tc, self.params)
        Cp = get_cp(Tc, self.params)
        tau_p = Rp * Cp
        
        term1 = -Vp / tau_p
        term2 = I / Cp # equivalent to (Rp * I) / tau_p
        
        return term1 + term2
