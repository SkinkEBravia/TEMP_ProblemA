"""
Thermal dynamics module.

Implements the two-node thermal model (Core and Surface).
"""

from .config import SimulationParams

class ThermalModel:
    """
    Two-node thermal model for battery temperature.
    
    Nodes:
        - Tc: Core temperature (internal)
        - Ts: Surface temperature (skin)
        
    Equations:
        Cc * dTc/dt = I^2*R0 + I*Tc*dU/dT - (Tc - Ts)/R_th_in
        Cs * dTs/dt = (Tc - Ts)/R_th_in - h_eff*A*(Ts - T_amb) + eta * P_sys
    """
    
    def __init__(self, params: SimulationParams):
        self.params = params
        # Initial temperatures assumed to be ambient if not specified
        # In a full sim, these would be part of the state vector passed in.
        
    def get_core_temp_derivative(self, 
                                 Tc: float, 
                                 Ts: float, 
                                 I: float, 
                                 R0: float, 
                                 dudt: float) -> float:
        """
        Calculate dTc/dt.
        
        Args:
            Tc: [arg] Core temperature (K).
            Ts: [arg] Surface temperature (K).
            I: [arg] Current (A). Positive for discharge.
            R0: [arg] Internal resistance (Ohm).
            dudt: [arg] Entropic coefficient (V/K).
            
        Returns:
            dTc/dt in K/s.
        """
        # Heat generation terms
        q_joule = (I ** 2) * R0
        
        # Reversible heat (Peltier)
        # Note: Sign convention for I. 
        # If I > 0 is discharge, and entropy dU/dT is typically negative for discharge heating?
        # Standard form: Q_rev = - I * T * dU/dT
        # Let's follow the markdown Eq 5.8: + I * Tc * dU/dT
        # We need to be careful with signs.
        # If dU/dT is negative (common for LCO/NMC at some SOCs), then discharge (I>0) * Tc * (-val) = cooling?
        # Actually, usually Q_rev = I * T * (dE/dT). 
        # Let's stick strictly to the markdown Eq 5.8:
        # Cc * dTc/dt = I^2*R0 + I * Tc * dU/dT - ...
        q_rev = I * Tc * dudt
        
        # Heat transfer
        q_cond = (Tc - Ts) / self.params.R_th_in
        
        dTc_dt = (q_joule + q_rev - q_cond) / self.params.C_c
        return dTc_dt

    def get_surface_temp_derivative(self, 
                                    Tc: float, 
                                    Ts: float, 
                                    P_sys: float) -> float:
        """
        Calculate dTs/dt.
        """
        # Get Ambient T (might be functional)
        T_amb = self.params.T_amb if not callable(self.params.T_amb) else self.params.T_amb()
        
        # Heat transfer from core
        q_cond = (Tc - Ts) / self.params.R_th_in
        
        # Convection to ambient
        q_conv = self.params.h_eff * self.params.A_surf * (Ts - T_amb)
        
        # External heating from system electronics
        q_ext = self.params.eta_therm * P_sys
        
        dTs_dt = (q_cond - q_conv + q_ext) / self.params.C_s
        return dTs_dt
