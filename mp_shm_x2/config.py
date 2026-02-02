"""
MP-SHM-X2 Configuration Module.

This module contains all the physical constants, model parameters, and lookup tables
required for the simulation.
Supports both Constant (C) and Functional (F) parameters.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Union, Any
import numpy as np

# Type alias for functional parameters
# Can be a scalar (float) or a callable
ParamFunc = Union[float, Callable[..., float]]

@dataclass
class SimulationParams:
    """
    Holds all simulation parameters.
    Parameters marked as [F] can be functions or scalars.
    """
    
    # --- Time & Simulation ---
    dt: float = 1.0
    """[C] Time step in seconds."""
    
    # --- Battery Design ---
    Q_design: float = 4.0
    """[C] Battery design capacity in Ah."""
    
    V_cut: float = 3.0
    """[C] Cut-off voltage in V."""
    
    SOH: ParamFunc = 1.0
    """[F] State of Health (0.0 to 1.0). Default constant 1.0.\n UNDEFINED FUNCTION"""

    # --- Electrochemical Parameters ---
    
    R0_ref: float = 0.08
    """[C] Reference internal resistance in Ohms."""
    
    E_a: float = 35000.0 # Updated from Parameters.md (20000-50000)
    """[C] Activation energy for Arrhenius equation in J/mol."""
    
    R_gas: float = 8.3145
    """[C] Ideal gas constant J/(mol K)."""
    
    T_ref: float = 298.15
    """[C] Reference temperature in Kelvin (25 degC)."""
    
    # Rp and Cp are often T dependent
    Rp_ref: float = 0.05
    """[C] Reference Polarization resistance."""
    
    Cp_ref: float = 2000.0
    """[C] Reference Polarization capacitance."""

    # Functional Interfaces for Electrochemical Parameters
    # Default implementations are provided in methods below or as lambdas
    
    # --- Thermal Parameters ---
    C_c: float = 80.0
    """[C] Core heat capacity in J/K."""
    
    C_s: float = 15.0
    """[C] Surface heat capacity in J/K."""
    
    R_th_in: float = 2.0
    """[C] Internal thermal resistance (Core-Surface) in K/W."""
    
    h_eff: float = 10.0
    """[C] Effective convection coefficient in W/(m^2 K)."""
    
    A_surf: float = 0.008
    """[C] Battery surface area in m^2."""
    
    eta_therm: float = 0.2
    """[C] Thermal coupling coefficient (System power -> Battery surface)."""
    
    T_amb: ParamFunc = 298.15
    """[F] Ambient temperature in Kelvin. Can be time dependent."""

    # --- Power/Excitation Parameters ---
    alpha_cpu: float = 0.2
    """[C] CPU power coefficient W/(GHz^3)."""
    
    P_driver: float = 0.1
    """[C] Screen driver base power in W."""
    
    C_oled: float = 0.002 
    """[C] OLED efficiency coefficient W/nits."""
    
    gamma_oled: float = 2.2
    """[C] OLED gamma."""
    
    P_net_idle: float = 0.05
    """[C] Network idle power in W."""
    
    P_net_max: float = 1.5
    """[C] Network max power in W."""
    
    tau_tail: float = 5.0
    """[C] Network tail state decay time constant."""
    
    P_base: float = 0.5
    """[C] Base system power in W."""

    # --- Stochastic Parameters ---
    tau_ou_f: float = 10.0
    """[C] CPU frequency OU process time constant."""
    
    sigma_ou_f: float = 0.1
    """[C] CPU frequency OU process volatility (GHz)."""
    
    tau_ou_L: float = 20.0
    """[C] Screen brightness OU process time constant."""
    
    sigma_ou_L: float = 50.0
    """[C] Screen brightness OU process volatility."""

    # --- User Behavior Parameters (State Dependent) ---
    # These should be lookups based on state string
    
    # Dictionaries to simulate lookup tables
    user_params: Dict[str, Dict[str, float]] = field(default_factory=lambda: {
        "Idle": {
            "mu_dwell": np.log(7200.0), "sigma_dwell": 0.5,
            "f_base": 0.2, "L_base": 0.0, "APL_base": 0.0,
            "lambda_net": 0.01
        },
        "Video": {
            "mu_dwell": np.log(2700.0), "sigma_dwell": 0.4,
            "f_base": 1.0, "L_base": 400.0, "APL_base": 0.6,
            "lambda_net": 0.1 # Packets per second
        },
        "Game": {
            "mu_dwell": np.log(1800.0), "sigma_dwell": 0.3,
            "f_base": 2.5, "L_base": 800.0, "APL_base": 0.8,
            "lambda_net": 0.5
        },
        "Call": {
            "mu_dwell": np.log(300.0), "sigma_dwell": 0.2,
            "f_base": 0.5, "L_base": 0.0, "APL_base": 0.0,
            "lambda_net": 0.05
        }
    })


# --- Parameter Functions (Implementations) ---

def get_ocv(z: float, Tc: float, params: SimulationParams = None) -> float:
    """
    [F] Open Circuit Voltage.
    Ideally a lookup table U_ocv(z, Tc).
    Currently a simplified linear model with Shepherd-like term and Temperature dependence.
    AI built this func
    """
    # Simplified linear model with Shepherd-like term and Temperature dependence
    # U = 3.0 + z + 0.01 * (Tc - T_ref) ? 
    
    # Temperature correction coefficient (approx 0.5 mV/K)
    k_T = 0.0005 
    T_ref = 298.15
    
    # Shepherd-like OCV curve
    # E0 - K*z/(z+0.1) + A*exp(-B*(1-z)) ... simplified to:
    # 3.2 + 0.9*z - 0.1/(z+0.01) (Just an example shape)
    
    # Use the simple linear one from before but add T dependence
    # U_ref = 3.0 + 1.0 * z 
    U_ref = 3.0 + 0.9 * z # Slightly flatter
    
    U_T = k_T * (Tc - T_ref)
    
    return U_ref + U_T

def get_r0(z: float, Tc: float, params: SimulationParams) -> float:
    """
    [F] Internal Resistance R0(z, Tc).
    Arrhenius + SOC dependence.
    """
    # 1. Arrhenius Term
    # R0(T) = R0_ref * exp( Ea/R * (1/T - 1/T_ref) )
    arrhenius_arg = (params.E_a / params.R_gas) * (1.0/Tc - 1.0/params.T_ref)
    arrhenius_factor = np.exp(arrhenius_arg)
    
    # 2. SOC Factor f_R(z)
    # U-shape: High at low SOC, flat in middle, slightly high at high SOC
    # Simple U-shape: 1 + 0.5*(z-0.5)^2
    soc_factor = 1.0 + 1.5 * ((z - 0.5)**2)
    
    return params.R0_ref * arrhenius_factor * soc_factor

def get_rp(Tc: float, params: SimulationParams) -> float:
    """
    [F] Polarization Resistance Rp(Tc).
    Arrhenius dependence.
    """
    # Assuming similar activation energy for transport
    arrhenius_arg = (params.E_a / params.R_gas) * (1.0/Tc - 1.0/params.T_ref)
    return params.Rp_ref * np.exp(arrhenius_arg)

def get_cp(Tc: float, params: SimulationParams) -> float:
    """
    [F] Polarization Capacitance Cp(Tc).
    Often increases with temperature.
    """
    # Simple linear dependence for now: Cp = Cp_ref * (1 + 0.01*(T-Tref))
    # Or just constant if not specified
    return params.Cp_ref

def get_dudt(z: float, Tc: float, params: SimulationParams = None) -> float:
    """
    [F] Entropic heat coefficient.
    """
    # Typically varies with SOC. 
    # Positive at low SOC, negative at high SOC for some chemistries.
    # Let's model a transition.
    # 0.2 mV/K at z=0, -0.2 mV/K at z=1
    return 0.0002 * (1.0 - 2.0 * z)

