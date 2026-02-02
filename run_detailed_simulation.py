"""
Script to run a detailed battery simulation and generate a comprehensive dashboard.
Includes: SOC/Voltage, Component Power Breakdown, and User State Gantt Chart.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import sys
import os

# Ensure project root is in path
sys.path.append(os.getcwd())

from mp_shm_x2.simulation import SimulationEngine
from mp_shm_x2.config import SimulationParams

def plot_detailed_results(engine: SimulationEngine) -> None:
    results = engine.results
    
    # Data Preparation
    time_hours = np.array(results.time) / 3600.0
    soc_percent = np.array(results.z) * 100.0
    
    # Create 4-panel plot
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 16), sharex=True, gridspec_kw={'height_ratios': [1, 2, 2, 2]})
    
    # --- Panel 1: User State Gantt Chart ---
    states = results.State
    times = results.time
    
    colors = {
        "Idle": "lightgray", "Video": "red", "Game": "purple", 
        "Call": "green", "Camera": "orange", "Gallery": "blue"
    }
    
    # Reconstruct segments from state log
    segments = []
    if len(states) > 0:
        seg_start = times[0]
        curr_s = states[0]
        for i in range(1, len(states)):
            if states[i] != curr_s:
                segments.append((seg_start, times[i], curr_s))
                seg_start = times[i]
                curr_s = states[i]
        segments.append((seg_start, times[-1], curr_s))
    
    # Draw bars
    y_pos = 0
    height = 1
    for start, end, state in segments:
        width = end - start
        if width <= 0: continue
        color = colors.get(state, "black")
        rect = mpatches.Rectangle((start/3600.0, y_pos), width/3600.0, height, color=color, alpha=0.8)
        ax1.add_patch(rect)
        
        # Label if wide enough (>15 mins)
        if width > 900: 
            center_x = (start + width/2) / 3600.0
            ax1.text(center_x, y_pos + height/2, state, ha='center', va='center', color='black', fontsize=8, rotation=90 if width < 1800 else 0)

    ax1.set_ylim(0, 1)
    ax1.set_yticks([])
    ax1.set_title('User Activity State')
    # Legend
    patches = [mpatches.Patch(color=c, label=l) for l, c in colors.items() if l in set(states)]
    ax1.legend(handles=patches, loc='upper right', ncol=len(patches))

    # --- Panel 2: Power Breakdown ---
    # Use stackplot for components
    ax2.stackplot(time_hours, 
                  results.P_cpu, results.P_screen, results.P_net, 
                  labels=['CPU', 'Screen', 'Network'], 
                  colors=['#ff9999', '#66b3ff', '#99ff99'], alpha=0.8)
    
    # Overlay total system power
    ax2.plot(time_hours, results.P_sys, color='black', linewidth=1, linestyle='--', label='Total P_sys')
    
    ax2.set_ylabel('Power (W)')
    ax2.set_title('System Power Consumption Breakdown')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    # --- Panel 3: SOC and Voltage ---
    ax3.plot(time_hours, soc_percent, label='SOC (%)', color='blue', linewidth=2)
    ax3.set_ylabel('SOC (%)', color='blue')
    ax3.tick_params(axis='y', labelcolor='blue')
    ax3.set_ylim(0, 105)
    ax3.grid(True)
    
    # Twin axis for Voltage
    ax3_r = ax3.twinx()
    ax3_r.plot(time_hours, results.V_term, label='Terminal Voltage (V)', color='green', linewidth=1.5)
    ax3_r.set_ylabel('Voltage (V)', color='green')
    ax3_r.tick_params(axis='y', labelcolor='green')
    
    ax3.set_xlabel('Time (hours)')
    ax3.set_title('Battery Status (SOC & Voltage)')

    # --- Panel 4: Thermal & Polarization ---
    # Convert Kelvin to Celsius for better readability
    Tc_celsius = np.array(results.Tc) - 273.15
    Ts_celsius = np.array(results.Ts) - 273.15
    
    ax4.plot(time_hours, Tc_celsius, label='Core Temp ($T_c$)', color='red', linewidth=1.5)
    ax4.plot(time_hours, Ts_celsius, label='Surface Temp ($T_s$)', color='orange', linewidth=1.5, linestyle='--')
    
    ax4.set_ylabel('Temperature (°C)', color='red')
    ax4.tick_params(axis='y', labelcolor='red')
    ax4.set_ylim(min(Tc_celsius.min(), Ts_celsius.min()) - 2, max(Tc_celsius.max(), Ts_celsius.max()) + 2)
    ax4.grid(True)
    
    # Twin axis for Polarization Voltage
    ax4_r = ax4.twinx()
    # Vp is usually small (mV to V range)
    ax4_r.plot(time_hours, results.Vp, label='Polarization Voltage ($V_p$)', color='purple', linewidth=1.0)
    ax4_r.set_ylabel('Polarization Voltage ($V_p$) [V]', color='purple')
    ax4_r.tick_params(axis='y', labelcolor='purple')
    
    ax4.set_xlabel('Time (hours)')
    ax4.set_title('Thermal & Electrochemical Dynamics')
    
    # Legend
    lines, labels = ax4.get_legend_handles_labels()
    lines2, labels2 = ax4_r.get_legend_handles_labels()
    ax4.legend(lines + lines2, labels + labels2, loc='upper right')

    plt.tight_layout()
    plt.savefig('detailed_simulation_report.png')
    print("Report saved to 'detailed_simulation_report.png'")

def main():
    # Setup
    params = SimulationParams()
    params.dt = 1.0
    
    # Initialize Engine
    engine = SimulationEngine(params)
    
    print("Starting detailed simulation (Target: 100% -> 0% SOC)...")
    
    # Run
    max_duration = 48 * 3600 
    engine.run(max_duration)
    
    if engine.results.Failure:
        print(f"Simulation ended: {engine.results.Failure} at t={engine.t/3600:.2f}h")
    else:
        print("Simulation reached max duration.")
        
    # Plot
    plot_detailed_results(engine)

if __name__ == "__main__":
    main()
