"""
Script to statistically verify UserBehaviorModel.
Runs a long simulation and prints distribution stats.
"""
import numpy as np
import sys
import os

# Ensure project root in path
sys.path.append(os.getcwd())

from mp_shm_x2.stochastic import UserBehaviorModel

def analyze_behavior():
    num_rounds = 10
    duration = 24 * 3600.0
    dt = 1.0
    
    print(f"Running {num_rounds} rounds of {duration/3600}h simulation...")
    print("-" * 80)
    
    # Aggregated Stats
    agg_counts = {}
    agg_times = {}
    
    for r in range(num_rounds):
        model = UserBehaviorModel()
        current_state = model.current_state
        segment_start = 0.0
        
        # Per round stats
        # print(f"Round {r+1}...", end="\r")
        
        steps = int(duration / dt)
        for i in range(steps):
            t = i * dt
            s = model.step(dt)
            
            if s != current_state:
                seg_len = t - segment_start
                
                agg_counts[current_state] = agg_counts.get(current_state, 0) + 1
                agg_times[current_state] = agg_times.get(current_state, 0.0) + seg_len
                
                current_state = s
                segment_start = t
                
        # Last segment
        seg_len = duration - segment_start
        agg_counts[current_state] = agg_counts.get(current_state, 0) + 1
        agg_times[current_state] = agg_times.get(current_state, 0.0) + seg_len

    print("\n--- Aggregated User Behavior Analysis (10 Rounds Average) ---")
    print(f"{'State':<10} | {'Avg Count':<10} | {'Avg Total (h)':<14} | {'% Time':<8} | {'Avg Dur (m)':<11}")
    print("-" * 80)
    
    total_sim_time = duration * num_rounds
    
    for state in sorted(agg_counts.keys()):
        total_count = agg_counts[state]
        total_time = agg_times[state]
        
        avg_count_per_day = total_count / num_rounds
        avg_time_per_day = (total_time / num_rounds) / 3600.0
        percent = (total_time / total_sim_time) * 100.0
        
        # Average duration of a single segment
        if total_count > 0:
            avg_dur_min = (total_time / total_count) / 60.0
        else:
            avg_dur_min = 0.0
            
        print(f"{state:<10} | {avg_count_per_day:<10.1f} | {avg_time_per_day:<14.2f} | {percent:<8.1f} | {avg_dur_min:<11.1f}")

if __name__ == "__main__":
    analyze_behavior()
