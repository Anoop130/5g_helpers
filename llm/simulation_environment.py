# simulation_environment.py (Updated to be more robust)
import math
import time

# The optimal target values from the new reference YAML
TARGET_FREQ = 1.842e9
TARGET_BW = 80e6
TARGET_GAIN = 70.0
TARGET_AMPLITUDE = 0.7
TARGET_AMPLITUDE_WIDTH = 0.05

def mock_run_simulation_and_get_reward(config: dict) -> float:
    print(f"--- Running Simulation ---")
    print(f"Config: {config}")

    # === NEW: Robustness check to prevent crashes ===
    if not config:
        print("Invalid config: The provided configuration was None or empty. Reward is -1.0.")
        return -1.0
        
    try:
        # Parse all five key parameters from the LLM's output
        center_freq = float(config.get("center_frequency", 0))
        bandwidth = float(config.get("bandwidth", 0))
        tx_gain = float(config.get("tx_gain", 0))
        amplitude = float(config.get("amplitude", 0))
        amplitude_width = float(config.get("amplitude_width", 0))
    except (TypeError, ValueError, AttributeError):
        print("Invalid config format or missing keys. Reward is -0.1.")
        return -0.1

    # --- Score calculation for each parameter (from 0.0 to 1.0) ---

    # Score for Frequency (how close to 1.842 GHz)
    freq_diff = abs(center_freq - TARGET_FREQ)
    max_reasonable_freq_diff = 200e6 # 200 MHz
    freq_score = max(0.0, 1.0 - (freq_diff / max_reasonable_freq_diff))

    # Score for Bandwidth (how close to 80 MHz)
    bw_diff = abs(bandwidth - TARGET_BW)
    bw_score = max(0.0, 1.0 - (bw_diff / (TARGET_BW * 2)))

    # Score for Gain (how close to 70)
    gain_diff = abs(tx_gain - TARGET_GAIN)
    gain_score = max(0.0, 1.0 - (gain_diff / 50.0))

    # NEW: Score for Amplitude (how close to 0.7)
    amp_diff = abs(amplitude - TARGET_AMPLITUDE)
    amplitude_score = max(0.0, 1.0 - (amp_diff / 1.0)) # Amplitude is between 0-1, so max diff is 1.0

    # NEW: Score for Amplitude Width (how close to 0.05)
    amp_width_diff = abs(amplitude_width - TARGET_AMPLITUDE_WIDTH)
    amp_width_score = max(0.0, 1.0 - (amp_width_diff / 0.2)) # A reasonable max difference

    # NEW: Final reward is a weighted average of all five scores.
    # Weights: Freq (40%), BW (20%), Gain (15%), Amp (15%), Amp_Width (10%)
    final_reward = (freq_score * 0.4) + (bw_score * 0.2) + (gain_score * 0.15) + (amplitude_score * 0.15) + (amp_width_score * 0.10)
    
    pdr = 1.0 - final_reward 
    
    if final_reward > 0.98: # Stricter bonus threshold
        final_reward = 1.0

    print(f"Scores -> Freq: {freq_score:.2f}, BW: {bw_score:.2f}, Gain: {gain_score:.2f}, Amp: {amplitude_score:.2f}, AmpW: {amp_width_score:.2f}")
    print(f"Result -> PDR: {pdr:.3f}, Reward: {final_reward:.3f}")
    print(f"--------------------------\n")
    
    return final_reward