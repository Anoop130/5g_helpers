# auto_tuner_agent.py (Simplified Architecture + Your Final Prompt)

import torch
import yaml
import re
import random
import argparse
from tqdm import tqdm
from itertools import product

# Imports for the simplified, no-fine-tuning approach
from transformers import (
    AutoTokenizer, GenerationConfig, BitsAndBytesConfig,
    AutoModelForCausalLM
)
from simulation_environment import mock_run_simulation_and_get_reward

# ===================================================================
# SECTION 1: AUTO-TUNER AGENT (Unchanged)
# ===================================================================
class HyperparameterAgent:
    def __init__(self, action_space: dict):
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.epsilon_decay = 0.95
        self.min_epsilon = 0.1

    def get_action(self) -> dict:
        actions = self._get_all_actions()
        if random.random() < self.epsilon:
            print("[Auto-Tuner] ACTION: Exploring with random hyperparameters.")
            return random.choice(actions)
        else:
            print("[Auto-Tuner] ACTION: Exploiting with best-known hyperparameters.")
            if not self.q_table: return random.choice(actions)
            best_action_tuple = max(self.q_table, key=self.q_table.get)
            return dict(best_action_tuple)

    def learn(self, action: dict, reward: float):
        action_tuple = tuple(sorted(action.items()))
        old_q = self.q_table.get(action_tuple, 0.0)
        new_q = old_q + self.learning_rate * (reward - old_q)
        self.q_table[action_tuple] = new_q
        print(f"[Auto-Tuner] LEARNING: Q-value for {dict(action_tuple)} updated to {new_q:.4f}")
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)
        print(f"[Auto-Tuner] STATUS: New exploration rate (epsilon) is {self.epsilon:.3f}")

    def _get_all_actions(self) -> list[dict]:
        keys, values = self.action_space.keys(), self.action_space.values()
        return [dict(zip(keys, instance)) for instance in product(*values)]

def _parse_llm_output(text: str) -> dict:
    try:
        yaml_marker = "### YAML Output:"
        if yaml_marker in text:
            yaml_string = text.split(yaml_marker, 1)[-1].strip()
            return yaml.safe_load(yaml_string)
        return None
    except yaml.YAMLError:
        return None

# ===================================================================
# SECTION 2: THE GENERATION RUNNER (No more training!)
# ===================================================================
def execute_generation_run(hparams: dict, model, tokenizer) -> float:
    print(f"\n--- [Worker] Starting run with Generation HPs: {hparams} ---")
    
    # Using the exact prompt you provided, structured as a template
    PROMPT_TEMPLATE = """You are a highly skilled RF engineer specializing in electronic countermeasures.
Your mission is to generate a complete YAML configuration file to effectively jam a target frequency.

### Instructions:
1.  Analyze the `Current Mission` input, which contains the target frequency and a set of fixed hardware parameters.
2.  Determine the optimal values for the following **variable parameters**:
    - `amplitude`
    - `amplitude_width`
    - `bandwidth`
    - `tx_gain`
3.  Accurately copy the **fixed parameters** provided in the `Current Mission` into your output. Do not change their values.
4.  Ensure the final output is a single, valid YAML block and nothing else.
5.  Pay close attention to data types: `center_frequency`, `bandwidth`, and `sampling_freq` must use scientific 'e' notation (e.g., `1.842e9`).

### Example Task:
Target Frequency: 0.915 GHz
Fixed Parameters:
  initial_phase: 0
  sampling_freq: 20e6
  num_samples: 10000
  output_iq_file: "output.fc32"
  output_csv_file: "output.csv"
  write_iq: false
  write_csv: true
  device_args: "type=b200"

### Example YAML Output for 0.915 GHz:
amplitude: 0.9
amplitude_width: 0.1
center_frequency: 9.15e8
bandwidth: 10e6
initial_phase: 0
sampling_freq: 20e6
num_samples: 10000
output_iq_file: "output.fc32"
output_csv_file: "output.csv"
write_iq: false
write_csv: true
device_args: "type=b200"
tx_gain: 55

---

### Current Mission:
Target Frequency: {freq:.4f} GHz
Fixed Parameters:
{fixed_params_str}

### YAML Output:
"""
    # Using the fixed parameters from your prompt's example
    fixed_parameters = {
      "initial_phase": 0,
      "sampling_freq": 40e6,
      "num_samples": 20000,
      "output_iq_file": "output.fc32",
      "output_csv_file": "output.csv",
      "write_iq": False,
      "write_csv": True,
      "device_args": "type=b200"
    }
    fixed_params_str = "\n".join([f"  {k}: {v}" for k, v in fixed_parameters.items()])

    total_score = 0
    # The actual target frequencies we want the model to solve for
    test_frequencies = [1.83, 1.842, 1.85] 
    
    generation_config = GenerationConfig(
        max_new_tokens=250,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True, # Must be true to use temperature/top_p
        **hparams # Directly apply the chosen generation hyperparameters
    )

    for freq in test_frequencies:
        prompt_text = PROMPT_TEMPLATE.format(freq=freq, fixed_params_str=fixed_params_str)
        inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
        
        output_tokens = model.generate(**inputs, generation_config=generation_config)


        # full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        # print("="*40)
        # print(f"DEBUGGING: Raw output from LLM for frequency {freq} GHz:")
        # print(full_text)
        # print("="*40)

        # Get the length of the input prompt in tokens
        input_token_length = inputs.input_ids.shape[1]
        # Get only the new tokens generated by the model
        generated_token_ids = output_tokens[0, input_token_length:]
        # Decode only the new tokens
        generated_text_only = tokenizer.decode(generated_token_ids, skip_special_tokens=True)

        print("="*40)
        print(f"DEBUGGING: Generated-only output for frequency {freq} GHz:")
        print(generated_text_only)
        print("="*40)

        full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        

        config = _parse_llm_output(full_text)
        score = mock_run_simulation_and_get_reward(config)
        total_score += score

    avg_score = total_score / len(test_frequencies)
    print(f"--- [Worker] Run Complete. Final average score: {avg_score:.4f} ---")
    return avg_score

# ===================================================================
# SECTION 3: THE MAIN CONTROLLER
# ===================================================================
def main():
    parser = argparse.ArgumentParser(description="Auto-Tuner for LLM Generation.")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model to use.")
    parser.add_argument("--loops", type=int, default=20, help="Number of hyperparameter sets to test.")
    args = parser.parse_args()

    # Hyperparameter space for text generation
    hyperparameter_space = {
        'temperature': [0.6, 0.8, 1.0],
        'top_p': [0.9, 0.95, 1.0],
        'repetition_penalty': [1.0, 1.2],
    }
    
    auto_tuner = HyperparameterAgent(hyperparameter_space)
    
    print("="*20 + " LOADING BASE MODEL (ONCE) " + "="*20)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("="*20 + " MODEL LOADED " + "="*20)

    print(f"Running for {args.loops} loops.")

    for i in range(args.loops):
        print(f"\n{'='*15} Auto-Tuner Episode {i+1}/{args.loops} {'='*15}")
        chosen_hps = auto_tuner.get_action()
        reward = execute_generation_run(
            hparams=chosen_hps, 
            model=model,
            tokenizer=tokenizer
        )
        auto_tuner.learn(action=chosen_hps, reward=reward)
    
    print("\n" + "="*20 + " AUTO-TUNING COMPLETE " + "="*20)
    if not auto_tuner.q_table:
        print("No trials were completed.")
        return
        
    sorted_q_table = sorted(auto_tuner.q_table.items(), key=lambda item: item[1], reverse=True)
    print("\nFinal discovered knowledge (Q-Table), from best to worst:")
    for (hps_tuple, score) in sorted_q_table:
        print(f"  Score: {score:.4f} | Hyperparameters: {dict(hps_tuple)}")
    
    best_hps = dict(sorted_q_table[0][0])
    print(f"\nRECOMMENDED GENERATION HYPERPARAMETERS: {best_hps}")

if __name__ == "__main__":
    main()