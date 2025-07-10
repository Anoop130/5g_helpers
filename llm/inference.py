# inference.py
# A simple script to generate a config using the best-known hyperparameters.

import torch
import yaml
import re
import argparse
from transformers import (
    AutoTokenizer, GenerationConfig, BitsAndBytesConfig,
    AutoModelForCausalLM
)
from simulation_environment import mock_run_simulation_and_get_reward

def _parse_llm_output(text: str) -> dict:
    """
    Parses the full text from the LLM to find and decode the YAML block.
    """
    try:
        if "### YAML Output:" in text:
            potential_yaml_section = text.split("### YAML Output:")[-1].strip()
            documents = list(yaml.safe_load_all(potential_yaml_section))
            if documents and isinstance(documents[0], dict):
                return documents[0]
    except (yaml.YAMLError, IndexError):
        pass
    return None

def main():
    parser = argparse.ArgumentParser(description="Generate a jammer configuration using a fine-tuned LLM.")
    parser.add_argument("--freq", type=float, required=True, help="The target frequency to jam in GHz (e.g., 1.842).")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="The base model to use.")
    parser.add_argument("--outfile", type=str, default="generated_config.yaml", help="The name of the file to save the generated config.")
    args = parser.parse_args()

    # --- THE BEST HYPERPARAMETERS DISCOVERED BY THE AUTO-TUNER ---
    best_hps = {
        'repetition_penalty': 1.1,
        'temperature': 0.7,
        'top_p': 0.9
    }
    print(f"Using recommended hyperparameters: {best_hps}")

    # --- Load the Model and Tokenizer (same as before) ---
    print(f"Loading model: {args.model}...")
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb_config, device_map="auto")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    print("Model loaded successfully.")

    # --- Define the Prompt ---
    PROMPT_TEMPLATE = """You are a highly skilled RF engineer specializing in electronic countermeasures.
Your mission is to generate a complete YAML configuration file to effectively jam a target frequency.

### Instructions:
1.  Analyze the `High-Level Goal`.
2.  Determine the optimal values for **all** required configuration parameters.
3.  The output MUST be a single, valid YAML block containing all necessary keys.
4.  Use snake_case for all keys (e.g., `center_frequency`).
5.  Use scientific 'e' notation for frequencies and bandwidth.

### Example:
High-Level Goal: Jam a target at 0.915 GHz
### Example YAML Output:
amplitude: 0.9
amplitude_width: 0.1
center_frequency: 9.15e8
bandwidth: 10e6
initial_phase: 0
sampling_freq: 20e6
num_samples: 20000
output_iq_file: "output.fc32"
output_csv_file: "output.csv"
write_iq: false
write_csv: true
device_args: "type=b200"
tx_gain: 55

---

### Current Task:
High-Level Goal: Jam a target at {freq:.4f} GHz

### YAML Output:
"""

    # --- Generate the Configuration ---
    print(f"\nGenerating configuration for {args.freq} GHz...")
    prompt_text = PROMPT_TEMPLATE.format(freq=args.freq)
    inputs = tokenizer(prompt_text, return_tensors="pt").to(model.device)
    
    generation_config = GenerationConfig(max_new_tokens=250, pad_token_id=tokenizer.eos_token_id, do_sample=True, **best_hps)
    
    output_tokens = model.generate(**inputs, generation_config=generation_config)
    full_text = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # --- Parse, Validate, and Save the Output ---
    config = _parse_llm_output(full_text)

    if config:
        print("\n" + "="*20 + " GENERATED CONFIGURATION " + "="*20)
        # Convert dict to clean YAML string for printing
        clean_yaml = yaml.dump(config, sort_keys=False)
        print(clean_yaml)
        
        # Save the generated config to a file
        with open(args.outfile, 'w') as f:
            f.write(clean_yaml)
        print(f"Configuration successfully saved to '{args.outfile}'")

        # Run the simulation to score the generated config
        print("\n--- Running simulation on generated config... ---")
        score = mock_run_simulation_and_get_reward(config)
        print(f"\n--- VALIDATION COMPLETE ---")
        print(f"Achieved Score: {score:.4f}")

    else:
        print("\n---! GENERATION FAILED !---")
        print("The model failed to produce a valid YAML configuration.")
        print("Raw output was:")
        print(full_text.split("### YAML Output:")[-1].strip())

if __name__ == "__main__":
    main()