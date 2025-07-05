# auto_tuner_agent.py (Now with full command-line control for testing)

import torch
import json
import re
import random
import argparse
from tqdm import tqdm
from itertools import product
import torch.nn as nn
from datasets import Dataset

# All imports needed for your specific TRL version's workflow
from transformers import (
    AutoTokenizer, GenerationConfig, BitsAndBytesConfig,
    AutoModelForCausalLM, LlamaConfig, LlamaForSequenceClassification
)
from transformers.modeling_outputs import SequenceClassifierOutput
from trl import PPOTrainer as TRL_PPOTrainer, PPOConfig
from simulation_environment import mock_run_simulation_and_get_reward
import simulation_environment

# ===================================================================
# SECTION 1: AUTO-TUNER AGENT and HELPER CLASSES/FUNCTIONS
# ===================================================================
class HyperparameterAgent:
    def __init__(self, action_space: dict):
        self.action_space = action_space
        self.q_table = {}
        self.learning_rate = 0.1
        self.epsilon = 0.9
        self.epsilon_decay = 0.95 # Slightly faster decay for shorter runs
        self.min_epsilon = 0.1

    def get_action(self) -> dict:
        actions = self._get_all_actions()
        # Epsilon-Greedy Strategy
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

class SimulationRewardModel(LlamaForSequenceClassification):
    # ... (This class is correct and remains unchanged) ...
    def __init__(self, tokenizer, model_name):
        config = LlamaConfig.from_pretrained(model_name); config.num_labels=1; super().__init__(config); self.tokenizer=tokenizer
    def _parse_llm_output(self, t):
        try:
            match = re.search(r'\{[^{}]*\}', t, re.DOTALL)
            return json.loads(match.group(0)) if match else None
        except: return None
    def forward(self, input_ids=None, **kwargs):
        rewards=[]
        for i in range(input_ids.shape[0]):
            text = self.tokenizer.decode(input_ids[i], skip_special_tokens=True)
            config = self._parse_llm_output(text)
            reward = mock_run_simulation_and_get_reward(config) if config else -1.0
            rewards.append(reward)
        return SequenceClassifierOutput(loss=None, logits=torch.tensor(rewards, device=self.device).unsqueeze(-1))

class LLMAgent:
    # This class incorporates the dtype mismatch fix
    def __init__(self, model_name, ppo_cfg, train_dataset):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model_dtype = torch.bfloat16

        self.model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=model_dtype, device_map="auto")
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=model_dtype, device_map="auto")
        self.value_model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=model_dtype, device_map="auto")
        
        self.value_model.score = nn.Linear(self.value_model.config.hidden_size, 1, bias=False).to(device=self.device, dtype=model_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None: self.tokenizer.pad_token = self.tokenizer.eos_token
        self.reward_model = SimulationRewardModel(self.tokenizer, model_name).to(device=self.device, dtype=model_dtype)
        self.model.generation_config = GenerationConfig(pad_token_id=self.tokenizer.pad_token_id, eos_token_id=self.tokenizer.eos_token_id, max_new_tokens=ppo_cfg.response_length, temperature=ppo_cfg.temperature, do_sample=True)
        self.ppo_trainer = TRL_PPOTrainer(args=ppo_cfg, model=self.model, ref_model=self.ref_model, reward_model=self.reward_model, value_model=self.value_model, processing_class=self.tokenizer, train_dataset=train_dataset)

# ===================================================================
# SECTION 2: THE MODEL TRAINER (The "Inner Loop" Function)
# ===================================================================
def execute_training_run(hparams: dict, base_model_name: str, inner_episodes: int) -> float:
    print(f"\n--- [Worker] Starting run with HPs: {hparams} ---")
    ppo_cfg = PPOConfig(
        learning_rate=hparams['learning_rate'], num_ppo_epochs=hparams['num_ppo_epochs'], kl_coef=hparams['kl_coef'],
        total_episodes=inner_episodes, exp_name="autotuner_worker", seed=random.randint(0, 10000),
        per_device_train_batch_size=2, kl_estimator="k3", temperature=0.95, response_length=150,
        num_sample_generations=0, bf16=True, **{'gamma': 0.99, 'lam': 0.95, 'cliprange': 0.2, 'cliprange_value': 0.2, 'vf_coef': 0.1}
    )
    base_prompt = "You are a world-class network security expert..."
    raw_prompts = [{"query": f"{base_prompt} ...jam a target at {round(random.uniform(3.5, 3.7), 4):.4f} GHz.\n\n### JSON Output:\n"} for _ in range(inner_episodes)]
    tokenizer = AutoTokenizer.from_pretrained(base_model_name)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_dataset = Dataset.from_list(raw_prompts).map(lambda x: tokenizer(x["query"])).remove_columns(["query"])
    train_dataset.set_format("torch")
    agent = LLMAgent(model_name=base_model_name, ppo_cfg=ppo_cfg, train_dataset=train_dataset)
    print(f"\n--- [Worker] Starting RL Training for {inner_episodes} episodes... ---")
    agent.ppo_trainer.train()
    print("\n--- [Worker] Evaluating final model performance... ---")
    total_score = 0
    test_frequencies = [3.55, 3.60, 3.65] # Reduced test set for speed
    for freq in test_frequencies:
        prompt_text = f"{base_prompt} ...jam a target at {freq:.4f} GHz.\n\n### JSON Output:\n"
        inputs = agent.tokenizer(prompt_text, return_tensors="pt").to(agent.device)
        output_tokens = agent.model.generate(**inputs, max_new_tokens=150, temperature=0.1, pad_token_id=agent.tokenizer.eos_token_id)
        full_text = agent.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        config = agent.reward_model._parse_llm_output(full_text)
        score = mock_run_simulation_and_get_reward(config) if config else -1.0
        total_score += score
    avg_score = total_score / len(test_frequencies)
    print(f"--- [Worker] Run Complete. Final average score: {avg_score:.4f} ---")
    del agent
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return avg_score

# ===================================================================
# SECTION 3: THE MAIN CONTROLLER
# ===================================================================
def main():
    # --- NEW: Command-line arguments to control the loops ---
    parser = argparse.ArgumentParser(description="Auto-Tuner for LLM Fine-Tuning.")
    parser.add_argument("--model", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="Base model to tune.")
    parser.add_argument("--outer_loops", type=int, default=5, help="Number of hyperparameter sets to test (meta-episodes).")
    parser.add_argument("--inner_loops", type=int, default=10, help="Number of PPO training episodes for each hyperparameter set.")
    args = parser.parse_args()

    # Reduced search space for quicker local testing
    hyperparameter_space = {
        'learning_rate': [1e-5, 5e-5],
        'num_ppo_epochs': [2, 4],
        'kl_coef': [0.05, 0.1]
    }
    
    auto_tuner = HyperparameterAgent(hyperparameter_space)
    
    print("="*20 + " STARTING AUTO-TUNER AGENT " + "="*20)
    print(f"Running for {args.outer_loops} outer loops, with {args.inner_loops} inner loops each.")

    for i in range(args.outer_loops):
        print(f"\n{'='*15} Auto-Tuner Episode {i+1}/{args.outer_loops} {'='*15}")
        chosen_hps = auto_tuner.get_action()
        reward = execute_training_run(
            hparams=chosen_hps, 
            base_model_name=args.model, 
            inner_episodes=args.inner_loops
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
    
    print(f"\nRECOMMENDED HYPERPARAMETERS: {dict(sorted_q_table[0][0])}")

if __name__ == "__main__":
    main()