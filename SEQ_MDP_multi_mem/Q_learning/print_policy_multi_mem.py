import pickle
import os
import numpy as np

def load_policy(filename):
    """Load a saved policy file"""
    with open(filename, 'rb') as f:
        policy = pickle.load(f)
    return policy

def print_policy(policy, filename, raw=False):
    """Print policy details with exclusive raw or formatted output"""
    if raw:
        print(f"\n{'='*50}")
        print(f"Raw contents of file: {filename}")
        print(f"{'='*50}")
        print(policy)
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"Formatted policy from file: {filename}")
        print(f"Number of state-action pairs: {len(policy)}")
        print(f"{'='*50}")
        
        for i, entry in enumerate(policy):
            state = entry['state']
            action_space = entry['action_space']
            one_hot = entry['policy']
            
            print(f"\nState-Action Pair {i + 1}:")
            print("-" * 30)
            print("State matrix:")
            print(state)
            print(f"Action space (possible swap nodes): {action_space}")
            print(f"Policy (one-hot): {one_hot}")
            optimal_action = action_space[np.argmax(one_hot)] if one_hot and action_space else None
            print(f"Optimal swap node: {optimal_action if optimal_action else 'None'}")
        print(f"{'='*50}")

def main():
    # Directory containing saved files
    directory = "training_data"
    print_raw = False  # Set to True for raw output only, False for formatted output only
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found!")
        return
    
    # Find all policy files
    policy_files = [f for f in os.listdir(directory) if f.endswith('.pkl') and not f.endswith('_rewards.pkl')]
    
    if not policy_files:
        print("No policy files found in training_data directory!")
        return
    
    for policy_file in policy_files:
        full_path = os.path.join(directory, policy_file)
        
        # Load and print policy
        policy = load_policy(full_path)
        print_policy(policy, policy_file, raw=print_raw)
    
    print("\nPolicy printing complete.")

if __name__ == "__main__":
    main()