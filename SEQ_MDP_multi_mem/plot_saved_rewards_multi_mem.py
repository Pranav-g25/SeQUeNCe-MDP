import pickle
import os
import numpy as np
import matplotlib.pyplot as plt

def load_rewards(filename):
    """Load saved rewards file"""
    with open(filename, 'rb') as f:
        rewards = pickle.load(f)
    return rewards

def calculate_moving_average(rewards, window_size=100):
    """Calculate moving average of rewards"""
    return np.convolve(rewards, np.ones(window_size)/window_size, mode='valid')

def plot_rewards(rewards_files, directory="training_data", window_size=100):
    """Plot raw rewards and moving average for all reward files"""
    for rewards_file in rewards_files:
        full_path = os.path.join(directory, rewards_file)
        rewards = load_rewards(full_path)
        episodes = range(len(rewards))
        
        # Calculate moving average
        if len(rewards) >= window_size:
            moving_avg = calculate_moving_average(rewards, window_size)
            avg_episodes = range(window_size-1, len(rewards))
        else:
            moving_avg = None
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot raw rewards
        plt.plot(episodes, rewards, 'b-', alpha=0.3, label='Raw Rewards')
        
        # Plot moving average if available
        if moving_avg is not None:
            plt.plot(avg_episodes, moving_avg, 'r-', linewidth=2, label=f'Moving Average (window={window_size})')
        
        plt.xlabel('Episode')
        plt.ylabel('Total Reward')
        plt.title(f'Rewards per Episode - {rewards_file}')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        output_file = os.path.join(directory, f"rewards_plot_{rewards_file.replace('.pkl', '.png')}")
        plt.savefig(output_file)
        plt.close()
        
        print(f"Plot saved as: {output_file}")

def main():
    directory = "training_data"
    window_size = 200  # Adjustable window size for moving average
    
    if not os.path.exists(directory):
        print(f"Directory '{directory}' not found!")
        return
    
    # Find all rewards files
    rewards_files = [f for f in os.listdir(directory) if f.endswith('_rewards.pkl')]
    
    if not rewards_files:
        print("No rewards files found in training_data directory!")
        return
    
    plot_rewards(rewards_files, directory, window_size)
    
    print("\nReward plotting complete.")

if __name__ == "__main__":
    main()