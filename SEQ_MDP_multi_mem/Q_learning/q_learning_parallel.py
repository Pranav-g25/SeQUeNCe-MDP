import os
import pickle
import time
import logging
from functools import partial
from multiprocessing import Pool, cpu_count

# Import the core environment and training function
from q_learning_multi_mem8 import QuantumNetworkEnv, train_agent

# Configure root logger to include timestamps and process names
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(processName)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)


def run_experiment(args):
    """
    Run one Q-learning experiment with logging of start, end, and duration.
    args: (index, total_runs, p, ps, C, alpha, gamma, epsilon, episodes)
    """
    idx, total_runs, p, ps, C, alpha, gamma, epsilon, episodes = args
    start_time = time.time()
    logging.info(f"[{idx}/{total_runs}] START p={p}, ps={ps}, C={C}, α={alpha}, γ={gamma}, ε={epsilon}")

    # Initialize environment and train
    env = QuantumNetworkEnv(p, ps, C)
    policy, rewards = train_agent(env, episodes, alpha, gamma, epsilon)

    # Save results
    p_str = "_".join(map(str, p))
    c_str = "_".join(map(str, C))
    filename_base = (
        f"training_data/policy_p{p_str}_ps{ps}_C{c_str}"
        f"_a{alpha}_g{gamma}_e{epsilon}"
    )
    os.makedirs("training_data", exist_ok=True)
    with open(f"{filename_base}.pkl", "wb") as f:
        pickle.dump(policy, f)
    with open(f"{filename_base}_rewards.pkl", "wb") as f:
        pickle.dump(rewards, f)

    elapsed = time.time() - start_time
    mins, secs = divmod(elapsed, 60)
    logging.info(
        f"[{idx}/{total_runs}] DONE p={p}, ps={ps}, C={C} in {int(mins)}m {int(secs)}s"
    )


def main():
    # Define your experiments here as full tuples:
    raw_runs = [
        ([0.3, 0.3, 0.3], 0.5, [4, 8, 8, 4], 0.01, 0.9, 0.1, 5000),
        ([0.3, 0.3, 0.3, 0.3], 0.5, [5, 5, 5, 5, 5], 0.01, 0.9, 0.1, 5000),
        ([0.3, 0.3, 0.3, 0.3], 0.5, [4, 4, 4, 4, 4], 0.01, 0.9, 0.1, 5000),
        ([0.3, 0.3, 0.3, 0.3], 0.5, [6, 6, 6, 6, 6], 0.01, 0.9, 0.1, 5000),

        # ... add more runs as needed
    ]
    total_runs = len(raw_runs)
    # Prepend indices and total count to each run tuple
    runs = [
        (i+1, total_runs, *params)
        for i, params in enumerate(raw_runs)
    ]

    logging.info(f"Launching {total_runs} experiments on {cpu_count()} cores...")
    start_all = time.time()

    # Parallel execution
    with Pool(processes=cpu_count()) as pool:
        pool.map(run_experiment, runs)

    total_elapsed = time.time() - start_all
    mins, secs = divmod(total_elapsed, 60)
    logging.info(f"All {total_runs} experiments completed in {int(mins)}m {int(secs)}s")


if __name__ == "__main__":
    main()
