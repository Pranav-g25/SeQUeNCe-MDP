import os
import multiprocessing
import subprocess
import sys
import time
import logging

def run_training(config):
    """Run a single training configuration and save results."""
    run_id, N, p, ps, C, alpha, gamma, epsilon = config
    logger = logging.getLogger(__name__)
    try:
        # Convert parameters to strings for filename
        p_str = "_".join(map(str, p))
        c_str = "_".join(map(str, C))
        filename = f"training_data/policy_N{N}_p{p_str}_ps{ps}_C{c_str}_a{alpha}_g{gamma}_e{epsilon}_run{run_id}"
        policy_file = f"{filename}.pkl"
        rewards_file = f"{filename}_rewards.pkl"
        plot_file = f"{filename}_rewards_plot.png"
        log_file = f"training_data/log_N{N}_p{p_str}_ps{ps}_C{c_str}_a{alpha}_g{gamma}_e{epsilon}_run{run_id}.log"

        # Prepare the command to run q_learning_multi_mem.py with arguments
        cmd = [
            sys.executable, "q_learning_multi_mem.py",
            str(N),
            str(p),
            str(ps),
            str(C),
            str(alpha),
            str(gamma),
            str(epsilon),
            str(run_id),
            "--log-interval", "100"
        ]

        # Ensure training_data directory exists
        os.makedirs("training_data", exist_ok=True)

        # Run the command
        logger.info(f"Starting Run {run_id}: N={N}, p={p}, ps={ps}, C={C}, alpha={alpha}, gamma={gamma}, epsilon={epsilon}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            # Check if all expected files exist
            missing_files = []
            for f in [policy_file, rewards_file, plot_file, log_file]:
                if not os.path.exists(f):
                    missing_files.append(f)
            if missing_files:
                logger.error(f"Run {run_id} completed but files missing: {', '.join(missing_files)}")
                return run_id, False
            logger.info(f"Run {run_id} completed successfully. Policy: {policy_file}, Rewards: {rewards_file}, Plot: {plot_file}, Log: {log_file}")
            return run_id, True
        else:
            logger.error(f"Run {run_id} failed. Error: {result.stderr}")
            return run_id, False
    except Exception as e:
        logger.error(f"Run {run_id} encountered an error: {e}")
        return run_id, False

def main():
    # Setup logging
    os.makedirs("training_data", exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler("training_data/parallel_training.log"),
            logging.StreamHandler(sys.stdout)
        ],
        force=True
    )
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Define the 2 configurations (based on your run)
    configurations = [
        (1, 5, [0.3, 0.3, 0.3, 0.3], 1.0, [5, 5, 5, 5, 5], 0.01, 0.95, 0.1),
        (2, 5, [0.5, 0.5, 0.5, 0.5], 1.0, [5, 5, 5, 5, 5], 0.01, 0.95, 0.1)
    ]

    # Determine the number of parallel workers (use CPU count - 1, but at least 1)
    max_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Starting parallel training with {max_workers} workers for {len(configurations)} configurations")

    start_time = time.time()
    
    # Run configurations in parallel
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.map(run_training, configurations)
    
    # Summarize results
    successful_runs = sum(1 for run_id, success in results if success)
    total_duration = time.time() - start_time
    duration_str = f"{int(total_duration // 60)}m {int(total_duration % 60)}s"
    logger.info(f"Parallel training completed. {successful_runs}/{len(configurations)} runs successful. Total duration: {duration_str}")
    logger.info("Each successful run saved a policy file, rewards file, reward plot, and log file in the 'training_data' directory.")

if __name__ == "__main__":
    main()