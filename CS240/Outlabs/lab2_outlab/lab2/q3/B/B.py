import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple, List, Dict, Any, Type
import matplotlib.pyplot as plt
from time import time as timer

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algos.optim import LSLROptimiser

# Try importing the algorithm classes (may not all be implemented)
AVAILABLE_ALGOS: Dict[str, Type[LSLROptimiser]] = {}

try:
    from algos import LSLRAlgo1
    AVAILABLE_ALGOS["<algo1_name> (Algo1)"] = LSLRAlgo1
except Exception as e:
    print(f"Warning: LSLRAlgo1 not available: {e}")

try:
    from algos import LSLRAlgo2
    AVAILABLE_ALGOS["<algo2_name> (Algo2)"] = LSLRAlgo2
except Exception as e:
    print(f"Warning: LSLRAlgo2 not available: {e}")

try:
    from algos import LSLRAlgo3
    AVAILABLE_ALGOS["<algo3_name> (Algo3)"] = LSLRAlgo3
except Exception as e:
    print(f"Warning: LSLRAlgo3 not available: {e}")


def read_dataset(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X.npy and y.npy from the folder."""
    X = np.load(f"{folder_path}/X.npy")
    y = np.load(f"{folder_path}/y.npy")
    return X, y


def exact_solve(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve for w using Moore-Penrose pseudoinverse.
    Returns (w_optimal, time_taken).
    """
    start_time = timer()
    w_optimal = np.linalg.pinv(X) @ y
    elapsed_time = timer() - start_time
    return w_optimal, elapsed_time


class LSLRTrainer:
    """
    Trainer for LSLR optimization algorithms.
    Supports both fixed-epoch training and convergence-based training.
    """
    
    def __init__(self, optimiser: LSLROptimiser, initial_point: np.ndarray,
                 output_dir: Optional[Path] = None) -> None:
        """
        Initialize the trainer.
        
        Args:
            optimiser: LSLROptimiser instance (LSLRAlgo1, LSLRAlgo2, or LSLRAlgo3)
            initial_point: Starting point w_0
            output_dir: Directory to save outputs (plots, metrics)
        """
        self.optimiser = optimiser
        self.initial_point = initial_point.copy()
        self.current_point = initial_point.copy()
        
        # Output directory setup
        self.output_dir = output_dir or Path(__file__).parent / "outputs"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.epochs_history: List[int] = []
        self.mse_history: List[float] = []
        self.time_history: List[float] = []  # Cumulative time at each epoch
        self.grad_norm_history: List[float] = []
        self.delta_w_history: List[float] = []  # ||w_t - w_{t-1}||
        self.params_history: List[np.ndarray] = []
        
        # Timing
        self.total_time: float = 0.0
        self.time_per_epoch: float = 0.0
        
    def reset(self) -> None:
        """Reset trainer to initial state."""
        self.current_point = self.initial_point.copy()
        self.optimiser.epochs_done = 0
        self.epochs_history = []
        self.mse_history = []
        self.time_history = []
        self.grad_norm_history = []
        self.delta_w_history = []
        self.params_history = []
        self.total_time = 0.0
        self.time_per_epoch = 0.0
    
    def _log_epoch(self, epoch: int, cumulative_time: float, prev_point: np.ndarray) -> None:
        """Log metrics for current epoch."""
        mse = self.optimiser.eval_lslr(self.current_point)
        grad = self.optimiser.full_grad(self.current_point)
        grad_norm = np.linalg.norm(grad)
        delta_w = np.linalg.norm(self.current_point - prev_point)
        
        self.epochs_history.append(epoch)
        self.mse_history.append(float(mse))
        self.time_history.append(cumulative_time)
        self.grad_norm_history.append(float(grad_norm))
        self.delta_w_history.append(float(delta_w))
        self.params_history.append(self.current_point.copy())
    
    def run_vanilla(self, n_epochs: int) -> Dict[str, Any]:
        """
        Run optimization for a fixed number of epochs.
        """
        self.reset()
        
        start_time = timer()
        prev_point = self.current_point.copy()
        
        # Log initial state
        self._log_epoch(0, 0.0, prev_point)
        
        for epoch in range(1, n_epochs + 1):
            prev_point = self.current_point.copy()
            
            # Optimization step
            self.current_point = self.optimiser.step(self.current_point)
            
            # Log metrics
            cumulative_time = timer() - start_time
            self._log_epoch(epoch, cumulative_time, prev_point)
        
        end_time = timer()
        self.total_time = end_time - start_time
        self.time_per_epoch = self.total_time / n_epochs if n_epochs > 0 else 0.0
        
        results = {
            "final_point": self.current_point.copy(),
            "final_mse": self.mse_history[-1],
            "n_epochs": n_epochs,
            "total_time": self.total_time,
            "time_per_epoch": self.time_per_epoch,
            "mse_history": self.mse_history.copy(),
            "epochs_history": self.epochs_history.copy(),
            "time_history": self.time_history.copy(),
            "grad_norm_history": self.grad_norm_history.copy(),
            "delta_w_history": self.delta_w_history.copy()
        }
        
        return results
    
    def run_until_convergence(self, w_exact: np.ndarray, tolerance: float = 1e-6,
                               max_epochs: int = 100000) -> Dict[str, Any]:
        """
        Run optimization until MSE matches exact solution within tolerance.
        """
        self.reset()
        
        # Compute target MSE from exact solution
        mse_exact = self.optimiser.eval_lslr(w_exact)
        
        start_time = timer()
        prev_point = self.current_point.copy()
        
        # Log initial state
        self._log_epoch(0, 0.0, prev_point)
        
        converged = False
        epochs_required = 0
        
        for epoch in range(1, max_epochs + 1):
            prev_point = self.current_point.copy()
            
            # Optimization step
            self.current_point = self.optimiser.step(self.current_point)
            
            # Log metrics
            cumulative_time = timer() - start_time
            self._log_epoch(epoch, cumulative_time, prev_point)
            
            # Check convergence
            current_mse = self.mse_history[-1]
            if abs(current_mse - mse_exact) <= tolerance:
                converged = True
                epochs_required = epoch
                break
            
            # Also check if MSE is below absolute tolerance
            if current_mse <= tolerance:
                converged = True
                epochs_required = epoch
                break
        
        if not converged:
            epochs_required = max_epochs
        
        end_time = timer()
        self.total_time = end_time - start_time
        self.time_per_epoch = self.total_time / epochs_required if epochs_required > 0 else 0.0
        
        results = {
            "final_point": self.current_point.copy(),
            "final_mse": self.mse_history[-1],
            "mse_exact": float(mse_exact),
            "tolerance": tolerance,
            "converged": converged,
            "epochs_required": epochs_required,
            "max_epochs": max_epochs,
            "total_time": self.total_time,
            "time_per_epoch": self.time_per_epoch,
            "mse_history": self.mse_history.copy(),
            "epochs_history": self.epochs_history.copy(),
            "time_history": self.time_history.copy(),
            "grad_norm_history": self.grad_norm_history.copy(),
            "delta_w_history": self.delta_w_history.copy()
        }
        
        return results
    
    def plot_mse_vs_epochs(self, title: str = "MSE vs Epochs", 
                           save_name: str = "mse_vs_epochs.png") -> None:
        """Plot and save MSE vs Epochs."""
        if not self.mse_history:
            print("No data to plot. Run training first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.epochs_history, self.mse_history, 'b-', linewidth=2, marker='o', 
                markersize=3, markevery=max(1, len(self.epochs_history)//20))
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add final MSE annotation
        final_mse = self.mse_history[-1]
        ax.axhline(y=final_mse, color='r', linestyle='--', alpha=0.5, 
                   label=f'Final MSE: {final_mse:.2e}')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_mse_vs_time(self, title: str = "MSE vs Wall-Clock Time",
                         save_name: str = "mse_vs_time.png") -> None:
        """Plot and save MSE vs Wall-Clock Time."""
        if not self.mse_history or not self.time_history:
            print("No data to plot. Run training first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.time_history, self.mse_history, 'g-', linewidth=2, marker='s',
                markersize=3, markevery=max(1, len(self.time_history)//20))
        ax.set_xlabel('Time (seconds)', fontsize=12)
        ax.set_ylabel('MSE', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        # Add timing info
        total_time = self.time_history[-1] if self.time_history else 0
        ax.axvline(x=total_time, color='r', linestyle='--', alpha=0.5,
                   label=f'Total Time: {total_time:.4f}s')
        ax.legend()
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def plot_delta_w_vs_epochs(self, title: str = "||w_t - w_{t-1}|| vs Epochs",
                                save_name: str = "delta_w_vs_epochs.png") -> None:
        """Plot and save ||w_t - w_{t-1}|| vs Epochs for debugging."""
        if not self.delta_w_history:
            print("No data to plot. Run training first.")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.plot(self.epochs_history, self.delta_w_history, 'm-', linewidth=2, marker='^',
                markersize=3, markevery=max(1, len(self.epochs_history)//20))
        ax.set_xlabel('Epochs', fontsize=12)
        ax.set_ylabel('||w_t - w_{t-1}||', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved: {save_path}")
    
    def save_experiment_metrics(self, results: Dict[str, Any], 
                                 algo_name: str, dataset_name: str = "dataset") -> None:
        """Save experiment metrics to JSON file."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "algorithm": algo_name,
            "learning_rate": self.optimiser.lr(),
            "results": {
                "final_mse": results.get("final_mse"),
                "mse_exact": results.get("mse_exact"),
                "epochs_required": results.get("epochs_required", results.get("n_epochs")),
                "converged": results.get("converged", True),
                "tolerance": results.get("tolerance"),
                "total_time": results.get("total_time"),
                "time_per_epoch": results.get("time_per_epoch"),
            },
            "initial_point_norm": float(np.linalg.norm(self.initial_point)),
            "final_point_norm": float(np.linalg.norm(results["final_point"])),
        }
        
        save_path = self.output_dir / f"experiment_metrics_{algo_name.replace(' ', '_').replace('(', '').replace(')', '')}.json"
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {save_path}")
    
    def save_full_history(self, algo_name: str, filename: str = None) -> None:
        """Save full training history to npz file."""
        if filename is None:
            filename = f"training_history_{algo_name.replace(' ', '_').replace('(', '').replace(')', '')}.npz"
        save_path = self.output_dir / filename
        np.savez(save_path,
                 epochs=np.array(self.epochs_history),
                 mse=np.array(self.mse_history),
                 time=np.array(self.time_history),
                 grad_norm=np.array(self.grad_norm_history),
                 delta_w=np.array(self.delta_w_history))
        print(f"Saved: {save_path}")


def run_single_algorithm(algo_class: Type[LSLROptimiser], algo_name: str,
                         X: np.ndarray, y: np.ndarray, w_exact: np.ndarray,
                         tolerance: float, max_epochs: int, 
                         output_dir: Path, dataset_name: str) -> Optional[Dict[str, Any]]:
    """
    Run a single algorithm and return results.
    """
    print(f"\n{'='*60}")
    print(f"Running: {algo_name}")
    print(f"{'='*60}")
    
    try:
        # Create optimizer
        optimiser = algo_class(X, y)
        print(f"  Learning rate: {optimiser.lr():.6e}")
        
        # Initial point (zeros)
        n_features = X.shape[1]
        w_init = np.zeros(n_features)
        
        # Create trainer
        trainer = LSLRTrainer(optimiser, w_init, output_dir=output_dir)
        
        # Run until convergence
        results = trainer.run_until_convergence(
            w_exact=w_exact,
            tolerance=tolerance,
            max_epochs=max_epochs
        )
        
        # Print results
        print(f"\nResults:")
        print(f"  Converged: {results['converged']}")
        print(f"  Epochs required: {results['epochs_required']}")
        print(f"  Total time: {results['total_time']:.6f} seconds")
        print(f"  Time per epoch: {results['time_per_epoch']:.6e} seconds")
        print(f"  Final MSE: {results['final_mse']:.10e}")
        print(f"  Target MSE: {results['mse_exact']:.10e}")
        print(f"  MSE gap: {abs(results['final_mse'] - results['mse_exact']):.10e}")
        
        # Save plots
        safe_name = algo_name.replace(' ', '_').replace('(', '').replace(')', '')
        trainer.plot_mse_vs_epochs(
            title=f"MSE vs Epochs ({algo_name}, Dataset {dataset_name})",
            save_name=f"mse_vs_epochs_{safe_name}_{dataset_name}.png"
        )
        trainer.plot_mse_vs_time(
            title=f"MSE vs Time ({algo_name}, Dataset {dataset_name})",
            save_name=f"mse_vs_time_{safe_name}_{dataset_name}.png"
        )
        trainer.plot_delta_w_vs_epochs(
            title=f"||w_t - w_{{t-1}}|| vs Epochs ({algo_name}, Dataset {dataset_name})",
            save_name=f"delta_w_vs_epochs_{safe_name}_{dataset_name}.png"
        )
        
        # Save metrics
        trainer.save_experiment_metrics(results, algo_name, dataset_name)
        trainer.save_full_history(algo_name)
        
        results['algo_name'] = algo_name
        return results
        
    except Exception as e:
        print(f"  ERROR: {algo_name} failed with: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task B: Algorithm LeaderBoard for LSLR")
    parser.add_argument("--data_folder", type=str, default="../data/datasets/A/", 
                        help="Path to the folder containing dataset (X.npy and y.npy)")
    parser.add_argument("--tolerance", type=float, default=1e-1,
                        help="Convergence tolerance for MSE")
    parser.add_argument("--max_epochs", type=int, default=10000,
                        help="Maximum number of epochs")
    parser.add_argument("--algo", type=str, default="all",
                        choices=["all", "1", "2", "3"],
                        help="Which algorithm to run (1, 2, 3, or all)")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TASK B: Algorithm LeaderBoard for LSLR")
    print("="*70)
    
    # Print available algorithms
    print(f"\nAvailable algorithms: {list(AVAILABLE_ALGOS.keys())}")

    # 1. Load dataset
    print(f"\n1. Loading dataset from: {args.data_folder}")
    X, y = read_dataset(args.data_folder)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    n_samples, n_features = X.shape

    # 2. Exact solve
    print("\n2. Exact Solution (Moore-Penrose Pseudoinverse)")
    print("-" * 50)
    w_exact, exact_time = exact_solve(X, y)
    mse_exact = np.mean((X @ w_exact - y) ** 2)
    print(f"   Time taken: {exact_time:.6f} seconds")
    print(f"   MSE (exact): {mse_exact:.10e}")
    print(f"   ||w*||: {np.linalg.norm(w_exact):.6f}")

    # Extract dataset name
    dataset_name = Path(args.data_folder).name

    # 3. Run algorithms
    print(f"\n3. Running Algorithms (tolerance={args.tolerance:.2e}, max_epochs={args.max_epochs})")
    print("-" * 50)
    
    all_results = []
    
    # Determine which algorithms to run
    if args.algo == "all":
        algos_to_run = list(AVAILABLE_ALGOS.items())
    else:
        algo_map = {"1": "GD (Algo1)", "2": "Kaczmarz (Algo2)", "3": "SVRG (Algo3)"}
        algo_name = algo_map.get(args.algo)
        if algo_name and algo_name in AVAILABLE_ALGOS:
            algos_to_run = [(algo_name, AVAILABLE_ALGOS[algo_name])]
        else:
            print(f"Algorithm {args.algo} not available")
            algos_to_run = []
    
    for algo_name, algo_class in algos_to_run:
        result = run_single_algorithm(
            algo_class, algo_name, X, y, w_exact,
            args.tolerance, args.max_epochs, output_dir, dataset_name
        )
        if result:
            all_results.append(result)

    # 4. Summary comparison
    if len(all_results) > 1:
        print("\n" + "="*70)
        print("COMPARISON SUMMARY")
        print("="*70)
        print(f"\n{'Algorithm':<25} {'Converged':<12} {'Epochs':<12} {'Time (s)':<12} {'Final MSE':<15}")
        print("-"*70)
        
        for r in all_results:
            print(f"{r['algo_name']:<25} {str(r['converged']):<12} {r['epochs_required']:<12} "
                  f"{r['total_time']:<12.6f} {r['final_mse']:<15.6e}")
        
        # Find best algorithm (lowest MSE if all converged, else fastest to converge)
        converged_results = [r for r in all_results if r['converged']]
        if converged_results:
            best_by_time = min(converged_results, key=lambda x: x['total_time'])
            print(f"\nBest by wall-clock time: {best_by_time['algo_name']} "
                  f"({best_by_time['total_time']:.6f}s)")
        
        best_by_mse = min(all_results, key=lambda x: x['final_mse'])
        print(f"Best by final MSE: {best_by_mse['algo_name']} "
              f"(MSE={best_by_mse['final_mse']:.6e})")

    # 5. Comparison with exact solve
    print("\n" + "="*70)
    print("EXACT vs ITERATIVE COMPARISON")
    print("="*70)
    print(f"  Exact solve time: {exact_time:.6f} seconds")
    for r in all_results:
        # speedup = exact_time / r['total_time'] if r['total_time'] > 0 else float('inf')
        print(f"  {r['algo_name']:<25}: {r['total_time']:.6f}s )")
    
    print("\n" + "="*70)
    print("Task B Complete! Check 'outputs/' folder for plots and metrics.")
    print("="*70)
