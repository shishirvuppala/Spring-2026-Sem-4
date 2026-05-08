import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple, List, Dict, Any
import matplotlib.pyplot as plt
from time import time as timer

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from algos import GradientDescent
from algos.optim import Optimiser
from metrics import DatasetMetrics, RunMetrics, ExperimentMetrics
from functions.func import func, LSLR


def read_dataset(folder_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load X.npy and y.npy from the folder."""
    X = np.load(f"{folder_path}/X.npy")
    y = np.load(f"{folder_path}/y.npy")
    return X, y


def solve_func(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Solve using pseudo-inverse (no assumption of full rank).
    
        returns the optimal weight vector w_optimal
    """

    ## TODO 
    X_pseudo_inv = np.linalg.pinv(X)
    w_optimal = X_pseudo_inv @ y
    return w_optimal

def exact_solve_function(X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Solve for w in least squares linear regression using Moore-Penrose pseudoinverse.
    
    Returns:
        w_optimal: The optimal weight vector
        elapsed_time: Time taken in seconds
    """
    ## TODO 
    start_time = timer()
    w_optimal = solve_func(X, y)
    end_time = timer()
    elapsed_time = end_time - start_time
    return w_optimal, elapsed_time

class Trainer:
    """
    Trainer for optimization algorithms with comprehensive metrics tracking.
    Supports both fixed-epoch training and convergence-based training.
    """
    
    def __init__(self, f: func, optimiser: Optimiser, initial_point: np.ndarray,
                 output_dir: Optional[Path] = None) -> None:
        """
        Initialize the trainer.
        
        Args:
            f: Function to optimize (LSLR instance)
            optimiser: Optimizer instance (e.g., GradientDescent)
            initial_point: Starting point w_0
            output_dir: Directory to save outputs (plots, metrics)
        """
        self.f = f
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
        self.params_history: List[np.ndarray] = []
        
        # Timing
        self.total_time: float = 0.0
        self.time_per_epoch: float = 0.0
        
    def reset(self) -> None:
        """Reset trainer to initial state."""
        self.current_point = self.initial_point.copy()
        self.epochs_history = []
        self.mse_history = []
        self.time_history = []
        self.grad_norm_history = []
        self.params_history = []
        self.total_time = 0.0
        self.time_per_epoch = 0.0
    
    def _log_epoch(self, epoch: int, cumulative_time: float) -> None:
        """Log metrics for current epoch."""
        mse = self.f.eval(self.current_point)
        grad = self.f.grad(self.current_point)
        grad_norm = np.linalg.norm(grad)
        
        self.epochs_history.append(epoch)
        self.mse_history.append(float(mse))
        self.time_history.append(cumulative_time)
        self.grad_norm_history.append(float(grad_norm))
        self.params_history.append(self.current_point.copy())
    
    def run_vanilla(self, n_epochs: int) -> Dict[str, Any]:
        """
        Run optimization for a fixed number of epochs.
        
        Args:
            n_epochs: Number of epochs to run
            
        Returns:
            Dictionary with results including final_point, mse_history, 
            total_time, time_per_epoch
        """
        self.reset()
        
        start_time = timer()
        
        # Log initial state
        self._log_epoch(0, 0.0)
        
        for epoch in range(1, n_epochs + 1):
            # Optimization step
            self.current_point = self.optimiser.step(self.current_point)
            
            # Log metrics
            cumulative_time = timer() - start_time
            self._log_epoch(epoch, cumulative_time)
        
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
            "grad_norm_history": self.grad_norm_history.copy()
        }
        
        return results
    
    def run_until_convergence(self, w_exact: np.ndarray, tolerance: float = 1e-6,
                               max_epochs: int = 100000) -> Dict[str, Any]:
        """
        Run optimization until MSE matches exact solution within tolerance.
        
        Args:
            w_exact: Optimal weights from exact solver
            tolerance: Convergence tolerance for MSE
            max_epochs: Maximum epochs before stopping
            
        Returns:
            Dictionary with results including epochs_required, convergence status,
            total_time, time_per_epoch
        """
        self.reset()
        
        # Compute target MSE from exact solution
        mse_exact = self.f.eval(w_exact)
        
        start_time = timer()
        
        # Log initial state
        self._log_epoch(0, 0.0)
        
        converged = False
        epochs_required = 0
        
        for epoch in range(1, max_epochs + 1):
            # Optimization step
            self.current_point = self.optimiser.step(self.current_point)
            
            # Log metrics
            cumulative_time = timer() - start_time
            self._log_epoch(epoch, cumulative_time)
            
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
            "grad_norm_history": self.grad_norm_history.copy()
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
    
    def save_experiment_metrics(self, results: Dict[str, Any], 
                                 dataset_name: str = "dataset") -> None:
        """Save experiment metrics to JSON file."""
        metrics = {
            "timestamp": datetime.now().isoformat(),
            "dataset": dataset_name,
            "algorithm": self.optimiser.__class__.__name__,
            "learning_rate": getattr(self.optimiser, 'lr', None),
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
        
        save_path = self.output_dir / "experiment_metrics.json"
        with open(save_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"Saved: {save_path}")
    
    def save_full_history(self, filename: str = "training_history.npz") -> None:
        """Save full training history to npz file."""
        save_path = self.output_dir / filename
        np.savez(save_path,
                 epochs=np.array(self.epochs_history),
                 mse=np.array(self.mse_history),
                 time=np.array(self.time_history),
                 grad_norm=np.array(self.grad_norm_history))
        print(f"Saved: {save_path}")


def compute_optimal_lr(X: np.ndarray) -> float:
    """
    Compute optimal learning rate for GD on LSLR.
    For f(w) = (1/n)||Xw - y||^2, Hessian = (2/n) X^T X
    Optimal lr = 1/L where L is the largest eigenvalue of Hessian.
    """
    n = X.shape[0]
    XTX = X.T @ X / n
    eigenvalues = np.linalg.eigvalsh(XTX)
    L = 2 * np.max(eigenvalues)  # Factor of 2 from our loss formulation
    optimal_lr = 1 / L
    return optimal_lr


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Task A: Exact Solver + Vanilla GD for LSLR")
    parser.add_argument("--data_folder", type=str, default="../data/datasets/D/", 
                        help="Path to the folder containing dataset (X.npy and y.npy)")
    parser.add_argument("--tolerance", type=float, default=1e-2,
                        help="Convergence tolerance for MSE")
    parser.add_argument("--max_epochs", type=int, default=10000,
                        help="Maximum number of epochs")
    parser.add_argument("--lr", type=float, default=None,
                        help="Learning rate (if None, uses optimal)")
    args = parser.parse_args()

    # Setup output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("TASK A: Least Squares Linear Regression")
    print("="*70)

    # 1. Load dataset
    print(f"\n1. Loading dataset from: {args.data_folder}")
    X, y = read_dataset(args.data_folder)
    print(f"   X shape: {X.shape}, y shape: {y.shape}")
    n_samples, n_features = X.shape

    # 2. Exact solve
    print("\n2. Exact Solution (Moore-Penrose Pseudoinverse)")
    print("-" * 50)
    w_exact, exact_time = exact_solve_function(X, y)
    mse_exact = np.mean((X @ w_exact - y) ** 2)
    print(f"   Time taken: {exact_time:.6f} seconds")
    print(f"   MSE (exact): {mse_exact:.10e}")
    print(f"   ||w*||: {np.linalg.norm(w_exact):.6f}")

    # 3. Setup Gradient Descent
    print("\n3. Gradient Descent Setup")
    print("-" * 50)
    
    # Create LSLR function
    lslr_func = LSLR(X, y)
    
    # Compute optimal learning rate
    optimal_lr = compute_optimal_lr(X)
    lr = args.lr if args.lr is not None else optimal_lr
    print(f"   Optimal LR (1/L): {optimal_lr:.6e}")
    print(f"   Using LR: {lr:.6e}")
    
    # Create optimizer
    gd_optimizer = GradientDescent(lslr_func, lr=lr)
    
    # Initial point (zeros)
    w_init = np.zeros(n_features)
    print(f"   Initial point: zeros (||w_0|| = 0)")
    print(f"   Initial MSE: {lslr_func.eval(w_init):.6e}")

    # 4. Create Trainer and run
    print("\n4. Training with Convergence Criterion")
    print("-" * 50)
    print(f"   Tolerance: {args.tolerance:.2e}")
    print(f"   Max epochs: {args.max_epochs}")
    
    trainer = Trainer(lslr_func, gd_optimizer, w_init, output_dir=output_dir)
    
    # Run until convergence
    results = trainer.run_until_convergence(
        w_exact=w_exact,
        tolerance=args.tolerance,
        max_epochs=args.max_epochs
    )

    # 5. Print Results
    print("\n5. Results")
    print("-" * 50)
    print(f"   Converged: {results['converged']}")
    print(f"   Epochs required: {results['epochs_required']}")
    print(f"   Total time: {results['total_time']:.6f} seconds")
    print(f"   Time per epoch: {results['time_per_epoch']:.6e} seconds")
    print(f"   Final MSE: {results['final_mse']:.10e}")
    print(f"   Target MSE (exact): {results['mse_exact']:.10e}")
    print(f"   MSE gap: {abs(results['final_mse'] - results['mse_exact']):.10e}")

    # 6. Save outputs
    print("\n6. Saving Outputs")
    print("-" * 50)
    
    # Extract dataset name from path
    dataset_name = Path(args.data_folder).name
    
    # Save plots
    trainer.plot_mse_vs_epochs(
        title=f"MSE vs Epochs (GD, Dataset {dataset_name})",
        save_name=f"mse_vs_epochs_{dataset_name}.png"
    )
    
    trainer.plot_mse_vs_time(
        title=f"MSE vs Time (GD, Dataset {dataset_name})",
        save_name=f"mse_vs_time_{dataset_name}.png"
    )
    
    # Save metrics
    trainer.save_experiment_metrics(results, dataset_name=dataset_name)
    trainer.save_full_history(filename=f"training_history_{dataset_name}.npz")

    # 7. Comparison Summary
    print("\n7. Comparison: Exact vs Iterative")
    print("-" * 50)
    print(f"   Exact solve time:    {exact_time:.6f} seconds")
    print(f"   GD total time:       {results['total_time']:.6f} seconds")
    print(f"   Speedup factor:      {exact_time / results['total_time']:.2f}x" 
          if results['total_time'] > 0 else "   N/A")
    
    print("\n" + "="*70)
    print("Task A Complete! Check 'outputs/' folder for plots and metrics.")
    print("="*70)


    



