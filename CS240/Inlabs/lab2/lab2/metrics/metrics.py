import numpy as np
import json
import time
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path


class DatasetMetrics:
    """Static properties of the dataset/function (computed once)."""
    
    def __init__(self, X: Optional[np.ndarray] = None, name: str = "dataset"):
        """
        Args:
            X: Feature matrix (for LSLR problems)
            name: Identifier for the dataset
        """
        self.name = name
        self.n_samples = X.shape[0] if X is not None else None
        self.n_features = X.shape[1] if X is not None else None
        
        # Spectral properties (for LSLR)
        if X is not None:
            self._compute_spectral_properties(X)
        else:
            self.L = None
            self.mu = None
            self.kappa = None
            self.stable_rank = None
            self.singular_values = None
    
    def _compute_spectral_properties(self, X: np.ndarray) -> None:
        """Compute Lipschitz constant, strong convexity, condition number, stable rank."""
        # For LSLR: Hessian = X^T X / n
        XTX = X.T @ X / X.shape[0]
        eigenvalues = np.linalg.eigvalsh(XTX)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]  # Filter near-zero
        
        self.L = np.max(eigenvalues)  # Lipschitz constant
        self.mu = np.min(eigenvalues) if len(eigenvalues) > 0 else 0  # Strong convexity
        self.kappa = self.L / self.mu if self.mu > 1e-10 else np.inf  # Condition number
        
        # Stable rank: sum(sigma_i^2) / sigma_max^2
        singular_values = np.linalg.svd(X, compute_uv=False)
        self.singular_values = singular_values
        self.stable_rank = np.sum(singular_values**2) / (singular_values[0]**2) if singular_values[0] > 0 else 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "name": self.name,
            "n_samples": self.n_samples,
            "n_features": self.n_features,
            "L": float(self.L) if self.L is not None else None,
            "mu": float(self.mu) if self.mu is not None else None,
            "kappa": float(self.kappa) if self.kappa is not None else None,
            "stable_rank": float(self.stable_rank) if self.stable_rank is not None else None,
            "singular_values": self.singular_values.tolist() if self.singular_values is not None else None
        }


class RunMetrics:
    """Dynamic metrics tracked during optimization (per algorithm run)."""
    
    def __init__(self, algorithm_name: str, hyperparams: Dict[str, Any]):
        """
        Args:
            algorithm_name: Name of the optimizer (e.g., "GD", "SGD", "Newton")
            hyperparams: Dict of hyperparameters (lr, batch_size, etc.)
        """
        self.algorithm_name = algorithm_name
        self.hyperparams = hyperparams
        
        # Time series data (lists for efficient append)
        self.iterations: List[int] = []
        self.wall_times: List[float] = []  # Cumulative wall-clock time
        self.mse_values: List[float] = []
        self.grad_norms: List[float] = []
        self.param_values: List[np.ndarray] = []  # Store x_t at each iteration
        
        # Tracking state
        self.start_time: Optional[float] = None
        self.total_time: float = 0.0
        self.converged: bool = False
        self.final_mse: Optional[float] = None
    
    def start_timer(self) -> None:
        """Start timing the optimization run."""
        self.start_time = time.perf_counter()
    
    def log_iteration(self, iteration: int, mse: float, grad_norm: float, 
                      params: np.ndarray) -> None:
        """
        Log metrics for a single iteration.
        
        Args:
            iteration: Current iteration number
            mse: Mean squared error at current iterate
            grad_norm: L2 norm of gradient
            params: Current parameter vector x_t
        """
        if self.start_time is None:
            raise RuntimeError("Must call start_timer() before logging iterations")
        
        elapsed = time.perf_counter() - self.start_time
        
        self.iterations.append(iteration)
        self.wall_times.append(elapsed)
        self.mse_values.append(mse)
        self.grad_norms.append(grad_norm)
        self.param_values.append(params.copy())
    
    def finalize(self) -> None:
        """Finalize the run (compute total time, final MSE)."""
        if self.start_time is not None:
            self.total_time = time.perf_counter() - self.start_time
        if len(self.mse_values) > 0:
            self.final_mse = self.mse_values[-1]
    
    def get_mse_at_time(self, target_time: float) -> Optional[float]:
        """
        Find MSE at a specific wall-clock time (via interpolation).
        
        Args:
            target_time: Target time in seconds
        
        Returns:
            MSE value at that time, or None if time exceeds run duration
        """
        if target_time > self.total_time:
            return None
        
        # Find closest recorded time
        idx = np.searchsorted(self.wall_times, target_time)
        if idx >= len(self.mse_values):
            return self.mse_values[-1]
        if idx == 0:
            return self.mse_values[0]
        
        # Linear interpolation
        t0, t1 = self.wall_times[idx-1], self.wall_times[idx]
        mse0, mse1 = self.mse_values[idx-1], self.mse_values[idx]
        alpha = (target_time - t0) / (t1 - t0) if t1 > t0 else 0
        return mse0 + alpha * (mse1 - mse0)
    
    def get_time_for_mse(self, target_mse: float) -> Optional[float]:
        """
        Find first time when MSE drops below target.
        
        Args:
            target_mse: Target MSE threshold
        
        Returns:
            Time in seconds, or None if never achieved
        """
        for t, mse in zip(self.wall_times, self.mse_values):
            if mse <= target_mse:
                return t
        return None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary (excludes large arrays)."""
        return {
            "algorithm_name": self.algorithm_name,
            "hyperparams": self.hyperparams,
            "total_time": self.total_time,
            "final_mse": self.final_mse,
            "converged": self.converged,
            "num_iterations": len(self.iterations),
            # Summary statistics
            "min_mse": float(np.min(self.mse_values)) if self.mse_values else None,
            "final_grad_norm": float(self.grad_norms[-1]) if self.grad_norms else None,
        }
    
    def save_arrays(self, save_dir: Path) -> None:
        """Save time-series arrays as NPZ file."""
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        filename = f"{self.algorithm_name}_{hash(str(self.hyperparams))}.npz"
        np.savez(
            save_dir / filename,
            iterations=np.array(self.iterations),
            wall_times=np.array(self.wall_times),
            mse_values=np.array(self.mse_values),
            grad_norms=np.array(self.grad_norms),
            param_values=np.array(self.param_values)
        )


class ExperimentMetrics:
    """Container for multiple runs + dataset info."""
    
    def __init__(self, dataset_metrics: DatasetMetrics, experiment_name: str = "experiment"):
        """
        Args:
            dataset_metrics: Static dataset properties
            experiment_name: Name for this experiment
        """
        self.experiment_name = experiment_name
        self.dataset_metrics = dataset_metrics
        self.runs: List[RunMetrics] = []
    
    def add_run(self, run: RunMetrics) -> None:
        """Add a completed optimization run."""
        self.runs.append(run)
    
    def get_best_run(self, metric: str = "final_mse") -> Optional[RunMetrics]:
        """
        Find best run according to a metric.
        
        Args:
            metric: One of "final_mse", "total_time", "num_iterations"
        
        Returns:
            RunMetrics object with best performance
        """
        if not self.runs:
            return None
        
        if metric == "final_mse":
            return min(self.runs, key=lambda r: r.final_mse if r.final_mse else float('inf'))
        elif metric == "total_time":
            return min(self.runs, key=lambda r: r.total_time)
        else:
            return min(self.runs, key=lambda r: len(r.iterations))
    
    def compare_at_time(self, target_time: float) -> Dict[str, float]:
        """
        Compare all runs at a specific wall-clock time.
        
        Returns:
            Dict mapping algorithm name to MSE at that time
        """
        comparison = {}
        for run in self.runs:
            mse = run.get_mse_at_time(target_time)
            if mse is not None:
                comparison[f"{run.algorithm_name}_{run.hyperparams}"] = mse
        return comparison
    
    def save(self, save_dir: Path) -> None:
        """
        Save experiment to disk.
        - Saves metadata as JSON
        - Saves time-series arrays as NPZ files
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save metadata
        metadata = {
            "experiment_name": self.experiment_name,
            "dataset": self.dataset_metrics.to_dict(),
            "runs": [run.to_dict() for run in self.runs]
        }
        
        with open(save_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Save arrays for each run
        arrays_dir = save_dir / "arrays"
        for run in self.runs:
            run.save_arrays(arrays_dir)
    
    @classmethod
    def load(cls, save_dir: Path) -> 'ExperimentMetrics':
        """Load experiment from disk (metadata only, arrays loaded on-demand)."""
        save_dir = Path(save_dir)
        
        with open(save_dir / "metadata.json", "r") as f:
            metadata = json.load(f)
        
        # Reconstruct dataset metrics
        ds_dict = metadata["dataset"]
        dataset_metrics = DatasetMetrics(X=None, name=ds_dict["name"])
        dataset_metrics.n_samples = ds_dict["n_samples"]
        dataset_metrics.n_features = ds_dict["n_features"]
        dataset_metrics.L = ds_dict["L"]
        dataset_metrics.mu = ds_dict["mu"]
        dataset_metrics.kappa = ds_dict["kappa"]
        dataset_metrics.stable_rank = ds_dict["stable_rank"]
        
        experiment = cls(dataset_metrics, metadata["experiment_name"])
        # Note: Run reconstruction would require loading NPZ files
        
        return experiment


if __name__ == "__main__":
    # Example usage
    print("Metrics system ready for optimization experiments!")
    
    # Demo: Create synthetic dataset
    np.random.seed(42)
    X = np.random.randn(100, 10)
    y = np.random.randn(100)
    
    # Compute dataset metrics
    ds_metrics = DatasetMetrics(X, name="synthetic_demo")
    print(f"\nDataset Properties:")
    print(f"  Condition Number (κ): {ds_metrics.kappa:.2f}")
    print(f"  Stable Rank: {ds_metrics.stable_rank:.2f}")
    print(f"  Lipschitz (L): {ds_metrics.L:.4f}")
    
    # Create experiment
    experiment = ExperimentMetrics(ds_metrics, "demo_experiment")
    
    # Simulate a run
    run = RunMetrics("GD", {"lr": 0.01})
    run.start_timer()
    
    for i in range(5):
        time.sleep(0.01)  # Simulate computation
        mse = 1.0 / (i + 1)  # Fake decreasing MSE
        grad_norm = 0.5 / (i + 1)
        params = np.random.randn(10)
        run.log_iteration(i, mse, grad_norm, params)
    
    run.finalize()
    experiment.add_run(run)
    
    # Query
    print(f"\nRun completed in {run.total_time:.3f}s")
    print(f"MSE at t=0.02s: {run.get_mse_at_time(0.02):.4f}")
    print(f"Time to reach MSE < 0.3: {run.get_time_for_mse(0.3):.4f}s")