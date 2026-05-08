import numpy as np
import json
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Callable, Optional, Tuple, List, Dict, Any

# Add parent directory to path to import modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from functions import func, rosenbrock, rot_anisotropic
from algos import Optimiser, GradientDescent
from metrics import DatasetMetrics, RunMetrics, ExperimentMetrics


class Trainer:
    def __init__(self, f: func, optimiser: Optimiser, initial_point: np.ndarray, 
                 max_iters: int = 1000, tol: float = 1e-6, record_metrics: bool = True) -> None:
        """
        Trainer for optimization algorithms with comprehensive metrics tracking.
        
        Args:
            f: Function to optimize
            optimiser: Optimizer instance
            initial_point: Starting point x_0
            max_iters: Maximum number of iterations
            tol: Convergence tolerance
            record_metrics: Whether to track detailed metrics
        """
        self.f = f
        self.optimiser = optimiser
        self.current_point = initial_point.copy()
        self.max_iters = max_iters
        self.tol = tol
        self.record_metrics = record_metrics
        
        # Metrics tracking
        self.run_metrics = None
        if record_metrics:
            hyperparams = {
                "lr": getattr(optimiser, 'lr', None),
                "max_iters": max_iters,
                "tol": tol,
                "initial_point": initial_point.tolist()
            }
            self.run_metrics = RunMetrics(
                algorithm_name=optimiser.__class__.__name__,
                hyperparams=hyperparams
            )

    def train(self) -> Tuple[np.ndarray, List[float]]:
        """
        Run optimization with full metrics tracking.
        
        Returns:
            final_point: Optimized parameters x^*
            history: List of function values at each iteration
        """
        history = []
        
        if self.run_metrics:
            self.run_metrics.start_timer()
        
        for i in range(self.max_iters):
            # Evaluate function and gradient
            current_value = self.f.eval(self.current_point)
            grad = self.f.grad(self.current_point)
            grad_norm = np.linalg.norm(grad)
            
            history.append(current_value)
            
            # Record metrics
            if self.run_metrics:
                self.run_metrics.log_iteration(i, current_value, grad_norm, self.current_point)
            
            # Optimization step
            next_point = self.optimiser.step(self.current_point)
            
            # Check convergence
            if np.linalg.norm(next_point - self.current_point) < self.tol:
                print(f"Converged at iteration {i+1}")
                break
            
            self.current_point = next_point
        
        if self.run_metrics:
            self.run_metrics.finalize()
        
        return self.current_point, history
    
    def get_metrics(self) -> Optional[RunMetrics]:
        """Return the metrics object."""
        return self.run_metrics


def run_rosenbrock_experiment(a: float = 1.0, b: float = 100.0, 
                               lr: float = 0.001, max_iters: int = 10000,
                               initial_point: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Run gradient descent on Rosenbrock function.
    
    Args:
        a: Rosenbrock parameter a
        b: Rosenbrock parameter b
        lr: Learning rate
        max_iters: Maximum iterations
        initial_point: Starting point (default: [-1.0, 2.0])
    
    Returns:
        Dictionary with results including x^*, analytic solution, MSE history, metrics
    """
    print("\n" + "="*60)
    print(f"ROSENBROCK EXPERIMENT: a={a}, b={b}, lr={lr}")
    print("="*60)
    
    # Create function
    f = rosenbrock(a=a, b=b)
    
    # Initial point
    if initial_point is None:
        initial_point = np.array([-1.0, 2.0])
    
    # Analytic solution: gradient = 0
    # For Rosenbrock: x^* = [a, a^2] = [1, 1] when a=1
    analytic_solution = np.array([a, a**2])
    
    # Create optimizer
    optimizer = GradientDescent(f, lr=lr)
    
    # Train
    trainer = Trainer(f, optimizer, initial_point, max_iters=max_iters, tol=1e-8)
    final_point, history = trainer.train()
    
    # Results
    final_mse = f.eval(final_point)
    error_from_analytic = np.linalg.norm(final_point - analytic_solution)
    
    print(f"\nFinal point x^*: {final_point}")
    print(f"Analytic solution: {analytic_solution}")
    print(f"Error from analytic: {error_from_analytic:.6e}")
    print(f"Final MSE: {final_mse:.6e}")
    print(f"Iterations: {len(history)}")
    
    # Get checkpoints at specific iterations
    checkpoints = {}
    for iter_check in [10, 100, 1000]:
        if iter_check < len(history):
            checkpoints[f"iter_{iter_check}"] = {
                "mse": history[iter_check],
                "iteration": iter_check
            }
    
    results = {
        "function": "rosenbrock",
        "parameters": {"a": a, "b": b},
        "hyperparameters": {"lr": lr, "max_iters": max_iters},
        "initial_point": initial_point.tolist(),
        "final_point": final_point.tolist(),
        "analytic_solution": analytic_solution.tolist(),
        "error_from_analytic": float(error_from_analytic),
        "final_mse": float(final_mse),
        "num_iterations": len(history),
        "mse_history": [float(v) for v in history],
        "checkpoints": checkpoints,
        "metrics": trainer.get_metrics().to_dict() if trainer.get_metrics() else None
    }
    
    return results


def run_rotated_anisotropic_experiment(U: np.ndarray, V: np.ndarray, 
                                        S: np.ndarray, b: np.ndarray,
                                        lr: float = 0.01, max_iters: int = 10000,
                                        initial_point: Optional[np.ndarray] = None,
                                        experiment_name: str = "default") -> Dict[str, Any]:
    """
    Run gradient descent on Rotated Anisotropic Quadratic: f(x) = x^T Q x - b^T x
    where Q = V S V^T
    
    Args:
        U: U matrix (not used in function, kept for reference)
        V: V matrix for rotation
        S: Diagonal matrix (singular values)
        b: Linear term vector
        lr: Learning rate
        max_iters: Maximum iterations
        initial_point: Starting point
        experiment_name: Name for this experiment
    
    Returns:
        Dictionary with results
    """
    print("\n" + "="*60)
    print(f"ROTATED ANISOTROPIC EXPERIMENT: {experiment_name}")
    print("="*60)
    
    # Create function
    f = rot_anisotropic(U, V, S, b)
    
    # Dimension
    d = V.shape[0]
    
    # Initial point
    if initial_point is None:
        initial_point = np.random.randn(d)
    

    # Analytic solution: ∇f = 0 => 2Qx - b = 0 => x^* = (Q^{-1} b) / 2
    Q = U @ S @ V.T
    try:
        analytic_solution = np.linalg.solve(Q+Q.T, b)
    except np.linalg.LinAlgError:
        analytic_solution = None
        print("Warning: Could not compute analytic solution (singular Q)")
    
    # Compute condition number
    eigenvalues = np.linalg.eigvalsh(Q)
    eigenvalues = eigenvalues[eigenvalues > 1e-10]

    # printing eigenvalues of Q
    print(f"Eigenvalues of Q: {eigenvalues}")

    if len(eigenvalues) > 0:
        L = np.max(eigenvalues)
        mu = np.min(eigenvalues)
        kappa = L / mu if mu > 1e-10 else np.inf
        stable_rank = np.sum(eigenvalues**2) / (eigenvalues[0]**2) if eigenvalues[0] > 0 else 0
    else:
        L, mu, kappa, stable_rank = None, None, None, None
    
    print(f"Condition number κ: {kappa}")
    print(f"Stable rank: {stable_rank}")
    print(f"Singular values: {np.diag(S)}")
    
    # Create optimizer
    optimizer = GradientDescent(f, lr=lr)
    
    # Train
    trainer = Trainer(f, optimizer, initial_point, max_iters=max_iters, tol=1e-8)
    final_point, history = trainer.train()
    
    # Results
    final_mse = f.eval(final_point)
    
    if analytic_solution is not None:
        error_from_analytic = np.linalg.norm(final_point - analytic_solution)
        print(f"\nFinal point x^*: {final_point}")
        print(f"Analytic solution: {analytic_solution}")
        print(f"Error from analytic: {error_from_analytic:.6e}")
    else:
        error_from_analytic = None
        print(f"\nFinal point x^*: {final_point}")
    

    #analytic min function value
    analytic_min = f.eval(analytic_solution) if analytic_solution is not None else None
    print(f"Analytic minimum function value: {analytic_min:.6e}" if analytic_min is not None else "Analytic minimum function value: N/A")
    print(f"Final function value: {final_mse:.6e}")
    print(f"Iterations: {len(history)}")
    
    # Get checkpoints
    checkpoints = {}
    for iter_check in [10, 100, 1000]:
        if iter_check < len(history):
            checkpoints[f"iter_{iter_check}"] = {
                "mse": history[iter_check],
                "iteration": iter_check
            }
    
    results = {
        "function": "rotated_anisotropic",
        "experiment_name": experiment_name,
        "parameters": {
            "U": U.tolist(),
            "V": V.tolist(),
            "S": S.tolist(),
            "b": b.tolist(),
            "singular_values": np.diag(S).tolist()
        },
        "spectral_properties": {
            "condition_number": float(kappa) if kappa else None,
            "stable_rank": float(stable_rank) if stable_rank else None,
            "L": float(L) if L else None,
            "mu": float(mu) if mu else None
        },
        "hyperparameters": {"lr": lr, "max_iters": max_iters},
        "initial_point": initial_point.tolist(),
        "final_point": final_point.tolist(),
        "analytic_solution": analytic_solution.tolist() if analytic_solution is not None else None,
        "error_from_analytic": float(error_from_analytic) if error_from_analytic else None,
        "final_function_value": float(final_mse),
        "num_iterations": len(history),
        "function_value_history": [float(v) for v in history],
        "checkpoints": checkpoints,
        "metrics": trainer.get_metrics().to_dict() if trainer.get_metrics() else None
    }
    
    return results


def save_results(results: Dict[str, Any], output_dir: str = "outputs") -> str:
    """
    Save results to JSON file with timestamp.
    
    Args:
        results: Results dictionary
        output_dir: Output directory path
    
    Returns:
        Path to saved file
    """
    # Create output directory
    output_path = Path(__file__).parent / output_dir
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Add timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results["timestamp"] = timestamp
    results["timestamp_24hr"] = datetime.now().strftime("%H:%M:%S")
    results["date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Generate filename
    function_name = results.get("function", "experiment")
    experiment_name = results.get("experiment_name", "")
    if experiment_name:
        filename = f"{function_name}_{experiment_name}_{timestamp}.json"
    else:
        filename = f"{function_name}_{timestamp}.json"
    
    filepath = output_path / filename
    
    # Save
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {filepath}")
    return str(filepath)


def run_all_experiments():
    """Run all miniature function experiments as specified in the assignment."""
    
    print("\n" + "#"*60)
    print("# MINIATURE FUNCTION GRADIENT DESCENT EXPERIMENTS")
    print("#"*60)
    
    all_results = []
    
    # =====================================================================
    # EXPERIMENT 1: Rosenbrock Function (a=1, b=100)
    # =====================================================================
    rosenbrock_results = run_rosenbrock_experiment(
        a=1.0, 
        b=100.0, 
        lr=0.001,
        max_iters=10000,
        initial_point=np.array([-1.0, 2.0])
    )
    save_results(rosenbrock_results)
    all_results.append(rosenbrock_results)
    
    # =====================================================================
    # EXPERIMENT 2: Rotated Anisotropic Quadratic
    # Define U, V matrices as given in assignment
    # =====================================================================
    
    U = np.array([
        [ 0.44718025, -0.11271586, -0.67027206,  0.56021303,  0.15563317],
        [-0.17470664, -0.07355529, -0.63440847, -0.472879  , -0.58135924],
        [ 0.21491915,  0.64201078, -0.24386904, -0.47489702,  0.50658921],
        [ 0.6547781 ,  0.38484544,  0.27653101,  0.04864147, -0.58679148],
        [-0.54275484,  0.64930802, -0.11099437,  0.48441062, -0.19194501]
    ])
    
    # V = np.array([
    #     [ 0.67490113, -0.36141784, -0.10150382,  0.5842934 ,  0.24936681],
    #     [ 0.1896522 ,  0.04660002, -0.9034855 , -0.21172844, -0.31740431],
    #     [-0.6090615 , -0.15872143, -0.411278  ,  0.27891672,  0.59741732],
    #     [-0.3625604 , -0.54175774,  0.04042161,  0.37215423, -0.65948149],
    #     [ 0.07832335, -0.74061572,  0.05125838, -0.63045935,  0.21271436]
    # ])

    # V = U + 0.00001 * np.random.randn(5,5)
    # Orthonormalize V using QR decomposition
    # V, _ = np.linalg.qr(V)  
    
    V = U


    #test ornthonormality by checking U.T @ U = I and V.T @ V = I with a tolerance
    assert np.allclose(U.T @ U, np.eye(5), atol=1e-6), "U is not orthonormal"
    assert np.allclose(V.T @ V, np.eye(5), atol=1e-6), "V is not orthonormal"
    
    # Random b vector
    np.random.seed(42)
    b = np.random.randn(5)
    
    # Test Case 1: σ_max very small (near-zero)
    print("\n" + "-"*60)
    print("Test Case 1: σ_max very small (σ = [0.01, 0.01, 0.01, 0.01, 0.01])")
    print("-"*60)
    S1 = np.diag([0.01, 0.01, 0.01, 0.01, 0.01])
    results1 = run_rotated_anisotropic_experiment(
        U, V, S1, b, 
        lr=0.0001,
        max_iters=100000,
        experiment_name="sigma_very_small"
    )
    save_results(results1)
    all_results.append(results1)
    
    # Test Case 2: σ_max = 100, others = 1 (ill-conditioned)
    print("\n" + "-"*60)
    print("Test Case 2: σ_max = 100, others = 1 (ill-conditioned)")
    print("-"*60)
    S2 = np.diag([100.0, 1.0, 1.0, 1.0, 1.0])
    results2 = run_rotated_anisotropic_experiment(
        U, V, S2, b,
        lr=0.0001,
        max_iters=100000,
        experiment_name="sigma_max_100"
    )
    save_results(results2)
    all_results.append(results2)
    
    # Test Case 3: All σ_i equal (well-conditioned)
    print("\n" + "-"*60)
    print("Test Case 3: All σ_i = 10 (well-conditioned)")
    print("-"*60)
    S3 = np.diag([10.0, 10.0, 10.0, 10.0, 10.0])
    results3 = run_rotated_anisotropic_experiment(
        U, V, S3, b,
        lr=0.0001,
        max_iters=100000,
        experiment_name="sigma_all_equal"
    )
    save_results(results3)
    all_results.append(results3)
    
    # =====================================================================
    # Save summary of all experiments
    # =====================================================================
    summary = {
        "timestamp": datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        "total_experiments": len(all_results),
        "experiments": [
            {
                "function": r["function"],
                "name": r.get("experiment_name", "rosenbrock"),
                "final_mse": r.get("final_function_value", r.get("final_mse")),
                "iterations": r["num_iterations"],
                "condition_number": r.get("spectral_properties", {}).get("condition_number") if "spectral_properties" in r else None
            }
            for r in all_results
        ]
    }
    
    summary_path = Path(__file__).parent / "outputs" / "summary_all_experiments.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "#"*60)
    print(f"# ALL EXPERIMENTS COMPLETED")
    print(f"# Summary saved to: {summary_path}")
    print("#"*60)
    

if __name__ == "__main__":
    run_all_experiments()