import numpy as np
import time
import json
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional
from math import sqrt
# Add parent directory to path to import modules
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))



def compute_dataset_properties(X: np.ndarray) -> Dict[str, Any]: # type: ignore
    """
    Compute spectral properties of a dataset matrix.
    
    Properties computed:
        - Condition number κ = L / μ
        - Stable rank = ||X||_F^2 / ||X||_2^2
        - Singular values
    
    Args:
        X: Feature matrix (n x d)
    
    Returns:
        properties: Dictionary containing all spectral properties
    """
    m, n = X.shape
    np.random.seed(42)  # For reproducibility
    # TODO: Compute spectral properties - can be skipped if not using 
    raise NotImplementedError("Spectral property computation not yet implemented")


def generate_gd_victory_dataset() -> Tuple[np.ndarray, np.ndarray, Dict]: #type: ignore
    """
    SCENARIO 1: Generate a dataset where Full-Batch Gradient Descent wins.
    Returns:
        X: Feature matrix
        y: Target vector
        properties: Dataset spectral properties
    """
    raise NotImplementedError("Dataset generation not yet implemented")


def generate_sgd_victory_dataset() -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    SCENARIO 2: Generate a dataset where Stochastic methods win.
    
    Returns:
        X: Feature matrix
        y: Target vector  
        properties: Dataset spectral properties
    """
    print("\n" + "="*70)
    print("SCENARIO 2: SGD VICTORY - High Rank + Large Dataset")
    print("="*70)
    
    np.random.seed(123)  # For reproducibility
    
    raise NotImplementedError("Dataset generation not yet implemented")

# =============================================================================
# DATASET SAVING AND VALIDATION
# =============================================================================

def save_dataset(X: np.ndarray, y: np.ndarray, properties: Dict, 
                 scenario_name: str, output_dir: str = "generated_datasets") -> Path:
    """
    Save generated dataset to disk.
    
    Args:
        X: Feature matrix
        y: Target vector
        properties: Dataset properties dictionary
        scenario_name: Name for this scenario
        output_dir: Output directory path
    
    Returns:
        save_path: Path where dataset was saved
    """
    output_path = Path(__file__).parent / output_dir / scenario_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save matrices
    np.save(output_path / "X.npy", X)
    np.save(output_path / "y.npy", y)
    
    # Save metadata
    metadata = {
        "scenario": scenario_name,
        "properties": properties,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "shape": X.shape
    }
    
    with open(output_path / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nDataset saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    print("\n" + "#"*70)
    print("# TASK C: THE UNO REVERSE")
    print("#"*70)
    
    # Generate Scenario 1: GD Victory
    try:
        print("\nGenerating Scenario 1...")
        X_gd, y_gd, props_gd = generate_gd_victory_dataset()
        
        save_dataset(X_gd, y_gd, props_gd, "scenario1_gd_victory")
    
    except NotImplementedError:
        print("Scenario 1 not yet implemented")
    
    # Generate Scenario 2: SGD Victory
    try:
        print("\nGenerating Scenario 2...")
        X_sgd, y_sgd, props_sgd = generate_sgd_victory_dataset()
        
        save_dataset(X_sgd, y_sgd, props_sgd, "scenario2_sgd_victory")
    except NotImplementedError:
        print("Scenario 2 not yet implemented")
    
    print("\n" + "#"*70)
    print("# TASK C COMPLETED")
    print("#"*70)
    
