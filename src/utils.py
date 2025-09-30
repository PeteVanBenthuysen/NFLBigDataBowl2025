"""
NFL Big Data Bowl 2026 - Utility Functions
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def setup_plotting_style():
    """Setup consistent plotting style across notebooks."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 11

def validate_environment():
    """Validate that all required packages are available."""
    required_packages = [
        'pandas', 'numpy', 'matplotlib', 'seaborn', 'plotly', 
        'sklearn', 'scipy', 'jupyter'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            missing_packages.append(package)
            print(f"✗ {package} (missing)")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✓ All required packages are available!")
        return True

def check_data_directory(data_dir: str = "data") -> bool:
    """Check if data directory and files exist."""
    data_path = Path(data_dir)
    
    if not data_path.exists():
        print(f"⚠ Data directory not found: {data_path.resolve()}")
        return False
    
    train_dir = data_path / "train"
    if not train_dir.exists():
        print(f"⚠ Training directory not found: {train_dir.resolve()}")
        return False
    
    # Check for expected files
    expected_files = [
        "test_input.csv",
        "test.csv", 
        "sample_submission.csv"
    ]
    
    missing_files = []
    for file in expected_files:
        if not (data_path / file).exists():
            missing_files.append(file)
    
    # Check training files
    train_files = list(train_dir.glob("*.csv"))
    expected_train_files = 36  # 18 input + 18 output files
    
    print(f"✓ Data directory found: {data_path.resolve()}")
    print(f"✓ Training files found: {len(train_files)}/{expected_train_files}")
    
    if missing_files:
        print(f"⚠ Missing files: {', '.join(missing_files)}")
        return False
    
    print("✓ All data files present!")
    return True

def get_memory_usage(df: pd.DataFrame) -> str:
    """Get formatted memory usage of a DataFrame."""
    memory_mb = df.memory_usage(deep=True).sum() / 1024**2
    return f"{memory_mb:.1f} MB"

def quick_data_summary(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """Print quick summary of DataFrame."""
    print(f"=== {name.upper()} SUMMARY ===")
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Memory usage: {get_memory_usage(df)}")
    print(f"Missing values: {df.isnull().sum().sum():,}")
    print(f"Duplicate rows: {df.duplicated().sum():,}")
    
    # Data types summary
    dtype_counts = df.dtypes.value_counts()
    print(f"Data types: {dict(dtype_counts)}")

def create_directory_structure():
    """Create the expected directory structure if it doesn't exist."""
    directories = [
        "data/train",
        "notebooks", 
        "src",
        "models",
        "submissions",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {directory}/")

def calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error - competition metric."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def evaluate_coordinates_prediction(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Evaluate coordinate predictions (x, y).
    
    Args:
        y_true: Ground truth coordinates (N, 2) 
        y_pred: Predicted coordinates (N, 2)
    
    Returns:
        Dictionary with RMSE metrics
    """
    if y_true.shape != y_pred.shape:
        raise ValueError(f"Shape mismatch: {y_true.shape} vs {y_pred.shape}")
    
    # Overall RMSE (competition metric)
    rmse_total = calculate_rmse(y_true, y_pred)
    
    # Separate RMSE for x and y coordinates
    rmse_x = calculate_rmse(y_true[:, 0], y_pred[:, 0])
    rmse_y = calculate_rmse(y_true[:, 1], y_pred[:, 1])
    
    # Mean absolute error
    mae_total = np.mean(np.abs(y_true - y_pred))
    mae_x = np.mean(np.abs(y_true[:, 0] - y_pred[:, 0]))
    mae_y = np.mean(np.abs(y_true[:, 1] - y_pred[:, 1]))
    
    return {
        'rmse_total': rmse_total,
        'rmse_x': rmse_x,
        'rmse_y': rmse_y,
        'mae_total': mae_total,
        'mae_x': mae_x,
        'mae_y': mae_y
    }

def format_submission_id(game_id: int, play_id: int, nfl_id: int, frame_id: int) -> str:
    """Format submission ID according to competition requirements."""
    return f"{game_id}_{play_id}_{nfl_id}_{frame_id}"

def parse_submission_id(submission_id: str) -> Tuple[int, int, int, int]:
    """Parse submission ID back to components."""
    parts = submission_id.split('_')
    if len(parts) != 4:
        raise ValueError(f"Invalid submission ID format: {submission_id}")
    
    return tuple(map(int, parts))

def save_submission(predictions: pd.DataFrame, filename: str = None) -> str:
    """
    Save predictions in competition submission format.
    
    Args:
        predictions: DataFrame with columns ['id', 'x', 'y']
        filename: Output filename (optional)
    
    Returns:
        Path to saved file
    """
    if filename is None:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"submission_{timestamp}.csv"
    
    submissions_dir = Path("submissions")
    submissions_dir.mkdir(exist_ok=True)
    
    filepath = submissions_dir / filename
    predictions.to_csv(filepath, index=False)
    
    print(f"✓ Submission saved: {filepath}")
    print(f"  Shape: {predictions.shape}")
    print(f"  Columns: {list(predictions.columns)}")
    
    return str(filepath)

if __name__ == "__main__":
    print("NFL Big Data Bowl 2026 - Utility Functions")
    print("==========================================")
    
    # Validate environment
    validate_environment()
    print()
    
    # Check data
    check_data_directory()
    print()
    
    # Create directories
    create_directory_structure()
    print()
    
    print("✓ Setup validation complete!")
    print("Ready to start data analysis!")