#!/usr/bin/env python3
"""
Git Exclusions Verification Script for NFL Big Data Bowl 2026

This script helps verify that large data files are properly excluded from git tracking.
Run this before committing to ensure the repository stays clean and lightweight.
"""

import os
import subprocess
from pathlib import Path

def get_file_size_mb(filepath):
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(filepath)
        return size_bytes / (1024 * 1024)
    except OSError:
        return 0

def check_git_tracking():
    """Check what files git is tracking vs ignoring"""
    
    print("NFL Big Data Bowl 2026 - Git Exclusions Verification")
    print("=" * 60)
    
    # Get git tracked files
    try:
        result = subprocess.run(['git', 'ls-files'], 
                              capture_output=True, text=True, check=True)
        tracked_files = result.stdout.strip().split('\n')
        tracked_files = [f for f in tracked_files if f]  # Remove empty strings
    except subprocess.CalledProcessError:
        print("Error: Not in a git repository or git not available")
        return
    
    print(f"Currently tracked files: {len(tracked_files)}")
    print()
    
    # Check for large files that might be tracked
    large_tracked_files = []
    for file in tracked_files:
        if os.path.exists(file):
            size_mb = get_file_size_mb(file)
            if size_mb > 1:  # Files larger than 1MB
                large_tracked_files.append((file, size_mb))
    
    if large_tracked_files:
        print("Large files currently tracked:")
        for file, size in large_tracked_files:
            print(f"   {file}: {size:.1f} MB")
        print()
    else:
        print("No large files are being tracked")
        print()
    
    # Check data directory
    data_dir = Path("data")
    if data_dir.exists():
        print("Data directory contents:")
        
        train_files = list(data_dir.glob("train/*.csv"))
        processed_files = list(data_dir.glob("processed/**/*.csv"))
        feature_files = list(data_dir.glob("features/**/*.csv"))
        
        print(f"   Training CSVs: {len(train_files)} files")
        print(f"   Processed CSVs: {len(processed_files)} files") 
        print(f"   Feature CSVs: {len(feature_files)} files")
        
        # Calculate total data size
        total_size = 0
        for pattern in ["train/*.csv", "processed/**/*", "features/**/*", "*.csv"]:
            for file in data_dir.glob(pattern):
                if file.is_file():
                    total_size += get_file_size_mb(file)
        
        print(f"   Total data size: {total_size:.1f} MB")
        print()
        
        # Check if any data files are tracked
        data_tracked = [f for f in tracked_files if f.startswith('data/')]
        if data_tracked:
            print("Data files being tracked (should be excluded):")
            for file in data_tracked:
                print(f"   {file}")
        else:
            print("No data files are being tracked")
    
    print()
    
    # Check models directory
    models_dir = Path("models")
    if models_dir.exists():
        model_files = list(models_dir.glob("**/*"))
        model_files = [f for f in model_files if f.is_file()]
        
        if model_files:
            total_model_size = sum(get_file_size_mb(f) for f in model_files)
            print(f"Model artifacts: {len(model_files)} files ({total_model_size:.1f} MB)")
            
            model_tracked = [f for f in tracked_files if f.startswith('models/')]
            if model_tracked:
                print("Model files being tracked (should be excluded):")
                for file in model_tracked:
                    print(f"   {file}")
            else:
                print("No model files are being tracked")
    
    print()
    
    # Check submissions directory
    submissions_dir = Path("submissions")
    if submissions_dir.exists():
        submission_files = list(submissions_dir.glob("*.csv"))
        if submission_files:
            print(f"Submission files: {len(submission_files)} files")
            
            submission_tracked = [f for f in tracked_files if f.startswith('submissions/')]
            if submission_tracked:
                print("Submission files being tracked (should be excluded):")
                for file in submission_tracked:
                    print(f"   {file}")
            else:
                print("No submission files are being tracked")
    
    print()
    print("Repository Health Summary:")
    print(f"   Clean tracking: {len(large_tracked_files) == 0}")
    print(f"   Data excluded: {len([f for f in tracked_files if f.startswith('data/')]) == 0}")
    print(f"   Models excluded: {len([f for f in tracked_files if f.startswith('models/')]) == 0}")
    print(f"   Submissions excluded: {len([f for f in tracked_files if f.startswith('submissions/')]) == 0}")
    
    print("\nTo add files safely:")
    print("   git add notebooks/ src/ requirements.txt README.md .gitignore")
    print("   git commit -m 'Add NFL Big Data Bowl 2026 project structure'")

if __name__ == "__main__":
    check_git_tracking()