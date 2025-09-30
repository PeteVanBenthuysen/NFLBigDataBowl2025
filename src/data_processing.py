"""
NFL Big Data Bowl 2026 - Data Processing Utilities
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class NFLDataLoader:
    """Class for loading and preprocessing NFL tracking data."""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.train_dir = self.data_dir / "train"
        
    def load_week_data(self, week: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load input and output data for a specific week."""
        input_file = self.train_dir / f"input_2023_w{week:02d}.csv"
        output_file = self.train_dir / f"output_2023_w{week:02d}.csv"
        
        input_df = pd.read_csv(input_file)
        output_df = pd.read_csv(output_file)
        
        return input_df, output_df
    
    def load_all_training_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and concatenate all training data."""
        all_input = []
        all_output = []
        
        for week in range(1, 19):  # Weeks 1-18
            try:
                input_df, output_df = self.load_week_data(week)
                all_input.append(input_df)
                all_output.append(output_df)
                print(f"Loaded week {week}: {len(input_df)} input rows, {len(output_df)} output rows")
            except FileNotFoundError:
                print(f"Week {week} data not found, skipping...")
                continue
        
        combined_input = pd.concat(all_input, ignore_index=True)
        combined_output = pd.concat(all_output, ignore_index=True)
        
        return combined_input, combined_output
    
    def load_test_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load test input and test identifier data."""
        test_input = pd.read_csv(self.data_dir / "test_input.csv")
        test_ids = pd.read_csv(self.data_dir / "test.csv")
        
        return test_input, test_ids
    
    def load_sample_submission(self) -> pd.DataFrame:
        """Load sample submission format."""
        return pd.read_csv(self.data_dir / "sample_submission.csv")

def parse_play_id(df: pd.DataFrame) -> pd.DataFrame:
    """Parse and add play-level identifiers."""
    df = df.copy()
    df['play_key'] = df['game_id'].astype(str) + '_' + df['play_id'].astype(str)
    return df

def get_play_summary(input_df: pd.DataFrame) -> pd.DataFrame:
    """Get summary statistics per play."""
    play_summary = input_df.groupby(['game_id', 'play_id']).agg({
        'frame_id': ['min', 'max', 'nunique'],
        'nfl_id': 'nunique',
        'num_frames_output': 'first',
        'ball_land_x': 'first',
        'ball_land_y': 'first',
        'play_direction': 'first',
        'absolute_yardline_number': 'first'
    }).reset_index()
    
    # Flatten column names
    play_summary.columns = ['_'.join(col).strip() if col[1] else col[0] 
                           for col in play_summary.columns]
    
    return play_summary

def filter_prediction_players(df: pd.DataFrame) -> pd.DataFrame:
    """Filter to only players that need predictions."""
    return df[df['player_to_predict'] == True].copy()

def calculate_distances_to_ball(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate distance from each player to ball landing location."""
    df = df.copy()
    df['distance_to_ball_land'] = np.sqrt(
        (df['x'] - df['ball_land_x'])**2 + 
        (df['y'] - df['ball_land_y'])**2
    )
    return df

def add_relative_positions(df: pd.DataFrame) -> pd.DataFrame:
    """Add relative positions to ball landing location."""
    df = df.copy()
    df['rel_x_to_ball'] = df['x'] - df['ball_land_x']
    df['rel_y_to_ball'] = df['y'] - df['ball_land_y']
    return df

def standardize_field_direction(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize all plays to move in the same direction."""
    df = df.copy()
    
    # Flip coordinates for left-moving plays
    left_plays = df['play_direction'] == 'left'
    
    df.loc[left_plays, 'x'] = 120 - df.loc[left_plays, 'x']
    df.loc[left_plays, 'ball_land_x'] = 120 - df.loc[left_plays, 'ball_land_x']
    df.loc[left_plays, 'dir'] = (df.loc[left_plays, 'dir'] + 180) % 360
    df.loc[left_plays, 'o'] = (df.loc[left_plays, 'o'] + 180) % 360
    
    return df

if __name__ == "__main__":
    # Example usage
    loader = NFLDataLoader()
    
    # Load first week for testing
    input_df, output_df = loader.load_week_data(1)
    print(f"Week 1 - Input: {input_df.shape}, Output: {output_df.shape}")
    
    # Get play summary
    play_summary = get_play_summary(input_df)
    print(f"Play summary shape: {play_summary.shape}")
    print(f"Average frames per play: {play_summary['num_frames_output_first'].mean():.1f}")