"""
NFL Big Data Bowl 2026 - Feature Engineering Module
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')

def calculate_velocity_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate velocity-based features."""
    df = df.copy()
    
    # Convert direction to radians for velocity components
    dir_rad = np.radians(df['dir'])
    
    # Velocity components
    df['v_x'] = df['s'] * np.cos(dir_rad)
    df['v_y'] = df['s'] * np.sin(dir_rad)
    
    # Acceleration components (if acceleration available)
    if 'a' in df.columns:
        df['a_x'] = df['a'] * np.cos(dir_rad)
        df['a_y'] = df['a'] * np.sin(dir_rad)
    
    return df

def calculate_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate temporal features within each play."""
    df = df.copy()
    
    # Sort by play and frame
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    # Calculate changes between frames
    df['dx'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['x'].diff()
    df['dy'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['y'].diff()
    df['ds'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['s'].diff()
    df['da'] = df.groupby(['game_id', 'play_id', 'nfl_id'])['a'].diff()
    
    # Calculate derived speeds
    df['speed_change'] = np.sqrt(df['dx']**2 + df['dy']**2) * 10  # 10 fps
    
    return df

def calculate_relative_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features relative to ball landing location and other players."""
    df = df.copy()
    
    # Distance and angle to ball landing
    df['dist_to_ball'] = np.sqrt((df['x'] - df['ball_land_x'])**2 + 
                                (df['y'] - df['ball_land_y'])**2)
    
    df['angle_to_ball'] = np.arctan2(df['ball_land_y'] - df['y'], 
                                    df['ball_land_x'] - df['x'])
    
    # Relative velocity toward ball
    v_x = df['s'] * np.cos(np.radians(df['dir']))
    v_y = df['s'] * np.sin(np.radians(df['dir']))
    
    ball_dir_x = (df['ball_land_x'] - df['x']) / (df['dist_to_ball'] + 1e-6)
    ball_dir_y = (df['ball_land_y'] - df['y']) / (df['dist_to_ball'] + 1e-6)
    
    df['velocity_toward_ball'] = v_x * ball_dir_x + v_y * ball_dir_y
    
    return df

def calculate_player_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate features based on interactions between players."""
    df = df.copy()
    
    # For each frame, calculate distances to other players
    features_list = []
    
    for (game_id, play_id, frame_id), frame_group in df.groupby(['game_id', 'play_id', 'frame_id']):
        frame_features = frame_group.copy()
        
        # Calculate distances to all other players
        positions = frame_group[['x', 'y']].values
        distances = np.sqrt(((positions[:, np.newaxis] - positions)**2).sum(axis=2))
        
        # Features for each player
        frame_features['min_distance_to_other'] = np.min(distances + np.eye(len(distances)) * 1000, axis=1)
        frame_features['avg_distance_to_others'] = np.mean(distances, axis=1)
        
        # Separate by team side
        offense_mask = frame_group['player_side'] == 'Offense'
        defense_mask = frame_group['player_side'] == 'Defense'
        
        if offense_mask.sum() > 0 and defense_mask.sum() > 0:
            offense_positions = frame_group.loc[offense_mask, ['x', 'y']].values
            defense_positions = frame_group.loc[defense_mask, ['x', 'y']].values
            
            # Distance to closest opponent
            for i, (idx, row) in enumerate(frame_group.iterrows()):
                if row['player_side'] == 'Offense':
                    opponent_distances = np.sqrt(((offense_positions[i:i+1] - defense_positions)**2).sum(axis=1))
                    frame_features.loc[idx, 'distance_to_closest_opponent'] = np.min(opponent_distances)
                else:
                    opponent_distances = np.sqrt(((defense_positions[i - offense_mask.sum():i - offense_mask.sum()+1] - offense_positions)**2).sum(axis=1))
                    frame_features.loc[idx, 'distance_to_closest_opponent'] = np.min(opponent_distances)
        
        features_list.append(frame_features)
    
    return pd.concat(features_list, ignore_index=True)

def calculate_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate physics-based features."""
    df = df.copy()
    
    # Momentum-based features
    df['momentum_x'] = df['s'] * np.cos(np.radians(df['dir']))
    df['momentum_y'] = df['s'] * np.sin(np.radians(df['dir']))
    df['momentum_magnitude'] = np.sqrt(df['momentum_x']**2 + df['momentum_y']**2)
    
    # Angular features
    df['orientation_diff'] = np.abs(df['dir'] - df['o'])
    df['orientation_diff'] = np.minimum(df['orientation_diff'], 360 - df['orientation_diff'])
    
    return df

def calculate_contextual_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate game context features."""
    df = df.copy()
    
    # Field position features
    df['distance_from_sideline'] = np.minimum(df['y'], 53.3 - df['y'])
    df['distance_from_endzone'] = np.minimum(df['x'], 120 - df['x'])
    
    # Play context
    df['frames_remaining'] = df['num_frames_output'] - df['frame_id']
    df['progress_through_play'] = df['frame_id'] / df['num_frames_output']
    
    return df

def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features."""
    df = df.copy()
    
    # Position encoding
    position_encoding = {
        'QB': 0, 'RB': 1, 'FB': 2, 'WR': 3, 'TE': 4,
        'T': 5, 'G': 6, 'C': 7, 'DE': 8, 'DT': 9,
        'NT': 10, 'LB': 11, 'CB': 12, 'S': 13, 'FS': 14, 'SS': 15
    }
    df['position_encoded'] = df['player_position'].map(position_encoding).fillna(-1)
    
    # Role encoding
    role_encoding = {
        'Pass': 0, 'Pass Block': 1, 'Pass Route': 2, 'Pass Rush': 3, 'Coverage': 4
    }
    df['role_encoded'] = df['player_role'].map(role_encoding).fillna(-1)
    
    # Side encoding
    df['side_encoded'] = (df['player_side'] == 'Offense').astype(int)
    
    return df

def create_sequence_features(df: pd.DataFrame, sequence_length: int = 4) -> pd.DataFrame:
    """Create lagged features for sequence modeling."""
    df = df.copy()
    df = df.sort_values(['game_id', 'play_id', 'nfl_id', 'frame_id'])
    
    feature_cols = ['x', 'y', 's', 'a', 'dir', 'o']
    
    for col in feature_cols:
        for lag in range(1, sequence_length + 1):
            df[f'{col}_lag{lag}'] = df.groupby(['game_id', 'play_id', 'nfl_id'])[col].shift(lag)
    
    return df

def create_target_receiver_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features related to the target receiver."""
    df = df.copy()
    
    # For each play, find the target receiver
    target_receivers = df[df['player_to_predict'] == True].groupby(['game_id', 'play_id']).first()[['nfl_id', 'x', 'y']].reset_index()
    target_receivers.columns = ['game_id', 'play_id', 'target_nfl_id', 'target_x', 'target_y']
    
    # Merge back to main dataframe
    df = df.merge(target_receivers, on=['game_id', 'play_id'], how='left')
    
    # Calculate features relative to target receiver
    df['distance_to_target_receiver'] = np.sqrt((df['x'] - df['target_x'])**2 + 
                                               (df['y'] - df['target_y'])**2)
    
    df['is_target_receiver'] = (df['nfl_id'] == df['target_nfl_id']).astype(int)
    
    return df

def create_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering functions."""
    print("Creating velocity features...")
    df = calculate_velocity_features(df)
    
    print("Creating temporal features...")
    df = calculate_temporal_features(df)
    
    print("Creating relative features...")
    df = calculate_relative_features(df)
    
    print("Creating physics features...")
    df = calculate_physics_features(df)
    
    print("Creating contextual features...")
    df = calculate_contextual_features(df)
    
    print("Encoding categorical features...")
    df = encode_categorical_features(df)
    
    print("Creating target receiver features...")
    df = create_target_receiver_features(df)
    
    print("Feature engineering complete!")
    return df

if __name__ == "__main__":
    # Example usage
    from data_processing import NFLDataLoader
    
    loader = NFLDataLoader()
    input_df, _ = loader.load_week_data(1)
    
    print(f"Original features: {input_df.shape[1]}")
    
    # Apply feature engineering
    featured_df = create_all_features(input_df.head(1000))  # Test on subset
    
    print(f"Features after engineering: {featured_df.shape[1]}")
    print(f"New feature columns: {set(featured_df.columns) - set(input_df.columns)}")