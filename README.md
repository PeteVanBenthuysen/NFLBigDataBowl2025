# NFL Big Data Bowl 2026 - Player Movement Prediction Competition

## Competition Overview

**Objective**: Predict player x,y coordinates during the "ball in air" phase of NFL pass plays

**The Challenge**: When a quarterback releases the ball, predict where every player on the field will be positioned during each frame until the pass is caught or incomplete.

## Scoring Metric: Root Mean Squared Error (RMSE)

```
RMSE = √(Σ(predicted_x - actual_x)² + (predicted_y - actual_y)²) / n)
```

- Lower RMSE = Better predictions
- Evaluated on x,y coordinate accuracy for all players across all frames
- Live scoring on 2025 NFL season games (last 5 weeks)

## Competition Timeline

### Training Phase:
- **Start**: September 25, 2025
- **Team Merger Deadline**: November 26, 2025
- **Entry Deadline**: November 26, 2025  
- **Final Submission**: December 3, 2025

### Live Forecasting Phase:
- **Live Scoring**: December 4, 2025 - January 5, 2026
- **Results Published**: January 6, 2026

## Prizes
- **1st Place**: $25,000
- **2nd Place**: $15,000  
- **3rd Place**: $10,000
- **Total Pool**: $50,000

## Submission Rules
- **Daily Limit**: 5 submissions per day
- **Final Submissions**: 2 submissions count for final ranking
- **Format**: Kaggle Notebook (≤9 hours runtime)
- **Output**: `submission.csv` with format: `id,x,y`
- **ID Format**: `{game_id}_{play_id}_{nfl_id}_{frame_id}`

## Repository Structure

```
NFLBigDataBowl2026/
├── data/                           # Competition datasets (excluded from git)
│   ├── train/                      # 2023 NFL season training data
│   │   ├── input_2023_w01.csv     # Week 1 input features
│   │   ├── output_2023_w01.csv    # Week 1 target coordinates
│   │   └── ...                    # Weeks 2-18
│   ├── test_input.csv             # Test features for prediction
│   ├── test.csv                   # Test identifiers  
│   ├── processed/                 # Processed feature datasets
│   └── sample_submission.csv      # Submission format example
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA and data understanding
│   ├── 02_feature_engineering_advanced.ipynb # Advanced feature engineering
│   ├── 03_model_development.ipynb # Model training
│   └── 04_submission.ipynb        # Final predictions
├── src/                           # Source code modules
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature creation functions
│   ├── models.py                  # Model architectures
│   └── utils.py                   # Utility functions
├── models/                        # Trained model artifacts (excluded from git)
├── submissions/                   # Generated submission files (excluded from git)
└── README.md                      # This file
```

## Data Schema

### Input Features (Pre-Pass Data):
- **Identifiers**: `game_id`, `play_id`, `nfl_id`, `frame_id`
- **Player Info**: `player_name`, `position`, `height`, `weight`, `side`, `role`
- **Game Context**: `play_direction`, `absolute_yardline_number`
- **Tracking Data**: `x`, `y`, `s` (speed), `a` (acceleration), `dir`, `o` (orientation)
- **Target Info**: `ball_land_x`, `ball_land_y`, `num_frames_output`
- **Prediction Flag**: `player_to_predict` (Boolean)

### Output Targets:
- **Coordinates**: `x`, `y` positions during ball-in-air phase
- **Temporal**: Multiple frames per play (10 fps tracking)

## Key Insights

1. **Scale**: 816MB training data, ~285K rows per week
2. **Temporal Nature**: Variable-length sequences (ball-in-air duration varies)
3. **Multi-Agent**: Predict all 22 players simultaneously  
4. **Physics-Informed**: Players have momentum, acceleration constraints
5. **Context-Aware**: Ball landing location and target receiver known

## Feature Engineering Highlights

### Advanced Features Implemented:
- **Physics-Based**: Kinetic energy, momentum, angular dynamics
- **Temporal Analysis**: Fourier transforms, game phase detection
- **Tactical Intelligence**: Route patterns, formation analysis
- **Player Profiling**: Physical attributes, performance percentiles
- **Interaction Features**: Multi-dimensional feature combinations

### Quality Metrics:
- 179+ validated features across 16 categories
- Comprehensive quality assurance pipeline
- Professional documentation standards

## Modeling Approaches

### Planned Architectures:
1. **LSTM/GRU**: Sequential movement modeling
2. **Transformer**: Attention-based player interactions
3. **Physics-Based**: Kinematic constraints + neural networks
4. **Graph Neural Networks**: Player-to-player relationships
5. **Ensemble**: Combining multiple approaches

### Key Features:
- Player trajectory history
- Relative positions to ball landing spot
- Player role-specific behavior patterns
- Field position and game context
- Inter-player distances and angles

## Getting Started

### 1. Environment Setup (Required First Step)

**Windows (PowerShell/Command Prompt):**
```bash
# Run automated setup
.\setup_env.bat

# OR manual setup:
python -m venv nfl_env
nfl_env\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=nfl_env --display-name="NFL Big Data Bowl"
```

**macOS/Linux:**
```bash
# Run automated setup
chmod +x setup_env.sh
./setup_env.sh

# OR manual setup:
python3 -m venv nfl_env
source nfl_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=nfl_env --display-name="NFL Big Data Bowl"
```

### 2. Data Analysis Workflow

1. **Activate Environment**: `nfl_env\Scripts\activate` (Windows) or `source nfl_env/bin/activate` (macOS/Linux)
2. **Start Jupyter**: `jupyter notebook notebooks/`
3. **Select Kernel**: Choose "NFL Big Data Bowl" kernel in notebooks
4. **Begin Analysis**: Start with `01_data_exploration.ipynb`

### 3. Data Management

**Important**: Large data files are excluded from git repository:
- All CSV files in `data/train/` and `data/test/`
- Processed feature files in `data/processed/`
- Model artifacts in `models/`
- Generated submission files in `submissions/`

**To get started:**
1. Download competition data from Kaggle
2. Place training data in `data/train/`
3. Place test data files in `data/` root
4. Run feature engineering notebooks to generate processed datasets

## Competition Rules Summary

- **Team Size**: Up to 4 members
- **External Data**: Prohibited - only provided NFL data allowed
- **Code Sharing**: Private sharing forbidden; public sharing must be open-source
- **Winners**: Must open-source full solution with documentation
- **Hardware**: CPU/GPU notebooks ≤9 hours runtime limit

**Tech Stack**: Python, pandas, scikit-learn, TensorFlow/PyTorch, Jupyter, plotly

## Project Status

✅ **Completed**:
- Advanced feature engineering (179+ features)
- Professional documentation standards
- Quality assurance pipeline
- Repository structure and setup

🔄 **In Progress**:
- Model development and training
- Cross-validation framework
- Submission pipeline

📋 **Planned**:
- Live forecasting implementation
- Model ensemble optimization
- Final competition submission
- Processed feature files and model artifacts
- Generated submission files

Download competition data from Kaggle and place in appropriate directories.

## Competition Rules Summaryg the "ball in air" phase of NFL pass plays

**The Challenge**: When a quarterback releases the ball, predict where every player on the field will be positioned during each frame until the pass is caught or incomplete.

## Scoring Metric: Root Mean Squared Error (RMSE)

```
RMSE = √(Σ(predicted_x - actual_x)² + (predicted_y - actual_y)²) / n)
```

- Lower RMSE = Better predictions
- Evaluated on x,y coordinate accuracy for all players across all frames
- Live scoring on 2025 NFL season games (last 5 weeks)

## Competition Timeline

### Training Phase:
- **Start**: September 25, 2025
- **Team Merger Deadline**: November 26, 2025
- **Entry Deadline**: November 26, 2025  
- **Final Submission**: December 3, 2025

### Live Forecasting Phase:
- **Live Scoring**: December 4, 2025 - January 5, 2026
- **Results Published**: January 6, 2026

## Prizes
- **1st Place**: $25,000
- **2nd Place**: $15,000  
- **3rd Place**: $10,000
- **Total Pool**: $50,000

## Submission Rules
- **Daily Limit**: 5 submissions per day
- **Final Submissions**: 2 submissions count for final ranking
- **Format**: Kaggle Notebook (≤9 hours runtime)
- **Output**: `submission.csv` with format: `id,x,y`
- **ID Format**: `{game_id}_{play_id}_{nfl_id}_{frame_id}`

## Repository Structure

```
NFLBigDataBowl2026/
├── data/                           # Competition datasets (excluded from git)
│   ├── train/                      # 2023 NFL season training data
│   │   ├── input_2023_w01.csv     # Week 1 input features
│   │   ├── output_2023_w01.csv    # Week 1 target coordinates
│   │   └── ...                    # Weeks 2-18
│   ├── test_input.csv             # Test features for prediction
│   ├── test.csv                   # Test identifiers  
│   └── sample_submission.csv      # Submission format example
├── notebooks/                      # Jupyter notebooks
│   ├── 01_data_exploration.ipynb  # EDA and data understanding
│   ├── 02_feature_engineering_advanced.ipynb # Advanced feature engineering
│   ├── 03_model_development.ipynb # Model training
│   └── 04_submission.ipynb        # Final predictions
├── src/                           # Source code modules
│   ├── data_processing.py         # Data loading and preprocessing
│   ├── feature_engineering.py     # Feature creation functions
│   ├── models.py                  # Model architectures
│   └── utils.py                   # Utility functions
├── models/                        # Trained model artifacts
├── submissions/                   # Generated submission files
└── README.md                      # This file
```

## Data Schema

### Input Features (Pre-Pass Data):
- **Identifiers**: `game_id`, `play_id`, `nfl_id`, `frame_id`
- **Player Info**: `player_name`, `position`, `height`, `weight`, `side`, `role`
- **Game Context**: `play_direction`, `absolute_yardline_number`
- **Tracking Data**: `x`, `y`, `s` (speed), `a` (acceleration), `dir`, `o` (orientation)
- **Target Info**: `ball_land_x`, `ball_land_y`, `num_frames_output`
- **Prediction Flag**: `player_to_predict` (Boolean)

### Output Targets:
- **Coordinates**: `x`, `y` positions during ball-in-air phase
- **Temporal**: Multiple frames per play (10 fps tracking)

## Key Insights

1. **Scale**: 816MB training data, ~285K rows per week
2. **Temporal Nature**: Variable-length sequences (ball-in-air duration varies)
3. **Multi-Agent**: Predict all 22 players simultaneously  
4. **Physics-Informed**: Players have momentum, acceleration constraints
5. **Context-Aware**: Ball landing location and target receiver known

## Modeling Approaches

### Planned Architectures:
1. **LSTM/GRU**: Sequential movement modeling
2. **Transformer**: Attention-based player interactions
3. **Physics-Based**: Kinematic constraints + neural networks
4. **Graph Neural Networks**: Player-to-player relationships
5. **Ensemble**: Combining multiple approaches

### Key Features:
- Player trajectory history
- Relative positions to ball landing spot
- Player role-specific behavior patterns
- Field position and game context
- Inter-player distances and angles

## Getting Started

### 1. Environment Setup (Required First Step)

**Windows (PowerShell/Command Prompt):**
```bash
# Run automated setup
.\setup_env.bat

# OR manual setup:
python -m venv nfl_env
nfl_env\Scripts\activate
pip install -r requirements.txt
python -m ipykernel install --user --name=nfl_env --display-name="NFL Big Data Bowl"
```

**macOS/Linux:**
```bash
# Run automated setup
chmod +x setup_env.sh
./setup_env.sh

# OR manual setup:
python3 -m venv nfl_env
source nfl_env/bin/activate
pip install -r requirements.txt
python -m ipykernel install --user --name=nfl_env --display-name="NFL Big Data Bowl"
```

### 2. Data Analysis Workflow

1. **Activate Environment**: `nfl_env\Scripts\activate` (Windows) or `source nfl_env/bin/activate` (macOS/Linux)
2. **Start Jupyter**: `jupyter notebook notebooks/`
3. **Select Kernel**: Choose "NFL Big Data Bowl" kernel in notebooks
4. **Begin Analysis**: Start with `01_data_exploration.ipynb`

### 3. Project Structure Navigation

## Competition Rules Summary

- **Team Size**: Up to 4 members
- **External Data**: Prohibited - only provided NFL data allowed
- **Code Sharing**: Private sharing forbidden; public sharing must be open-source
- **Winners**: Must open-source full solution with documentation
- **Hardware**: CPU/GPU notebooks ≤9 hours runtime limit

**Tech Stack**: Python, pandas, scikit-learn, TensorFlow/PyTorch, Jupyter, plotly
