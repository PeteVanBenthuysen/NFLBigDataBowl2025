# NFL Big Data Bowl 2026 - Player Movement Prediction

A machine learning competition solution for predicting NFL player movement during pass plays. Using Next Gen Stats tracking data, this project develops models to forecast player positions from the moment the quarterback releases the ball until the pass is complete or incomplete.

## Competition Overview

- Predict x,y coordinates of all players during the "ball in air" phase of pass plays
- Training data: Full 2023 NFL season (18 weeks)
- Evaluation: Root Mean Squared Error on live 2025 season games
- Prize Pool: $50,000 ($25k/$15k/$10k for top 3)

## Dataset

- Pre-pass tracking data (player positions, velocities, orientations)
- Target receiver identification and ball landing locations
- 10 frames per second tracking resolution
- Historical data from 2023 season for training

## Key Features

- Data preprocessing and feature engineering pipelines
- Multiple model architectures (LSTM, Transformer, Physics-based)
- Player movement prediction algorithms
- Comprehensive evaluation and validation framework

**Tech Stack:** Python, pandas, scikit-learn, TensorFlow/PyTorch, Jupyter, plotly
