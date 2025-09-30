# NFL Big Data Bowl 2026 - Complete Competition Summary

## SCORING METRIC: Root Mean Squared Error (RMSE)

**Formula**: 
```
RMSE = √(Σ[(predicted_x - actual_x)² + (predicted_y - actual_y)²] / n)
```

**Key Points**:
- Lower RMSE = Better performance
- Evaluated on x,y coordinate accuracy across ALL players and ALL frames
- Live evaluation on unseen 2025 NFL games (last 5 weeks of season)
- Final ranking based on private leaderboard (live game results)

---

## COMPETITION UNDERSTANDING

### **The Core Challenge**
Predict where every NFL player will be positioned during the "ball in air" phase of pass plays.

**Timeline**: From QB ball release → until pass caught/incomplete
**Frequency**: 10 frames per second tracking data
**Players**: All 22 players on field (11 offense, 11 defense)

### **What We Have**:
- Pre-pass tracking data (positions, velocities, accelerations, orientations)  
- Target receiver identification  
- Ball landing location  
- Player attributes (height, weight, position, role)  
- Game context (field position, play direction)  

### **What We Predict**:
- x,y coordinates for each player during ball-in-air frames  
- Variable sequence lengths (different pass durations)  
- Only players marked with `player_to_predict=True`

---

## DATA STRUCTURE ANALYSIS

### **Training Data Scale**:
- **18 weeks** of 2023 NFL season
- **816MB** total size
- **~285K rows** per week
- **36 files** total (18 input + 18 output)

### **Input Schema** (`input_2023_wXX.csv`):
```
game_id, play_id, nfl_id, frame_id          # Identifiers
player_to_predict                           # Boolean flag
play_direction, absolute_yardline_number     # Game context  
player_name, height, weight, position       # Player info
player_side, player_role                     # Team/role info
x, y, s, a, dir, o                          # Tracking data
num_frames_output                           # Sequence length
ball_land_x, ball_land_y                    # Target info
```

### **Output Schema** (`output_2023_wXX.csv`):
```
game_id, play_id, nfl_id, frame_id          # Identifiers
x, y                                        # Target coordinates
```

### **Test Data**:
- `test_input.csv`: 49,755 rows of features
- `test.csv`: 5,839 prediction targets
- Live scoring on 2025 season games

---

## REPOSITORY ORGANIZATION

```
NFLBigDataBowl2025/
├── data/                    # Competition datasets
│   ├── train/              # 2023 training data (18 weeks)
│   ├── test_input.csv      # Test features  
│   ├── test.csv            # Test targets
│   └── sample_submission.csv # Submission format
├── notebooks/              # Analysis & development
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_development.ipynb
│   └── 04_submission.ipynb
├── src/                    # Source code modules
│   ├── data_processing.py  # Data loading utilities
│   ├── feature_engineering.py # Feature creation
│   ├── models.py           # Model architectures
│   └── utils.py            # Helper functions
├── models/                 # Trained artifacts
├── submissions/            # Generated submissions
├── README.md               # Project overview
└── COMPETITION_RULES.md    # Complete rules
```

---

## CRITICAL TIMELINE

| Date | Milestone | Status |
|------|-----------|--------|
| Sept 25, 2025 | Competition Start | DONE |
| Nov 26, 2025 | Team Merger Deadline | 58 days |
| Nov 26, 2025 | Entry Deadline | 58 days |
| Dec 3, 2025 | Final Submission | 65 days |
| Dec 4-Jan 5, 2026 | Live Scoring | Future |
| Jan 6, 2026 | Results Published | Future |

---

## STRATEGIC APPROACH

### **Phase 1: Data Understanding** (Week 1-2)
- Data exploration and visualization
- Play-by-play analysis  
- Player movement patterns
- Target receiver behavior

### **Phase 2: Feature Engineering** (Week 3-4)
- Temporal features (velocity, acceleration changes)
- Relative features (distance to ball, other players)
- Physics features (momentum, trajectory)
- Context features (field position, game situation)

### **Phase 3: Model Development** (Week 5-8)
- **Baseline**: Linear regression, physics-based
- **Sequential**: LSTM, GRU for temporal patterns
- **Attention**: Transformer for player interactions
- **Graph**: GNN for spatial relationships
- **Ensemble**: Combine multiple approaches

### **Phase 4: Validation & Tuning** (Week 9-10)
- Cross-validation on historical data
- Live validation framework
- Hyperparameter optimization
- Ensemble weighting

---

## SUCCESS METRICS

### **Technical Targets**:
- **RMSE < 2.0 yards**: Competitive baseline
- **RMSE < 1.5 yards**: Strong performance  
- **RMSE < 1.0 yards**: Top-tier result

### **Key Success Factors**:
1. **Physics-Informed**: Respect movement constraints
2. **Multi-Scale**: Capture both individual and team dynamics
3. **Robust**: Handle variable sequence lengths
4. **Live-Ready**: Perform on unseen 2025 data

---

## MODELING HYPOTHESES

### **Core Assumptions**:
1. Players move toward strategic positions (not random)
2. Movement influenced by ball trajectory and other players
3. Position-specific behavior patterns exist
4. Physics constraints limit possible movements

### **Key Features**:
- **Trajectory**: Historical movement patterns
- **Intent**: Direction toward ball/coverage zones  
- **Interaction**: Distances and relative positions
- **Context**: Field position, down & distance
- **Physics**: Acceleration/speed constraints

---

## COMPETITION ADVANTAGES

### **What We Know**:
- Ball landing location (huge advantage!)  
- Target receiver identity  
- Pre-pass player states  
- Full historical season data  

### **Challenges**:
- Variable sequence lengths  
- Multi-agent prediction complexity  
- Live evaluation (distribution shift)  
- Physics constraint handling  

---

## NEXT STEPS

1. **Immediate**: Create EDA notebook
2. **Week 1**: Understand data patterns and player behavior
3. **Week 2**: Build baseline models and evaluation framework
4. **Week 3**: Advanced feature engineering
5. **Week 4**: Deep learning model development

**Ready to dominate this competition!**