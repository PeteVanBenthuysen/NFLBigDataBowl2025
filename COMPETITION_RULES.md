# NFL Big Data Bowl 2026 - Complete Competition Rules

## Competition Specifics

### Objective
Predict player movement during the "ball in air" phase of pass plays using pre-pass tracking data, target receiver information, and ball landing locations.

### Data Details
- **Tracking Resolution**: 10 frames per second
- **Training Data**: Complete 2023 NFL season (18 weeks)
- **Excluded Plays**: Quick passes (<0.5s), deflected passes, throwaway passes
- **Live Evaluation**: Last 5 weeks of 2025 NFL season

### Evaluation Metric
**Root Mean Squared Error (RMSE)** between predicted and observed x,y coordinates.

## Timeline

### Training Phase
- **Competition Start**: September 25, 2025
- **Team Merger Deadline**: November 26, 2025 (11:59 PM UTC)
- **Entry Deadline**: November 26, 2025 (11:59 PM UTC)
- **Final Submission Deadline**: December 3, 2025 (11:59 PM UTC)

### Forecasting Phase
- **Live Scoring Period**: December 4, 2025 - January 5, 2026
- **Leaderboard Updates**: After each week's NFL games
- **First Scored Week**: December 4, 7, 8 games
- **Final Regular Season Game**: January 4, 2026
- **Competition End**: January 6, 2026

## Submission Requirements

### Technical Constraints
- **Submission Method**: Kaggle Notebooks only
- **Runtime Limit**: ≤9 hours (CPU or GPU)
- **Internet Access**: Disabled during execution
- **External Data**: Prohibited
- **Output File**: Must be named `submission.csv`

### Submission Format
```csv
id,x,y
{game_id}_{play_id}_{nfl_id}_{frame_id},x_coordinate,y_coordinate
```

### Submission Limits
- **Daily Submissions**: 5 per day maximum
- **Final Submissions**: Select 2 for judging
- **Tie-Breaking**: Earlier submission wins

## Team Rules

### Team Structure
- **Maximum Size**: 4 members
- **Team Mergers**: Allowed if total submissions ≤ maximum allowed
- **Registration**: All members must register individually before joining

### Code Sharing Rules
- **Private Sharing**: Forbidden outside team
- **Public Sharing**: Allowed on Kaggle forums/notebooks (must be open-source)
- **Winner Requirements**: Full code + documentation (training, inference, environment)

## Data Usage Rules

### Permitted Use
- Competition participation
- Research and education
- Non-commercial purposes only

### Prohibited Actions
- Data redistribution
- Unauthorized access or leaks
- Commercial use
- External data supplementation

### Security Requirements
- Keep data secure at all times
- Report any data leaks immediately
- No sharing outside authorized channels

## Eligibility Requirements

### Basic Requirements
- Registered Kaggle user
- Age 18+ or legal majority in your country
- Under 18 can participate with parental consent (but cannot win prizes)

### Employment Restrictions
- NFL, Kaggle, or affiliate employees may participate but cannot win prizes
- Must have employer permission if entering on behalf of organization

## Winner Obligations

### Code Requirements
- **Full Solution**: Training, inference, environment setup
- **Documentation**: Methodology, architecture, preprocessing, hyperparameters
- **Open Source License**: OSI-approved license allowing commercial use
- **Reproducibility**: Sufficient detail to reproduce results

### Legal Requirements
- Sign acceptance documents within 1 week of notification
- Complete tax forms (US winners receive IRS 1099)
- Sign liability releases
- Allow use of name/likeness for publicity

### Prize Distribution
- Split equally among team members unless specified otherwise
- Tax responsibility belongs to winners
- Must respond to winner notification within 1 week

## Disqualification Grounds

### Automatic Disqualification
- Multiple Kaggle accounts
- Private code/data sharing outside team
- External data use
- Cheating, fraud, or harassment
- Deadline violations
- Leaderboard manipulation

### Consequences
- Removal from leaderboard
- Loss of medals/points eligibility
- Potential ban from future competitions

## Legal Framework

### Governing Law
- New York law governs competition
- Disputes handled in NYC courts

### Liability
- No employment relationship created
- Participants indemnify NFL/Kaggle from claims
- Organizers not responsible for technical issues

### Modifications
- NFL/Kaggle may cancel, modify, or suspend competition if necessary
- Rule changes communicated through official channels
- Decisions are final

## Privacy and Data Handling

### Information Collection
- Name, email, and registration data collected
- Used for competition administration
- May be transferred internationally

### Data Retention
- Competition data retained per NFL requirements
- Personal data handled per platform privacy policies

## Key Reminders

### Critical Rules
1. **One Account Rule**: Strictly enforced
2. **No External Data**: Only provided NFL dataset allowed
3. **Open Source Requirement**: Winners must open-source solutions
4. **Team Size Limit**: Maximum 4 members
5. **Submission Limits**: 5/day, 2 final selections
6. **Code Sharing**: Private sharing forbidden
7. **Security**: Keep dataset secure at all times

### Success Factors
- Focus on RMSE optimization
- Understand temporal dynamics of player movement
- Leverage physics constraints and player behavior patterns
- Build robust validation framework
- Prepare for live evaluation on unseen 2025 data