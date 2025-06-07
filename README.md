# IPL Match Predictor

A comprehensive Machine Learning application that predicts IPL match outcomes and provides detailed statistics.

## Features

- **Match Prediction**: Predicts the winner and win probabilities for both teams
- **Recent Form**: Shows last 5 matches for both teams
- **Head-to-Head Records**: Complete H2H statistics between the teams
- **Venue Statistics**: Team performance at specific venues
- **Overall Team Stats**: Complete team statistics (matches, wins, losses, win percentage)
- **Web Interface**: User-friendly web application
- **REST API**: API endpoints for programmatic access

## Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Dataset
- Save your IPL dataset as `ipl_dataset.csv` in the same directory
- Make sure your dataset has all the required columns mentioned in the code

### 3. Train the Model
```bash
python trainmodel.py
```

This will create the following pickle files:
- `ipl_model.pkl` - The trained Random Forest model
- `encoders.pkl` - Label encoders for categorical variables
- `winner_encoder.pkl` - Encoder for match winners
- `team_stats.pkl` - Overall team statistics
- `venue_stats.pkl` - Venue-wise team statistics
- `h2h_stats.pkl` - Head-to-head records between teams
- `processed_data.pkl` - Processed dataset for analysis

### 4. Run the Application
```bash
python app.py
```

The application will be available at `http://localhost:5000`

## API Endpoints

### Web Interface
- `GET /` - Main web application

### API Endpoints
- `GET /api/data` - Get available teams and venues
- `POST /predict` - Predict match outcome (JSON payload)
- `GET /api/predict/<team1>/<team2>/<venue>` - Direct prediction API

### API Usage Example

```python
import requests

# Predict match
response = requests.post('http://localhost:5000/predict', 
                        json={
                            'team1': 'Mumbai Indians',
                            'team2': 'Chennai Super Kings',
                            'venue': 'Wankhede Stadium'
                        })

result = response.json()
print(f"Predicted Winner: {result['prediction']['predicted_winner']}")
print(f"Win Probabilities: {result['prediction']['team1_win_probability']}% - {result['prediction']['team2_win_probability']}%")
```

## Model Details

- **Algorithm**: Random Forest Classifier
- **Features Used**:
  - Team encodings
  - Venue encoding
  - City encoding
  - Toss winner and decision
  - Year and month
  - Toss advantage features

## Output Structure

The prediction returns comprehensive data including:

```json
{
  "prediction": {
    "predicted_winner": "Team Name",
    "team1_win_probability": 65.5,
    "team2_win_probability": 34.5
  },
  "team_stats": {
    "Team1": {
      "matches_played": 150,
      "wins": 85,
      "losses": 65,
      "win_percentage": 56.7
    }
  },
  "recent_form": {
    "Team1": [
      {
        "date": "2024-05-15",
        "opponent": "Team2",
        "venue": "Stadium",
        "result": "Won"
      }
    ]
  },
  "head_to_head": {
    "total_matches": 25,
    "Team1_wins": 15,
    "Team2_wins": 10
  },
  "venue_stats": {
    "venue": "Stadium Name",
    "total_matches_at_venue": 100,
    "Team1": {
      "matches": 20,
      "wins": 12,
      "losses": 8,
      "win_percentage": 60.0
    }
  }
}
```

## Dataset Requirements

Your CSV file should contain these columns:
- id, season, city, date, match_type, player_of_match, venue
- team1, team2, toss_winner, toss_decision, winner
- result, result_margin, target_runs, target_overs, super_over
- method, umpire1, umpire2

## Troubleshooting

1. **FileNotFoundError**: Make sure `ipl_dataset.csv` exists in the project directory
2. **Model Loading Error**: Run `trainmodel.py` first to create pickle files
3. **Team/Venue Not Found**: The model uses teams and venues from training data only

## Future Enhancements

- Add more sophisticated features (player stats, weather data)
- Implement different ML algorithms
- Add real-time data integration
- Enhanced visualization with charts and graphs


-Implement weighted prediction based on:
Venue performance (40%)
Overall win percentage (30%)
Recent form (20%)
Head-to-head record (10%)