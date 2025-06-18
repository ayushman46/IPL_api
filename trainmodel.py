import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
import os

warnings.filterwarnings('ignore')

def standardize_team_names(df):
    """Standardize team names to handle variations"""
    team_mapping = {
        # Royal Challengers variations
        'Royal Challengers Bengaluru': 'Royal Challengers Bangalore',
        'Royal Challengers Bangalore': 'Royal Challengers Bangalore',
        'RCB': 'Royal Challengers Bangalore',
        
        # Delhi variations
        'Delhi Daredevils': 'Delhi Capitals',
        'DD': 'Delhi Capitals',
        'DC': 'Delhi Capitals',
        
        # Kings XI Punjab variations
        'Kings XI Punjab': 'Punjab Kings',
        'KXIP': 'Punjab Kings',
        'PK': 'Punjab Kings',
        'PBKS': 'Punjab Kings',
        
        # Chennai Super Kings variations
        'Chennai Super Kings': 'Chennai Super Kings',
        'CSK': 'Chennai Super Kings',
        
        # Mumbai Indians variations
        'Mumbai Indians': 'Mumbai Indians',
        'MI': 'Mumbai Indians',
        
        # Kolkata Knight Riders variations
        'Kolkata Knight Riders': 'Kolkata Knight Riders',
        'KKR': 'Kolkata Knight Riders',
        
        # Rajasthan Royals variations
        'Rajasthan Royals': 'Rajasthan Royals',
        'RR': 'Rajasthan Royals',
        
        # Sunrisers Hyderabad variations
        'Sunrisers Hyderabad': 'Sunrisers Hyderabad',
        'SRH': 'Sunrisers Hyderabad',
        
        # Other teams
        'Deccan Chargers': 'Deccan Chargers',
        'Gujarat Lions': 'Gujarat Lions',
        'Rising Pune Supergiant': 'Rising Pune Supergiant',
        'Rising Pune Supergiants': 'Rising Pune Supergiant',
        'Pune Warriors': 'Pune Warriors',
        'Kochi Tuskers Kerala': 'Kochi Tuskers Kerala',
        'Gujarat Titans': 'Gujarat Titans',
        'Lucknow Super Giants': 'Lucknow Super Giants'
    }
    
    # Apply team name standardization
    for col in ['team1', 'team2', 'toss_winner', 'winner']:
        if col in df.columns:
            df[col] = df[col].map(team_mapping).fillna(df[col])
    
    return df

def standardize_venue_names(df):
    """Standardize venue names to handle variations"""
    venue_mapping = {
        # Punjab Cricket Association Stadium, Mohali variations
        'Punjab Cricket Association IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association IS Bindra Stadium, Mohali, Chandigarh': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association Stadium, Mohali': 'Punjab Cricket Association Stadium, Mohali',
        'IS Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'I S Bindra Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'Punjab Cricket Association Stadium': 'Punjab Cricket Association Stadium, Mohali',
        'PCA Stadium': 'Punjab Cricket Association Stadium, Mohali',

        # DY Patil Stadium - single standardized name
        'Dr DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',
        'Dr. DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',
        'Dr DY Patil Sports Academy, Mumbai': 'DY Patil Stadium, Mumbai',
        'Dr. DY Patil Sports Academy, Mumbai': 'DY Patil Stadium, Mumbai',
        'DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',
        'D.Y.Patil Stadium': 'DY Patil Stadium, Mumbai',
        'DY Patil Stadium': 'DY Patil Stadium, Mumbai',


        # Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium variations
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium, Visakhapatnam': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',
        'Dr YS Rajasekhara Reddy ACA-VDCA Cricket Stadium': 'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium',

        # Rajiv Gandhi International Stadium variations
        'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Stadium, Uppal, Hyderabad': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Cricket Stadium': 'Rajiv Gandhi International Stadium',
        
        # M. Chinnaswamy Stadium variations
        'M Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'M. Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'M.Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'Chinnaswamy Stadium': 'M. Chinnaswamy Stadium',
        'M Chinnaswamy Stadium, Bengaluru': 'M. Chinnaswamy Stadium',
        'M. Chinnaswamy Stadium, Bangalore': 'M. Chinnaswamy Stadium',
        
        # Wankhede Stadium variations
        'Wankhede Stadium': 'Wankhede Stadium',
        'Wankhede Stadium, Mumbai': 'Wankhede Stadium',
        
        # Eden Gardens variations
        'Eden Gardens': 'Eden Gardens',
        'Eden Gardens, Kolkata': 'Eden Gardens',
        
        # Feroz Shah Kotla variations
        'Feroz Shah Kotla': 'Arun Jaitley Stadium',
        'Arun Jaitley Stadium': 'Arun Jaitley Stadium',
        'Feroz Shah Kotla Ground': 'Arun Jaitley Stadium',
        'Arun Jaitley Stadium, Delhi': 'Arun Jaitley Stadium',
        
        # MA Chidambaram Stadium variations
        'MA Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium, Chepauk',
        'MA Chidambaram Stadium, Chepauk, Chennai': 'MA Chidambaram Stadium, Chepauk',
        'M.A. Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
        'M.A.Chidambaram Stadium': 'MA Chidambaram Stadium, Chepauk',
        'Chepauk': 'MA Chidambaram Stadium, Chepauk',

        # Sawai Mansingh Stadium variations
        'Sawai Mansingh Stadium': 'Sawai Mansingh Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        
        # Other venues
        'Narendra Modi Stadium': 'Narendra Modi Stadium',
        'Sardar Patel Stadium': 'Narendra Modi Stadium',
        'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium',
        'MCA Stadium': 'Maharashtra Cricket Association Stadium',
        'Dr DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',        # Changed
        'Dr. DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',       # Changed
        'DY Patil Sports Academy': 'DY Patil Stadium, Mumbai',           # Added
        'D.Y.Patil Stadium': 'DY Patil Stadium, Mumbai',                # Added
        'Brabourne Stadium': 'Brabourne Stadium, Mumbai',
        'Brabourne Stadium, Mumbai': 'Brabourne Stadium, Mumbai',
        'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium',
        'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium',
        'Sheikh Zayed Stadium': 'Sheikh Zayed Stadium',
        'Himachal Pradesh Cricket Association Stadium': 'HPCA Stadium',
        'HPCA Stadium': 'HPCA Stadium',
        'Himachal Pradesh Cricket Association Stadium, Dharamsala': 'HPCA Stadium',
        'HPCA Stadium, Dharamsala': 'HPCA Stadium',
        'Holkar Cricket Stadium': 'Holkar Cricket Stadium',
        'Green Park': 'Green Park Stadium',
        'Barabati Stadium': 'Barabati Stadium',
        'JSCA International Stadium Complex': 'JSCA International Stadium',
        'Ekana Cricket Stadium': 'Ekana Cricket Stadium'
    }
    
    # Apply venue name standardization
    if 'venue' in df.columns:
        df['venue'] = df['venue'].map(venue_mapping).fillna(df['venue'])
    
    return df

def preprocess_data(df):
    """Preprocess the IPL dataset"""
    # Drop rows with missing winner
    df = df.dropna(subset=['winner'])
    
    # Standardize team and venue names
    df = standardize_team_names(df)
    df = standardize_venue_names(df)
    
    # Handle missing values
    if 'city' in df.columns:
        df['city'].fillna('Unknown', inplace=True)
    if 'venue' in df.columns:
        df['venue'].fillna('Unknown', inplace=True)
    if 'toss_winner' in df.columns:
        df['toss_winner'].fillna('Unknown', inplace=True)
    if 'toss_decision' in df.columns:
        df['toss_decision'].fillna('Unknown', inplace=True)
    if 'player_of_match' in df.columns:
        df['player_of_match'].fillna('Unknown', inplace=True)
    
    # Convert date to datetime
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
    else:
        # If no date column, create dummy year and month
        df['year'] = 2023
        df['month'] = 1
    
    return df

def create_team_stats(df):
    """Create team statistics"""
    team_stats = {}
    
    # Get all unique teams
    all_teams = set()
    if 'team1' in df.columns:
        all_teams.update(df['team1'].dropna().unique())
    if 'team2' in df.columns:
        all_teams.update(df['team2'].dropna().unique())
    
    for team in all_teams:
        # Total matches played
        matches_played = len(df[(df['team1'] == team) | (df['team2'] == team)])
        
        # Total wins
        wins = len(df[df['winner'] == team])
        
        # Win percentage
        win_pct = (wins / matches_played * 100) if matches_played > 0 else 0
        
        # Toss wins
        toss_wins = 0
        if 'toss_winner' in df.columns:
            toss_wins = len(df[df['toss_winner'] == team])
        toss_win_pct = (toss_wins / matches_played * 100) if matches_played > 0 else 0
        
        team_stats[team] = {
            'matches_played': matches_played,
            'wins': wins,
            'losses': matches_played - wins,
            'win_percentage': win_pct,
            'toss_wins': toss_wins,
            'toss_win_percentage': toss_win_pct
        }
    
    return team_stats

def create_venue_stats(df):
    """Create venue statistics"""
    venue_stats = {}
    
    if 'venue' not in df.columns:
        return venue_stats
    
    venues = df['venue'].dropna().unique()
    venues = [venue for venue in venues if venue != 'Unknown']
    
    for venue in venues:
        venue_matches = df[df['venue'] == venue]
        
        # Team-wise stats at this venue
        team_venue_stats = {}
        all_teams = set()
        if 'team1' in df.columns:
            all_teams.update(venue_matches['team1'].dropna().unique())
        if 'team2' in df.columns:
            all_teams.update(venue_matches['team2'].dropna().unique())
        
        for team in all_teams:
            team_matches_at_venue = venue_matches[
                (venue_matches['team1'] == team) | (venue_matches['team2'] == team)
            ]
            team_wins_at_venue = venue_matches[venue_matches['winner'] == team]
            
            matches_count = len(team_matches_at_venue)
            wins_count = len(team_wins_at_venue)
            
            team_venue_stats[team] = {
                'matches': matches_count,
                'wins': wins_count,
                'losses': matches_count - wins_count,
                'win_percentage': (wins_count / matches_count * 100) if matches_count > 0 else 0
            }
        
        venue_stats[venue] = {
            'total_matches': len(venue_matches),
            'team_stats': team_venue_stats
        }
    
    return venue_stats

def create_head_to_head_stats(df):
    """Create head-to-head statistics"""
    h2h_stats = {}
    
    # Get all unique teams
    all_teams = set()
    if 'team1' in df.columns:
        all_teams.update(df['team1'].dropna().unique())
    if 'team2' in df.columns:
        all_teams.update(df['team2'].dropna().unique())
    
    for team1 in all_teams:
        for team2 in all_teams:
            if team1 != team2:
                # Get matches between these two teams
                h2h_matches = df[
                    ((df['team1'] == team1) & (df['team2'] == team2)) |
                    ((df['team1'] == team2) & (df['team2'] == team1))
                ]
                
                if len(h2h_matches) > 0:
                    team1_wins = len(h2h_matches[h2h_matches['winner'] == team1])
                    team2_wins = len(h2h_matches[h2h_matches['winner'] == team2])
                    
                    # Get last 5 H2H matches
                    if 'date' in df.columns:
                        last_5_h2h = h2h_matches.sort_values('date', ascending=False).head(5)
                        last_5_records = last_5_h2h[['team1', 'team2', 'winner']].to_dict('records')
                        if 'date' in last_5_h2h.columns:
                            for i, record in enumerate(last_5_records):
                                record['date'] = last_5_h2h.iloc[i]['date']
                    else:
                        last_5_records = []
                    
                    h2h_stats[f"{team1}_vs_{team2}"] = {
                        'total_matches': len(h2h_matches),
                        'team1_wins': team1_wins,
                        'team2_wins': team2_wins,
                        'last_5_matches': last_5_records
                    }
    
    return h2h_stats

def get_recent_form(df, team, n=5):
    """Get recent form of a team (last n matches)"""
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
    
    if 'date' in df.columns:
        recent_matches = team_matches.sort_values('date', ascending=False).head(n)
    else:
        recent_matches = team_matches.tail(n)
    
    recent_form = []
    for _, match in recent_matches.iterrows():
        result = 'W' if match['winner'] == team else 'L'
        opponent = match['team2'] if match['team1'] == team else match['team1']
        venue = match.get('venue', 'Unknown')
        date_str = match['date'].strftime('%Y-%m-%d') if 'date' in match and pd.notna(match['date']) else 'Unknown'
        
        recent_form.append({
            'date': date_str,
            'opponent': opponent,
            'venue': venue,
            'result': result
        })
    
    return recent_form

def create_features(df):
    """Create features for ML model"""
    features_df = df.copy()
    
    # Initialize label encoders
    encoders = {}
    
    # Handle team1
    if 'team1' in features_df.columns:
        le_team1 = LabelEncoder()
        features_df['team1_encoded'] = le_team1.fit_transform(features_df['team1'].astype(str))
        encoders['team1'] = le_team1
    else:
        features_df['team1_encoded'] = 0
        encoders['team1'] = None
    
    # Handle team2
    if 'team2' in features_df.columns:
        le_team2 = LabelEncoder()
        features_df['team2_encoded'] = le_team2.fit_transform(features_df['team2'].astype(str))
        encoders['team2'] = le_team2
    else:
        features_df['team2_encoded'] = 0
        encoders['team2'] = None
    
    # Handle venue
    if 'venue' in features_df.columns:
        le_venue = LabelEncoder()
        features_df['venue_encoded'] = le_venue.fit_transform(features_df['venue'].astype(str))
        encoders['venue'] = le_venue
    else:
        features_df['venue_encoded'] = 0
        encoders['venue'] = None
    
    # Handle city
    if 'city' in features_df.columns:
        le_city = LabelEncoder()
        features_df['city_encoded'] = le_city.fit_transform(features_df['city'].astype(str))
        encoders['city'] = le_city
    else:
        features_df['city_encoded'] = 0
        encoders['city'] = None
    
    # Handle toss_winner
    if 'toss_winner' in features_df.columns:
        le_toss_winner = LabelEncoder()
        features_df['toss_winner_encoded'] = le_toss_winner.fit_transform(features_df['toss_winner'].astype(str))
        encoders['toss_winner'] = le_toss_winner
    else:
        features_df['toss_winner_encoded'] = 0
        encoders['toss_winner'] = None
    
    # Handle toss_decision
    if 'toss_decision' in features_df.columns:
        le_toss_decision = LabelEncoder()
        features_df['toss_decision_encoded'] = le_toss_decision.fit_transform(features_df['toss_decision'].astype(str))
        encoders['toss_decision'] = le_toss_decision
    else:
        features_df['toss_decision_encoded'] = 0
        encoders['toss_decision'] = None
    
    # Create toss advantage features
    if 'team1' in features_df.columns and 'toss_winner' in features_df.columns:
        features_df['team1_won_toss'] = (features_df['team1'] == features_df['toss_winner']).astype(int)
        features_df['team2_won_toss'] = (features_df['team2'] == features_df['toss_winner']).astype(int)
    else:
        features_df['team1_won_toss'] = 0
        features_df['team2_won_toss'] = 0
    
    return features_df, encoders

def train_model(df):
    """Train the prediction model"""
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    print("Creating features...")
    features_df, encoders = create_features(df)
    
    # Select features for training
    feature_columns = [
        'team1_encoded', 'team2_encoded', 'venue_encoded', 'city_encoded',
        'toss_winner_encoded', 'toss_decision_encoded', 'year', 'month',
        'team1_won_toss', 'team2_won_toss'
    ]
    
    X = features_df[feature_columns]
    
    # Create target (winner encoded)
    le_winner = LabelEncoder()
    y = le_winner.fit_transform(features_df['winner'])
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    print("Training model...")
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Model Accuracy: {accuracy:.2f}")
    
    # Create and save all required data
    print("Creating team statistics...")
    team_stats = create_team_stats(df)
    
    print("Creating venue statistics...")
    venue_stats = create_venue_stats(df)
    
    print("Creating head-to-head statistics...")
    h2h_stats = create_head_to_head_stats(df)
    
    # Save all components
    print("Saving model and data...")
    
    # Save model
    with open('ipl_model.pkl', 'wb') as f:
        pickle.dump(model, f)
        print("✓ Saved: ipl_model.pkl")
    
    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
        print("✓ Saved: encoders.pkl")
    
    # Save winner encoder
    with open('winner_encoder.pkl', 'wb') as f:
        pickle.dump(le_winner, f)
        print("✓ Saved: winner_encoder.pkl")
    
    # Save statistics
    with open('team_stats.pkl', 'wb') as f:
        pickle.dump(team_stats, f)
        print("✓ Saved: team_stats.pkl")
    
    with open('venue_stats.pkl', 'wb') as f:
        pickle.dump(venue_stats, f)
        print("✓ Saved: venue_stats.pkl")
    
    with open('h2h_stats.pkl', 'wb') as f:
        pickle.dump(h2h_stats, f)
        print("✓ Saved: h2h_stats.pkl")
    
    # Save processed dataframe for recent form analysis
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)
        print("✓ Saved: processed_data.pkl")
    
    print("\n" + "="*50)
    print("SUCCESS! All files saved successfully!")
    print("="*50)
    print("Files created:")
    print("- ipl_model.pkl")
    print("- encoders.pkl") 
    print("- winner_encoder.pkl")
    print("- team_stats.pkl")
    print("- venue_stats.pkl")
    print("- h2h_stats.pkl")
    print("- processed_data.pkl")
    
    # Display some statistics
    print(f"\nDataset Summary:")
    print(f"- Total matches processed: {len(df)}")
    print(f"- Unique teams: {len(team_stats)}")
    print(f"- Unique venues: {len(venue_stats)}")
    print(f"- Model accuracy: {accuracy:.2%}")
    
    return model, encoders, le_winner, team_stats, venue_stats, h2h_stats

# Helper functions (moved outside of main execution)
def get_recent_form_score(team, processed_data, n=5):
    """Calculate score based on recent form"""
    team_matches = processed_data[
        (processed_data['team1'] == team) | 
        (processed_data['team2'] == team)
    ]
    
    if 'date' in processed_data.columns:
        team_matches = team_matches.sort_values('date', ascending=False).head(n)
    else:
        team_matches = team_matches.tail(n)
    
    wins = sum(1 for _, match in team_matches.iterrows() 
              if match['winner'] == team)
    return (wins / len(team_matches) * 100) if len(team_matches) > 0 else 50

def get_h2h_score(team1, team2, h2h_stats):
    """Calculate head to head winning percentage"""
    h2h_key = f"{team1}_vs_{team2}"
    if h2h_key in h2h_stats:
        total = h2h_stats[h2h_key]['total_matches']
        wins = h2h_stats[h2h_key]['team1_wins']
        return (wins / total * 100) if total > 0 else 50
    return 50  # Default to 50% if no history

def predict_match_outcome(row, team_stats, venue_stats, h2h_stats, processed_data):
    """Enhanced prediction logic based on multiple factors"""
    team1 = row['team1']
    team2 = row['team2']
    venue = row.get('venue', 'Unknown')
    
    # Get venue stats
    venue_data = venue_stats.get(venue, {'team_stats': {}})
    team1_venue = venue_data['team_stats'].get(team1, {'win_percentage': 0})
    team2_venue = venue_data['team_stats'].get(team2, {'win_percentage': 0})
    
    # Get overall stats
    team1_overall = team_stats.get(team1, {'win_percentage': 0})
    team2_overall = team_stats.get(team2, {'win_percentage': 0})
    
    # Calculate score based on all factors
    team1_score = (
        team1_venue['win_percentage'] * 0.4 +  # Venue performance (40% weight)
        team1_overall['win_percentage'] * 0.3 +  # Overall performance (30% weight)
        get_recent_form_score(team1, processed_data) * 0.2 +    # Recent form (20% weight)
        get_h2h_score(team1, team2, h2h_stats) * 0.1       # Head to head (10% weight)
    )
    
    team2_score = (
        team2_venue['win_percentage'] * 0.4 +
        team2_overall['win_percentage'] * 0.3 +
        get_recent_form_score(team2, processed_data) * 0.2 +
        get_h2h_score(team2, team1, h2h_stats) * 0.1
    )
    
    return team1 if team1_score > team2_score else team2

if __name__ == "__main__":
    print("\n=== IPL Match Predictor Model Training ===\n")
    
    try:
        # Check if matches.csv exists
        if not os.path.exists('matches.csv'):
            print("Error: matches.csv not found!")
            print("Current directory:", os.getcwd())
            exit(1)
            
        # Load dataset with minimal required columns
        print("Loading matches.csv...")
        df = pd.read_csv('matches.csv')
        
        # Check minimum required columns
        required_columns = ['team1', 'team2', 'venue', 'winner']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            print(f"Error: Missing required columns: {missing_columns}")
            print("Available columns:", df.columns.tolist())
            exit(1)
            
        print(f"Successfully loaded {len(df)} matches")
        
        # Train model and save files
        model, encoders, winner_encoder, team_stats, venue_stats, h2h_stats = train_model(df)
        
        # Verify all files were created
        required_files = [
            'ipl_model.pkl', 'encoders.pkl', 'winner_encoder.pkl',
            'team_stats.pkl', 'venue_stats.pkl', 'h2h_stats.pkl',
            'processed_data.pkl'
        ]
        
        missing_files = [f for f in required_files if not os.path.exists(f)]
        if missing_files:
            print(f"\nWarning: Some files were not created: {missing_files}")
        else:
            print("\nAll required files created successfully!")
            print("You can now run app.py")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        import traceback
        traceback.print_exc()