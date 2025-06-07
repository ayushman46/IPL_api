import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import pickle
import warnings
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
        'MA Chidambaram Stadium': 'MA Chidambaram Stadium',
        'M. A. Chidambaram Stadium': 'MA Chidambaram Stadium',
        'M.A. Chidambaram Stadium': 'MA Chidambaram Stadium',
        'Chepauk Stadium': 'MA Chidambaram Stadium',
        'MA Chidambaram Stadium, Chepauk': 'MA Chidambaram Stadium',
        
        # Rajiv Gandhi International Stadium variations
        'Rajiv Gandhi International Stadium': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi Intl. Cricket Stadium': 'Rajiv Gandhi International Stadium',
        'Rajiv Gandhi International Cricket Stadium': 'Rajiv Gandhi International Stadium',
        
        # Sawai Mansingh Stadium variations
        'Sawai Mansingh Stadium': 'Sawai Mansingh Stadium',
        'Sawai Mansingh Stadium, Jaipur': 'Sawai Mansingh Stadium',
        
        # IS Bindra Stadium variations
        'IS Bindra Stadium': 'IS Bindra Stadium',
        'I S Bindra Stadium': 'IS Bindra Stadium',
        'Punjab Cricket Association Stadium': 'IS Bindra Stadium',
        'PCA Stadium': 'IS Bindra Stadium',
        
        # Other venues
        'Narendra Modi Stadium': 'Narendra Modi Stadium',
        'Sardar Patel Stadium': 'Narendra Modi Stadium',
        'Maharashtra Cricket Association Stadium': 'Maharashtra Cricket Association Stadium',
        'MCA Stadium': 'Maharashtra Cricket Association Stadium',
        'Dr DY Patil Sports Academy': 'Dr. DY Patil Sports Academy',
        'Dr. DY Patil Sports Academy': 'Dr. DY Patil Sports Academy',
        'Brabourne Stadium': 'Brabourne Stadium',
        'Sharjah Cricket Stadium': 'Sharjah Cricket Stadium',
        'Dubai International Cricket Stadium': 'Dubai International Cricket Stadium',
        'Sheikh Zayed Stadium': 'Sheikh Zayed Stadium',
        'Himachal Pradesh Cricket Association Stadium': 'HPCA Stadium',
        'HPCA Stadium': 'HPCA Stadium',
        'Holkar Cricket Stadium': 'Holkar Cricket Stadium',
        'Green Park': 'Green Park Stadium',
        'Barabati Stadium': 'Barabati Stadium',
        'JSCA International Stadium Complex': 'JSCA International Stadium',
        'Ekana Cricket Stadium': 'Ekana Cricket Stadium'
    }
    
    # Apply venue name standardization
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
    df['city'].fillna('Unknown', inplace=True)
    df['venue'].fillna('Unknown', inplace=True)
    df['toss_winner'].fillna('Unknown', inplace=True)
    df['toss_decision'].fillna('Unknown', inplace=True)
    df['player_of_match'].fillna('Unknown', inplace=True)
    
    # Convert date to datetime
    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    
    return df

def create_team_stats(df):
    """Create team statistics"""
    team_stats = {}
    
    # Get all unique teams
    all_teams = set(df['team1'].unique()) | set(df['team2'].unique())
    all_teams = [team for team in all_teams if pd.notna(team)]
    
    for team in all_teams:
        # Total matches played
        matches_played = len(df[(df['team1'] == team) | (df['team2'] == team)])
        
        # Total wins
        wins = len(df[df['winner'] == team])
        
        # Win percentage
        win_pct = (wins / matches_played * 100) if matches_played > 0 else 0
        
        # Toss wins
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
    
    venues = df['venue'].unique()
    venues = [venue for venue in venues if pd.notna(venue) and venue != 'Unknown']
    
    for venue in venues:
        venue_matches = df[df['venue'] == venue]
        
        # Team-wise stats at this venue
        team_venue_stats = {}
        all_teams = set(venue_matches['team1'].unique()) | set(venue_matches['team2'].unique())
        
        for team in all_teams:
            if pd.notna(team):
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
    all_teams = set(df['team1'].unique()) | set(df['team2'].unique())
    all_teams = [team for team in all_teams if pd.notna(team)]
    
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
                    last_5_h2h = h2h_matches.sort_values('date', ascending=False).head(5)
                    
                    h2h_stats[f"{team1}_vs_{team2}"] = {
                        'total_matches': len(h2h_matches),
                        'team1_wins': team1_wins,
                        'team2_wins': team2_wins,
                        'last_5_matches': last_5_h2h[['date', 'team1', 'team2', 'winner']].to_dict('records')
                    }
    
    return h2h_stats

def get_recent_form(df, team, n=5):
    """Get recent form of a team (last n matches)"""
    team_matches = df[(df['team1'] == team) | (df['team2'] == team)]
    recent_matches = team_matches.sort_values('date', ascending=False).head(n)
    
    recent_form = []
    for _, match in recent_matches.iterrows():
        result = 'W' if match['winner'] == team else 'L'
        opponent = match['team2'] if match['team1'] == team else match['team1']
        recent_form.append({
            'date': match['date'].strftime('%Y-%m-%d'),
            'opponent': opponent,
            'venue': match['venue'],
            'result': result
        })
    
    return recent_form

def create_features(df):
    """Create features for ML model"""
    features_df = df.copy()
    
    # Label encode categorical variables
    le_team1 = LabelEncoder()
    le_team2 = LabelEncoder()
    le_venue = LabelEncoder()
    le_city = LabelEncoder()
    le_toss_winner = LabelEncoder()
    le_toss_decision = LabelEncoder()
    
    # Fit and transform
    features_df['team1_encoded'] = le_team1.fit_transform(features_df['team1'].astype(str))
    features_df['team2_encoded'] = le_team2.fit_transform(features_df['team2'].astype(str))
    features_df['venue_encoded'] = le_venue.fit_transform(features_df['venue'].astype(str))
    features_df['city_encoded'] = le_city.fit_transform(features_df['city'].astype(str))
    features_df['toss_winner_encoded'] = le_toss_winner.fit_transform(features_df['toss_winner'].astype(str))
    features_df['toss_decision_encoded'] = le_toss_decision.fit_transform(features_df['toss_decision'].astype(str))
    
    # Create toss advantage feature
    features_df['team1_won_toss'] = (features_df['team1'] == features_df['toss_winner']).astype(int)
    features_df['team2_won_toss'] = (features_df['team2'] == features_df['toss_winner']).astype(int)
    
    # Save label encoders
    encoders = {
        'team1': le_team1,
        'team2': le_team2,
        'venue': le_venue,
        'city': le_city,
        'toss_winner': le_toss_winner,
        'toss_decision': le_toss_decision
    }
    
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
    
    # Save encoders
    with open('encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)
    
    # Save winner encoder
    with open('winner_encoder.pkl', 'wb') as f:
        pickle.dump(le_winner, f)
    
    # Save statistics
    with open('team_stats.pkl', 'wb') as f:
        pickle.dump(team_stats, f)
    
    with open('venue_stats.pkl', 'wb') as f:
        pickle.dump(venue_stats, f)
    
    with open('h2h_stats.pkl', 'wb') as f:
        pickle.dump(h2h_stats, f)
    
    # Save processed dataframe for recent form analysis
    with open('processed_data.pkl', 'wb') as f:
        pickle.dump(df, f)
    
    print("All files saved successfully!")
    print("Files created:")
    print("- ipl_model.pkl")
    print("- encoders.pkl") 
    print("- winner_encoder.pkl")
    print("- team_stats.pkl")
    print("- venue_stats.pkl")
    print("- h2h_stats.pkl")
    print("- processed_data.pkl")
    
    return model, encoders, le_winner, team_stats, venue_stats, h2h_stats

def predict_match_outcome(row):
    """Enhanced prediction logic based on multiple factors"""
    team1 = row['team1']
    team2 = row['team2']
    venue = row['venue']
    
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
        get_recent_form_score(team1) * 0.2 +    # Recent form (20% weight)
        get_h2h_score(team1, team2) * 0.1       # Head to head (10% weight)
    )
    
    team2_score = (
        team2_venue['win_percentage'] * 0.4 +
        team2_overall['win_percentage'] * 0.3 +
        get_recent_form_score(team2) * 0.2 +
        get_h2h_score(team2, team1) * 0.1
    )
    
    return team1 if team1_score > team2_score else team2

def get_recent_form_score(team, n=5):
    """Calculate score based on recent form"""
    team_matches = processed_data[
        (processed_data['team1'] == team) | 
        (processed_data['team2'] == team)
    ].sort_values('date', ascending=False).head(n)
    
    wins = sum(1 for _, match in team_matches.iterrows() 
              if match['winner'] == team)
    return (wins / n) * 100

def get_h2h_score(team1, team2):
    """Calculate head to head winning percentage"""
    h2h_key = f"{team1}_vs_{team2}"
    if h2h_key in h2h_stats:
        total = h2h_stats[h2h_key]['total_matches']
        wins = h2h_stats[h2h_key]['team1_wins']
        return (wins / total * 100) if total > 0 else 50
    return 50  # Default to 50% if no history

if __name__ == "__main__":
    # Load your IPL dataset
    print("Loading data...")
    try:
        # Try to load the matches.csv file
        df = pd.read_csv('matches.csv')
        print(f"Dataset loaded with {len(df)} rows")
        
        # Show team name standardization
        print("\nTeam names before standardization:")
        print("Team1 unique values:", sorted(df['team1'].dropna().unique()))
        print("Team2 unique values:", sorted(df['team2'].dropna().unique()))
        
        print("\nVenue names before standardization:")
        print("Venue unique values:", sorted(df['venue'].dropna().unique()))
        
        # Train model and create all required files
        train_model(df)
        
        # Show results after standardization
        try:
            processed_df = preprocess_data(df)  # Use the original dataframe instead of loading again
            
            print("\n" + "="*50)
            print("STANDARDIZATION RESULTS:")
            print("="*50)
            print("\nTeam names after standardization:")
            all_teams = set(processed_df['team1'].dropna().unique()) | set(processed_df['team2'].dropna().unique())
            print("All teams:", sorted(all_teams))
            
            print("\nVenue names after standardization:")
            print("All venues:", sorted(processed_df['venue'].dropna().unique()))
            
        except Exception as e:
            print(f"Error during standardization: {e}")
        
    except FileNotFoundError:
        print("Error: Please make sure 'matches.csv' exists in the current directory")
        print("The file should be named 'matches.csv' and contain the IPL match data")
    except Exception as e:
        print(f"Error loading dataset: {e}")