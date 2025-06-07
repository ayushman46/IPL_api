from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

def standardize_team_names_single(team_name):
    """Standardize a single team name"""
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
    
    return team_mapping.get(team_name, team_name)

def standardize_venue_names_single(venue_name):
    """Standardize a single venue name"""
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
    
    return venue_mapping.get(venue_name, venue_name)

# Load all required models and data
try:
    with open('ipl_model.pkl', 'rb') as f:
        model = pickle.load(f)
    
    with open('encoders.pkl', 'rb') as f:
        encoders = pickle.load(f)
    
    with open('winner_encoder.pkl', 'rb') as f:
        winner_encoder = pickle.load(f)
    
    with open('team_stats.pkl', 'rb') as f:
        team_stats = pickle.load(f)
    
    with open('venue_stats.pkl', 'rb') as f:
        venue_stats = pickle.load(f)
    
    with open('h2h_stats.pkl', 'rb') as f:
        h2h_stats = pickle.load(f)
    
    with open('processed_data.pkl', 'rb') as f:
        processed_data = pickle.load(f)
    
    print("All models and data loaded successfully!")
    
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please run trainmodel.py first to create the required pickle files")

def get_recent_form(team, n=5):
    """Get recent form of a team"""
    team_matches = processed_data[(processed_data['team1'] == team) | (processed_data['team2'] == team)]
    recent_matches = team_matches.sort_values('date', ascending=False).head(n)
    
    recent_form = []
    for _, match in recent_matches.iterrows():
        result = 'Won' if match['winner'] == team else 'Lost'
        opponent = match['team2'] if match['team1'] == team else match['team1']
        recent_form.append({
            'date': match['date'].strftime('%Y-%m-%d'),
            'opponent': opponent,
            'venue': match['venue'],
            'result': result
        })
    
    return recent_form

def get_head_to_head(team1, team2, n=5):
    """Get head-to-head record between two teams"""
    h2h_key1 = f"{team1}_vs_{team2}"
    h2h_key2 = f"{team2}_vs_{team1}"
    
    h2h_data = None
    if h2h_key1 in h2h_stats:
        h2h_data = h2h_stats[h2h_key1]
    elif h2h_key2 in h2h_stats:
        h2h_data = h2h_stats[h2h_key2]
        # Swap the wins for correct perspective
        temp = h2h_data['team1_wins']
        h2h_data['team1_wins'] = h2h_data['team2_wins']
        h2h_data['team2_wins'] = temp
    
    if h2h_data:
        return {
            'total_matches': h2h_data['total_matches'],
            f'{team1}_wins': h2h_data['team1_wins'],
            f'{team2}_wins': h2h_data['team2_wins'],
            'recent_matches': h2h_data['last_5_matches'][:n]
        }
    else:
        return {
            'total_matches': 0,
            f'{team1}_wins': 0,
            f'{team2}_wins': 0,
            'recent_matches': []
        }

def get_venue_stats_for_teams(team1, team2, venue):
    """Get venue statistics for both teams"""
    venue_data = venue_stats.get(venue, {'total_matches': 0, 'team_stats': {}})
    
    team1_venue_stats = venue_data['team_stats'].get(team1, {
        'matches': 0, 
        'wins': 0,
        'losses': 0,
        'win_percentage': 0
    })
    
    team2_venue_stats = venue_data['team_stats'].get(team2, {
        'matches': 0, 
        'wins': 0,
        'losses': 0, 
        'win_percentage': 0
    })
    
    return {
        'venue': venue,
        'total_matches_at_venue': venue_data['total_matches'],
        team1: team1_venue_stats,
        team2: team2_venue_stats
    }

def predict_match(team1, team2, venue):
    """Enhanced prediction logic"""
    try:
        # Standardize names
        team1 = standardize_team_names_single(team1)
        team2 = standardize_team_names_single(team2)
        venue = standardize_venue_names_single(venue)
        
        # Get venue stats
        venue_data = venue_stats.get(venue, {'team_stats': {}})
        team1_venue = venue_data['team_stats'].get(team1, {'win_percentage': 0})
        team2_venue = venue_data['team_stats'].get(team2, {'win_percentage': 0})
        
        # Get overall stats
        team1_overall = team_stats.get(team1, {'win_percentage': 0})
        team2_overall = team_stats.get(team2, {'win_percentage': 0})
        
        # Calculate scores
        team1_recent = get_recent_form(team1)
        team2_recent = get_recent_form(team2)
        team1_recent_wins = sum(1 for match in team1_recent if match['result'] == 'Won')
        team2_recent_wins = sum(1 for match in team2_recent if match['result'] == 'Won')
        
        h2h = get_head_to_head(team1, team2)
        team1_h2h = h2h.get(f'{team1}_wins', 0)
        team2_h2h = h2h.get(f'{team2}_wins', 0)
        
        # Calculate final scores with weights
        team1_score = (
            team1_venue.get('win_percentage', 0) * 0.4 +    # Venue: 40%
            team1_overall.get('win_percentage', 0) * 0.3 +  # Overall: 30%
            (team1_recent_wins / 5 * 100) * 0.2 +          # Recent form: 20%
            (team1_h2h / max(1, h2h['total_matches']) * 100) * 0.1  # H2H: 10%
        )
        
        team2_score = (
            team2_venue.get('win_percentage', 0) * 0.4 +
            team2_overall.get('win_percentage', 0) * 0.3 +
            (team2_recent_wins / 5 * 100) * 0.2 +
            (team2_h2h / max(1, h2h['total_matches']) * 100) * 0.1
        )
        
        # Determine winner and probabilities
        total_score = team1_score + team2_score
        team1_prob = (team1_score / total_score * 100) if total_score > 0 else 50
        team2_prob = (team2_score / total_score * 100) if total_score > 0 else 50
        
        predicted_winner = team1 if team1_score > team2_score else team2
        
        # Return comprehensive analysis
        return {
            'prediction': {
                'predicted_winner': predicted_winner,
                'team1_win_probability': round(team1_prob, 2),
                'team2_win_probability': round(team2_prob, 2)
            },
            'team_stats': {
                team1: team1_overall,
                team2: team2_overall
            },
            'recent_form': {
                team1: team1_recent,
                team2: team2_recent
            },
            'head_to_head': h2h,
            'venue_stats': get_venue_stats_for_teams(team1, team2, venue)
        }
        
    except Exception as e:
        return {'error': str(e)}

# HTML template for the web interface
HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>
<head>
    <title>IPL Match Predictor</title>
    <link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700&display=swap" rel="stylesheet">
    <style>
        body {
            font-family: 'Poppins', sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            color: #333;
        }
        .container {
            background: rgba(255, 255, 255, 0.95);
            padding: 40px;
            border-radius: 20px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(10px);
        }
        h1 {
            text-align: center;
            color: #1e3c72;
            font-size: 2.5em;
            margin-bottom: 30px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        }
        .form-group {
            margin-bottom: 25px;
        }
        label {
            display: block;
            margin-bottom: 8px;
            font-weight: 500;
            color: #2a5298;
            font-size: 1.1em;
        }
        select {
            width: 100%;
            padding: 12px;
            border: 2px solid #e1e8f0;
            border-radius: 12px;
            font-size: 1em;
            font-family: 'Poppins', sans-serif;
            background: white;
            transition: all 0.3s ease;
        }
        select:focus {
            border-color: #2a5298;
            box-shadow: 0 0 0 3px rgba(42, 82, 152, 0.2);
            outline: none;
        }
        button {
            width: 100%;
            padding: 14px;
            border: none;
            border-radius: 12px;
            font-size: 1.1em;
            font-weight: 600;
            color: white;
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            cursor: pointer;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(42, 82, 152, 0.3);
        }
        .result {
            margin-top: 40px;
            animation: fadeIn 0.5s ease-out;
        }
        .prediction {
            text-align: center;
            margin-bottom: 40px;
            padding: 30px;
            background: linear-gradient(135deg, #f6f9fc 0%, #edf2f7 100%);
            border-radius: 15px;
            border: 1px solid #e1e8f0;
        }
        .prediction h2 {
            color: #1e3c72;
            margin-bottom: 20px;
        }
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        .stats-section {
            background: white;
            padding: 25px;
            border-radius: 15px;
            border: 1px solid #e1e8f0;
            transition: transform 0.2s ease;
        }
        .stats-section:hover {
            transform: translateY(-5px);
            box-shadow: 0 8px 24px rgba(42, 82, 152, 0.15);
        }
        .stats-section h3 {
            margin-top: 0;
            color: #1e3c72;
            font-size: 1.3em;
            border-bottom: 2px solid #e1e8f0;
            padding-bottom: 10px;
        }
        .match-result {
            padding: 10px;
            margin: 5px 0;
            border-radius: 8px;
            font-size: 0.9em;
        }
        .won {
            background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%);
            color: #155724;
        }
        .lost {
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            color: #721c24;
        }
        .error {
            color: #721c24;
            background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%);
            padding: 15px;
            border-radius: 12px;
            text-align: center;
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        @media (max-width: 768px) {
            .stats-grid {
                grid-template-columns: 1fr;
            }
            .container {
                padding: 20px;
            }
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>🏏 IPL Match Predictor</h1>
        
        <form id="predictionForm">
            <div class="form-group">
                <label for="team1">Team 1:</label>
                <select id="team1" name="team1" required>
                    <option value="">Select Team 1</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="team2">Team 2:</label>
                <select id="team2" name="team2" required>
                    <option value="">Select Team 2</option>
                </select>
            </div>
            
            <div class="form-group">
                <label for="venue">Venue:</label>
                <select id="venue" name="venue" required>
                    <option value="">Select Venue</option>
                </select>
            </div>
            
            <button type="submit">Predict Match</button>
        </form>
        
        <div id="results"></div>
    </div>

    <script>
        // Populate dropdowns with teams and venues
        async function loadData() {
            try {
                const response = await fetch('/api/data');
                const data = await response.json();
                
                const team1Select = document.getElementById('team1');
                const team2Select = document.getElementById('team2');
                const venueSelect = document.getElementById('venue');
                
                // Populate teams
                data.teams.forEach(team => {
                    team1Select.innerHTML += `<option value="${team}">${team}</option>`;
                    team2Select.innerHTML += `<option value="${team}">${team}</option>`;
                });
                
                // Populate venues
                data.venues.forEach(venue => {
                    venueSelect.innerHTML += `<option value="${venue}">${venue}</option>`;
                });
                
            } catch (error) {
                console.error('Error loading data:', error);
            }
        }
        
        // Handle form submission
        document.getElementById('predictionForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const formData = new FormData(e.target);
            const team1 = formData.get('team1');
            const team2 = formData.get('team2');
            const venue = formData.get('venue');
            
            if (team1 === team2) {
                alert('Please select different teams');
                return;
            }
            
            try {
                const response = await fetch('/predict', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        team1: team1,
                        team2: team2,
                        venue: venue
                    })
                });
                
                const result = await response.json();
                displayResults(result, team1, team2, venue);
                
            } catch (error) {
                document.getElementById('results').innerHTML = 
                    `<div class="error">Error: ${error.message}</div>`;
            }
        });
        
        function displayResults(data, team1, team2, venue) {
            if (data.error) {
                document.getElementById('results').innerHTML = 
                    `<div class="error">Error: ${data.error}</div>`;
                return;
            }
            
            // Create default stats object
            const defaultStats = {
                matches_played: 0,
                wins: 0,
                losses: 0,
                win_percentage: 0
            };

            // Get team stats with defaults
            const team1Stats = data.team_stats?.[team1] || defaultStats;
            const team2Stats = data.team_stats?.[team2] || defaultStats;
            
            // Get venue stats with defaults
            const venueStats = data.venue_stats || {
                total_matches_at_venue: 0,
                [team1]: { matches: 0, wins: 0, losses: 0, win_percentage: 0 },
                [team2]: { matches: 0, wins: 0, losses: 0, win_percentage: 0 }
            };

            const results = document.getElementById('results');
            results.innerHTML = `
                <div class="result">
                    <div class="prediction">
                        <h2>Match Prediction</h2>
                        <h3>${team1} vs ${team2} at ${venue}</h3>
                        <h2>🏆 Predicted Winner: ${data.prediction?.predicted_winner || 'Unknown'}</h2>
                        <p><strong>${team1}:</strong> ${data.prediction?.team1_win_probability?.toFixed(1) || 0}% chance</p>
                        <p><strong>${team2}:</strong> ${data.prediction?.team2_win_probability?.toFixed(1) || 0}% chance</p>
                    </div>
                    
                    <div class="stats-grid">
                        <div class="stats-section">
                            <h3>Overall Team Statistics</h3>
                            <h4>${team1}:</h4>
                            <p>Matches: ${team1Stats.matches_played}</p>
                            <p>Wins: ${team1Stats.wins}</p>
                            <p>Losses: ${team1Stats.losses}</p>
                            <p>Win %: ${team1Stats.win_percentage?.toFixed(1) || 0}%</p>
                            
                            <h4>${team2}:</h4>
                            <p>Matches: ${team2Stats.matches_played}</p>
                            <p>Wins: ${team2Stats.wins}</p>
                            <p>Losses: ${team2Stats.losses}</p>
                            <p>Win %: ${team2Stats.win_percentage?.toFixed(1) || 0}%</p>
                        </div>
                        
                        <div class="stats-section">
                            <h3>Head-to-Head Record</h3>
                            <p>Total Matches: ${data.head_to_head.total_matches}</p>
                            <p>${team1} Wins: ${data.head_to_head[team1 + '_wins'] || 0}</p>
                            <p>${team2} Wins: ${data.head_to_head[team2 + '_wins'] || 0}</p>
                        </div>
                        
                        <div class="stats-section">
                            <h3>Recent Form (Last 5 matches)</h3>
                            <h4>${team1}:</h4>
                            ${data.recent_form[team1].map(match => 
                                `<div class="match-result ${match.result.toLowerCase()}">
                                    ${match.date}: vs ${match.opponent} - ${match.result}
                                </div>`
                            ).join('')}
                            
                            <h4>${team2}:</h4>
                            ${data.recent_form[team2].map(match => 
                                `<div class="match-result ${match.result.toLowerCase()}">
                                    ${match.date}: vs ${match.opponent} - ${match.result}
                                </div>`
                            ).join('')}
                        </div>
                        
                        <div class="stats-section">
                            <h3>Venue Statistics - ${venue}</h3>
                            <p>Total matches at venue: ${data.venue_stats.total_matches_at_venue}</p>
                            
                            <h4>${team1} at ${venue}:</h4>
                            <p>Matches: ${data.venue_stats[team1].matches}</p>
                            <p>Wins: ${data.venue_stats[team1].wins}</p>
                            <p>Losses: ${data.venue_stats[team1].losses}</p>
                            <p>Win %: ${data.venue_stats[team1].win_percentage.toFixed(1)}%</p>
                            
                            <h4>${team2} at ${venue}:</h4>
                            <p>Matches: ${data.venue_stats[team2].matches}</p>
                            <p>Wins: ${data.venue_stats[team2].wins}</p>
                            <p>Losses: ${data.venue_stats[team2].losses}</p>
                            <p>Win %: ${data.venue_stats[team2].win_percentage.toFixed(1)}%</p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        // Load data when page loads
        loadData();
    </script>
</body>
</html>
'''

@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/data')
def get_data():
    """Return available teams and venues"""
    try:
        teams = list(team_stats.keys())
        venues = list(venue_stats.keys())
        return jsonify({
            'teams': sorted(teams),
            'venues': sorted(venues)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        data = request.json
        team1 = data.get('team1')
        team2 = data.get('team2')
        venue = data.get('venue')
        
        if not all([team1, team2, venue]):
            return jsonify({'error': 'Missing required fields'}), 400
        
        if team1 == team2:
            return jsonify({'error': 'Teams must be different'}), 400
        
        result = predict_match(team1, team2, venue)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/predict/<team1>/<team2>/<venue>')
def predict_api(team1, team2, venue):
    """API endpoint for predictions"""
    try:
        result = predict_match(team1, team2, venue)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Add a health check endpoint
@app.route('/api/health')
def health_check():
    return jsonify({"status": "healthy"}), 200

if __name__ == '__main__':
    print("Starting IPL Match Predictor...")
    print("Make sure you have run trainmodel.py first to create all required pickle files")
    print("Access the application at: http://localhost:5000")
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)