import pandas as pd
import numpy as np

print("Downloading Premier League Data (23/24 Season)...")
# This URL goes straight to the CSV on football-data.co.uk
url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)

# Keep only the columns we care about
# FTHG = Home Goals, FTAG = Away Goals, FTR = Match Result (H, D, A)
df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()

# 1. Map the Target to Integers (for our C++ math later)
# Home Win = 1, Draw = 0, Away Win = -1
result_map = {'H': 1, 'D': 0, 'A': -1}
df['Target'] = df['FTR'].map(result_map)

print("Engineering 'Recent Form' Features...")

# 2. Dictionaries to track the last 3 goals scored/conceded for every team
team_goals_scored = {team: [] for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()}
team_goals_conceded = {team: [] for team in team_goals_scored.keys()}

# These lists will become our new columns
home_form_scored, away_form_scored = [], []
home_form_conceded, away_form_conceded = [], []

# 3. Time-Travel Loop: Walk through the season chronologically
for index, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    
    # Calculate averages of the last 3 games (or fewer if season just started)
    # Note: We do this BEFORE updating the lists with the current game's goals! (No Data Leakage)
    h_scored_avg = np.mean(team_goals_scored[home][-3:]) if team_goals_scored[home] else 0
    h_conceded_avg = np.mean(team_goals_conceded[home][-3:]) if team_goals_conceded[home] else 0
    
    a_scored_avg = np.mean(team_goals_scored[away][-3:]) if team_goals_scored[away] else 0
    a_conceded_avg = np.mean(team_goals_conceded[away][-3:]) if team_goals_conceded[away] else 0
    
    home_form_scored.append(round(h_scored_avg, 2))
    home_form_conceded.append(round(h_conceded_avg, 2))
    away_form_scored.append(round(a_scored_avg, 2))
    away_form_conceded.append(round(a_conceded_avg, 2))
    
    # NOW we update the team's history with today's result
    team_goals_scored[home].append(row['FTHG'])
    team_goals_conceded[home].append(row['FTAG'])
    
    team_goals_scored[away].append(row['FTAG'])
    team_goals_conceded[away].append(row['FTHG'])

# Add the engineered lists back to the dataframe
df['Home_Scored_Last_3'] = home_form_scored
df['Home_Conceded_Last_3'] = home_form_conceded
df['Away_Scored_Last_3'] = away_form_scored
df['Away_Conceded_Last_3'] = away_form_conceded

# 4. Drop the first 30 games (because the early season averages are inaccurate)
# and drop the raw goals columns (so our model doesn't cheat)
final_df = df.iloc[30:].drop(columns=['Date', 'FTHG', 'FTAG', 'FTR'])

print("\n--- Final Dataset Glimpse ---")
print(final_df.head(5).to_string())

# Save it to a clean CSV
final_df.to_csv("clean_football_data.csv", index=False)
print("\nSaved to 'clean_football_data.csv'. Ready for Machine Learning!")
