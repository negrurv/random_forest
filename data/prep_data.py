import pandas as pd
import numpy as np

url = "https://www.football-data.co.uk/mmz4281/2324/E0.csv"
df = pd.read_csv(url)


df = df[['Date', 'HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR']].copy()


result_map = {'H': 1, 'D': 0, 'A': -1}
df['Target'] = df['FTR'].map(result_map)


team_goals_scored = {team: [] for team in pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()}
team_goals_conceded = {team: [] for team in team_goals_scored.keys()}

home_form_scored, away_form_scored = [], []
home_form_conceded, away_form_conceded = [], []

for index, row in df.iterrows():
    home = row['HomeTeam']
    away = row['AwayTeam']
    
    
    h_scored_avg = np.mean(team_goals_scored[home][-3:]) if team_goals_scored[home] else 0
    h_conceded_avg = np.mean(team_goals_conceded[home][-3:]) if team_goals_conceded[home] else 0
    
    a_scored_avg = np.mean(team_goals_scored[away][-3:]) if team_goals_scored[away] else 0
    a_conceded_avg = np.mean(team_goals_conceded[away][-3:]) if team_goals_conceded[away] else 0
    
    home_form_scored.append(round(h_scored_avg, 2))
    home_form_conceded.append(round(h_conceded_avg, 2))
    away_form_scored.append(round(a_scored_avg, 2))
    away_form_conceded.append(round(a_conceded_avg, 2))
    
    team_goals_scored[home].append(row['FTHG'])
    team_goals_conceded[home].append(row['FTAG'])
    
    team_goals_scored[away].append(row['FTAG'])
    team_goals_conceded[away].append(row['FTHG'])

df['Home_Scored_Last_3'] = home_form_scored
df['Home_Conceded_Last_3'] = home_form_conceded
df['Away_Scored_Last_3'] = away_form_scored
df['Away_Conceded_Last_3'] = away_form_conceded


final_df = df.iloc[30:].drop(columns=['Date', 'FTHG', 'FTAG', 'FTR'])

print(final_df.head(5).to_string())

final_df.to_csv("clean_football_data.csv", index=False)
