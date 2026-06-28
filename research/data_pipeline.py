import os
import pandas as pd
import numpy as np

# Team normalization dictionary to map historical team names to their modern successors/standard names
TEAM_NORMALIZATION = {
    "German DR": "Germany",
    "West Germany": "Germany",
    "Soviet Union": "Russia",
    "Yugoslavia": "Serbia",
    "Czechoslovakia": "Czech Republic",
    "Zaire": "DR Congo",
    "Dutch East Indies": "Indonesia",
    "Korea Republic": "South Korea",
    "United States": "USA",
    "China PR": "China",
}

def clean_team_name(name):
    name = str(name).strip()
    return TEAM_NORMALIZATION.get(name, name)

def get_k_factor(tournament):
    """
    Returns the K-factor based on match importance.
    - World Cup finals matches (Group + Knockout): 60
    - Major Continental tournaments (Euros, Copa America, etc.): 50
    - World Cup/Continental qualifiers: 40
    - Friendly matches: 20
    - Others: 30
    """
    t_lower = tournament.lower()
    if "fifa world cup" in t_lower:
        if "qualification" in t_lower:
            return 40
        return 60
    elif any(term in t_lower for term in ["uefa euro", "copa américa", "african cup of nations", "afc asian cup", "concacaf gold cup"]):
        if "qualification" in t_lower or "qualifying" in t_lower:
            return 40
        return 50
    elif "friendly" in t_lower:
        return 20
    else:
        return 30

def get_goal_diff_multiplier(gd):
    """
    Goal differential multiplier (G) to reward larger margins of victory:
    - If gd <= 1: G = 1
    - If gd == 2: G = 1.5
    - If gd >= 3: G = (11 + gd) / 8
    """
    if gd <= 1:
        return 1.0
    elif gd == 2:
        return 1.5
    else:
        return (11.0 + gd) / 8.0

def calculate_elo_probability(r_home, r_away, neutral_venue):
    """
    Expected outcome We using a standard logistic formula.
    Includes home advantage of +100 Elo rating points if venue is not neutral.
    """
    home_adv = 0 if neutral_venue else 100
    diff = r_home + home_adv - r_away
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def run_pipeline(start_year=1970, data_dir="data", force_download=False):
    print("--- Starting Quant Data Engineering Pipeline ---")
    
    os.makedirs(data_dir, exist_ok=True)
    raw_path = os.path.join(data_dir, "results.csv")
    
    # Download Kaggle historical international matches if not already cached
    if force_download or not os.path.exists(raw_path):
        url = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
        print(f"Downloading historical matches from {url}...")
        try:
            df_raw = pd.read_csv(url)
            df_raw.to_csv(raw_path, index=False)
            print("Successfully saved raw data to cache.")
        except Exception as e:
            print(f"Error downloading data: {e}")
            raise e
    else:
        print("Using cached results.csv data.")
        df_raw = pd.read_csv(raw_path)
        
    print(f"Loaded {len(df_raw)} historical matches.")
    
    # Cleaning and Preprocessing
    df = df_raw.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"]).copy()
    df["date"] = pd.to_datetime(df["date"])
    
    # Normalize team names
    df["home_team"] = df["home_team"].apply(clean_team_name)
    df["away_team"] = df["away_team"].apply(clean_team_name)
    
    # Filter for relevant eras (default modern era >= 1970)
    df = df[df["date"].dt.year >= start_year].copy()
    df = df.sort_values(by="date").reset_index(drop=True)
    
    print(f"Filtered to {len(df)} matches since {start_year}.")
    
    # Point-in-time ratings and rest state tracker
    elo_ratings = {}         # team -> rating (float)
    last_match_date = {}     # team -> last match datetime
    
    # Initialize trackers
    all_teams = set(df["home_team"].unique()).union(set(df["away_team"].unique()))
    for team in all_teams:
        elo_ratings[team] = 1500.0
        last_match_date[team] = None
        
    # Feature lists
    home_elo_list = []
    away_elo_list = []
    home_rest_days_list = []
    away_rest_days_list = []
    is_neutral_list = []
    targets = [] # Home team outcome: 1.0 (win), 0.5 (draw), 0.0 (loss)
    
    print("Computing point-in-time features chronologically (no future leakage)...")
    
    for idx, row in df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])
        neutral = bool(row["neutral"])
        tournament = row["tournament"]
        match_date = row["date"]
        
        # 1. Capture POINT-IN-TIME features BEFORE the match updates
        h_elo = elo_ratings[home]
        a_elo = elo_ratings[away]
        
        # Calculate point-in-time rest days (clipped to a max of 30 to limit outliers)
        if last_match_date[home] is None:
            home_rest = 30.0
        else:
            home_rest = min(30.0, float((match_date - last_match_date[home]).days))
            
        if last_match_date[away] is None:
            away_rest = 30.0
        else:
            away_rest = min(30.0, float((match_date - last_match_date[away]).days))
            
        home_elo_list.append(h_elo)
        away_elo_list.append(a_elo)
        home_rest_days_list.append(home_rest)
        away_rest_days_list.append(away_rest)
        is_neutral_list.append(1.0 if neutral else 0.0)
        
        # Match result target
        if h_score > a_score:
            w_home = 1.0
        elif h_score == a_score:
            w_home = 0.5
        else:
            w_home = 0.0
            
        targets.append(w_home)
        
        # 2. Update state trackers AFTER capturing features (strictly chronologically)
        last_match_date[home] = match_date
        last_match_date[away] = match_date
        
        # Calculate expected win probability for Home
        we_home = calculate_elo_probability(h_elo, a_elo, neutral)
        
        # Calculate goal diff multiplier and K-factor
        gd = abs(h_score - a_score)
        G = get_goal_diff_multiplier(gd)
        K = get_k_factor(tournament)
        
        # Elo ratings update
        delta = K * G * (w_home - we_home)
        
        elo_ratings[home] = h_elo + delta
        elo_ratings[away] = a_elo - delta
        
    # Append features to dataframe
    df["Home_Elo"] = home_elo_list
    df["Away_Elo"] = away_elo_list
    df["Home_Rest_Days"] = home_rest_days_list
    df["Away_Rest_Days"] = away_rest_days_list
    df["Is_Neutral_Venue"] = is_neutral_list
    df["Target"] = targets
    
    processed_path = os.path.join(data_dir, "processed_matches.csv")
    df.to_csv(processed_path, index=False)
    print(f"Processed dataset saved to {processed_path}. Shape: {df.shape}")
    
    return elo_ratings

if __name__ == "__main__":
    run_pipeline()
