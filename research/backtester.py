import numpy as np
import pandas as pd

# Popular teams that are subject to public sentiment bias (over-bet by the general public)
POPULAR_TEAMS = {"Brazil", "Argentina", "England", "France", "Spain", "Germany", "Portugal"}

def calculate_brier_score(predictions, actual_labels):
    """
    Computes the multi-class Brier Score: 1/N * sum_t(sum_c((f_tc - o_tc)^2))
    predictions: 2D array of shape [N, 3] (Home Win, Draw, Away Win probabilities)
    actual_labels: 1D array of floats [N] where:
                   1.0 is Home Win, 0.5 is Draw, 0.0 is Away Win
    """
    N = len(actual_labels)
    if N == 0:
        return 0.0, 0.0
    
    # One-hot encode the actual labels
    # class 0: Home Win, class 1: Draw, class 2: Away Win
    o = np.zeros((N, 3))
    for i, val in enumerate(actual_labels):
        if val == 1.0:
            o[i, 0] = 1.0
        elif val == 0.5:
            o[i, 1] = 1.0
        elif val == 0.0:
            o[i, 2] = 1.0
            
    # Calculate multi-class Brier Score
    squared_errors = (predictions - o) ** 2
    mc_brier = np.mean(np.sum(squared_errors, axis=1))
    
    # Calculate binary Brier Score for Home Win (class 0)
    bin_brier = np.mean((predictions[:, 0] - o[:, 0]) ** 2)
    
    return mc_brier, bin_brier

def generate_bookmaker_odds(df_test, margin=0.06):
    """
    Simulates bookmaker odds for the test set.
    Uses Elo expected outcomes as the baseline probabilities and applies:
    1. A draw probability baseline of 24%.
    2. Public sentiment bias: lowers the odds on highly popular teams (e.g. Brazil, Argentina) 
       because the public over-bets them, and increases the odds on their underdogs.
    3. Bookmaker vig (margin).
    """
    N = len(df_test)
    odds = np.zeros((N, 3)) # Home, Draw, Away odds
    
    # Expected outcome for home team:
    home_advs = np.where(df_test["neutral"], 0.0, 100.0)
    elo_diffs = df_test["Home_Elo"].values - df_test["Away_Elo"].values + home_advs
    p_elo_home = 1.0 / (1.0 + 10.0 ** (-elo_diffs / 400.0))
    
    for i in range(N):
        p_home = p_elo_home[i]
        p_draw = 0.24
        
        # Scale home and away to fit the draw rate
        scale = 1.0 - p_draw
        p_home_scaled = p_home * scale
        p_away_scaled = (1.0 - p_home) * scale
        
        # Apply public sentiment bias
        home_team = df_test.iloc[i]["home_team"]
        away_team = df_test.iloc[i]["away_team"]
        
        # Shift 5% probability towards the popular team
        if home_team in POPULAR_TEAMS and away_team not in POPULAR_TEAMS:
            p_home_scaled = min(0.90, p_home_scaled + 0.05)
            p_away_scaled = max(0.05, p_away_scaled - 0.05)
        elif away_team in POPULAR_TEAMS and home_team not in POPULAR_TEAMS:
            p_away_scaled = min(0.90, p_away_scaled + 0.05)
            p_home_scaled = max(0.05, p_home_scaled - 0.05)
            
        # Add bookmaker margin (vig)
        p_home_vig = p_home_scaled * (1.0 + margin)
        p_draw_vig = p_draw * (1.0 + margin)
        p_away_vig = p_away_scaled * (1.0 + margin)
        
        # Bounds check
        total_p = p_home_vig + p_draw_vig + p_away_vig
        p_home_vig /= total_p
        p_draw_vig /= total_p
        p_away_vig /= total_p
        
        # Offered Odds
        odds[i, 0] = 1.0 / p_home_vig
        odds[i, 1] = 1.0 / p_draw_vig
        odds[i, 2] = 1.0 / p_away_vig
        
    return odds

def run_backtest(df_test, predictions, edge_threshold=0.05, margin=0.06):
    """
    Runs betting backtest simulation on the test set.
    """
    N = len(df_test)
    odds = generate_bookmaker_odds(df_test, margin=margin)
    targets = df_test["Target"].values
    
    total_bets = 0
    total_pnl = 0.0
    wins = 0
    pnl_history = []
    
    # Expected outcome for home team from Elo baseline:
    home_advs = np.where(df_test["neutral"], 0.0, 100.0)
    elo_diffs = df_test["Home_Elo"].values - df_test["Away_Elo"].values + home_advs
    p_elo_home = 1.0 / (1.0 + 10.0 ** (-elo_diffs / 400.0))
    
    # Construct baseline Elo probabilities array for comparison
    elo_preds = np.zeros((N, 3))
    for i in range(N):
        elo_preds[i, 0] = p_elo_home[i] * 0.76
        elo_preds[i, 1] = 0.24
        elo_preds[i, 2] = (1.0 - p_elo_home[i]) * 0.76
        
    for i in range(N):
        y = targets[i]
        # Implied probabilities from bookmaker odds
        implied_p = 1.0 / odds[i]
        
        # Find outcomes where our model's probability exceeds the bookmaker's implied probability
        edges = predictions[i] - implied_p
        best_class = np.argmax(edges)
        
        # Bet on the class with the largest positive edge exceeding the threshold
        current_pnl_change = 0.0
        if edges[best_class] > edge_threshold:
            total_bets += 1
            # Check if this class is the actual result
            actual_class = 2 # Away Win (0.0)
            if y == 1.0:
                actual_class = 0 # Home Win
            elif y == 0.5:
                actual_class = 1 # Draw
                
            if best_class == actual_class:
                win_payout = odds[i, best_class] - 1.0
                total_pnl += win_payout
                current_pnl_change = win_payout
                wins += 1
            else:
                total_pnl -= 1.0
                current_pnl_change = -1.0
                
        pnl_history.append(total_pnl)
        
    roi = (total_pnl / total_bets) if total_bets > 0 else 0.0
    win_rate = (wins / total_bets) if total_bets > 0 else 0.0
    
    mc_brier_model, bin_brier_model = calculate_brier_score(predictions, targets)
    mc_brier_elo, bin_brier_elo = calculate_brier_score(elo_preds, targets)
    
    return {
        "total_bets": total_bets,
        "total_pnl": total_pnl,
        "roi": roi,
        "win_rate": win_rate,
        "brier_model_mc": mc_brier_model,
        "brier_model_bin": bin_brier_model,
        "brier_elo_mc": mc_brier_elo,
        "brier_elo_bin": bin_brier_elo,
        "pnl_history": pnl_history,
        "odds": odds
    }
