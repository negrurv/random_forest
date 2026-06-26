import os
import sys
import pandas as pd
import numpy as np

# Add the current directory to sys.path to allow imports from research/
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from research.data_pipeline import run_pipeline, get_k_factor
from research.backtester import run_backtest, calculate_brier_score
import rf_cpp

# Define 2026 World Cup Group Composition
GROUPS_2026 = {
    "A": ["Mexico", "South Africa", "South Korea", "Czech Republic"],
    "B": ["Canada", "Qatar", "Switzerland", "Bosnia and Herzegovina"],
    "C": ["Brazil", "Morocco", "Haiti", "Scotland"],
    "D": ["USA", "Paraguay", "Australia", "Turkey"],
    "E": ["Germany", "Curaçao", "Ivory Coast", "Ecuador"],
    "F": ["Netherlands", "Japan", "Sweden", "Tunisia"],
    "G": ["Portugal", "Uzbekistan", "Colombia", "DR Congo"],
    "H": ["England", "Croatia", "Panama", "Ghana"],
    "I": ["France", "Senegal", "Iraq", "Norway"],
    "J": ["Spain", "Saudi Arabia", "Uruguay", "Cape Verde"],
    "K": ["Argentina", "Austria", "Jordan", "Algeria"],
    "L": ["Belgium", "Egypt", "Iran", "New Zealand"]
}

def get_team_state_as_of(df, team, date):
    """
    Computes the exact point-in-time Elo rating and identifies the last match date
    for a team at a specific date, using only matches played BEFORE that date.
    """
    team_matches = df[((df["home_team"] == team) | (df["away_team"] == team)) & (df["date"] < pd.to_datetime(date))]
    if team_matches.empty:
        return {
            "elo": 1500.0,
            "last_match_date": None
        }
    
    team_matches = team_matches.sort_values(by="date")
    last_match = team_matches.iloc[-1]
    
    h_elo = last_match["Home_Elo"]
    a_elo = last_match["Away_Elo"]
    neutral = last_match["neutral"]
    h_score = last_match["home_score"]
    a_score = last_match["away_score"]
    tournament = last_match["tournament"]
    
    w_home = 1.0 if h_score > a_score else (0.5 if h_score == a_score else 0.0)
    home_adv = 0 if neutral else 100
    we_home = 1.0 / (1.0 + 10.0 ** (-((h_elo + home_adv) - a_elo) / 400.0))
    
    gd = abs(h_score - a_score)
    G = 1.0 if gd <= 1 else (1.5 if gd == 2 else (11.0 + gd) / 8.0)
    K = get_k_factor(tournament)
    delta = K * G * (w_home - we_home)
    
    if last_match["home_team"] == team:
        post_elo = h_elo + delta
    else:
        post_elo = a_elo - delta
        
    return {
        "elo": post_elo,
        "last_match_date": last_match["date"]
    }

def get_matchup_features(df, team_a, team_b, date):
    state_a = get_team_state_as_of(df, team_a, date)
    state_b = get_team_state_as_of(df, team_b, date)
    
    if state_a["last_match_date"] is None:
        rest_a = 30.0
    else:
        rest_a = min(30.0, float((pd.to_datetime(date) - state_a["last_match_date"]).days))
        
    if state_b["last_match_date"] is None:
        rest_b = 30.0
    else:
        rest_b = min(30.0, float((pd.to_datetime(date) - state_b["last_match_date"]).days))
        
    feat = [state_a["elo"], state_b["elo"], rest_a, rest_b, 1.0] # Is_Neutral_Venue = 1.0
    return feat, state_a, state_b

def main():
    # Enforce deterministic reproducibility
    import random
    random.seed(42)
    np.random.seed(42)

    print("==================================================================")
    print("      WORLD CUP KNOCKOUT GAMES PREDICTION ENGINE (PHASE 2-4)       ")
    print("==================================================================")
    
    # 1. Run Data Engineering Pipeline to get clean historical matches and Elos
    run_pipeline(start_year=1970)
    
    processed_path = "data/processed_matches.csv"
    if not os.path.exists(processed_path):
        print(f"Error: processed matches file not found at {processed_path}")
        sys.exit(1)
        
    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])
    
    # 2. Split into Train & Test sets chronologically
    train_cutoff = "2022-11-20"
    df_train = df[df["date"] < pd.to_datetime(train_cutoff)].copy()
    df_test = df[df["date"] >= pd.to_datetime(train_cutoff)].copy()
    
    features_cols = ["Home_Elo", "Away_Elo", "Home_Rest_Days", "Away_Rest_Days", "Is_Neutral_Venue"]
    
    # STRICT float64 C-CONTIGUOUS NUMPY ARRAYS
    X_train = np.ascontiguousarray(df_train[features_cols].values, dtype=np.float64)
    y_train = np.ascontiguousarray(df_train["Target"].values, dtype=np.float64)
    
    # 3. Train the C++ Random Forest Classifier (Zero-Copy)
    num_trees = 100
    max_depth = 10
    min_samples_split = 6
    feature_fraction = 0.8
    
    print("\n--- Training C++ Classification Forest (Zero-Copy Numpy Interop) ---")
    model = rf_cpp.RandomForest(num_trees, max_depth, min_samples_split, feature_fraction)
    model.train(X_train, y_train)
    print("Training complete.")
    
    # Predict KO matchup helper (Symmetrized for Neutral Venues)
    def predict_ko_match(team_a, team_b, date):
        feat_forward, state_a, state_b = get_matchup_features(df, team_a, team_b, date)
        X_forward = np.ascontiguousarray([feat_forward], dtype=np.float64)
        probs_forward = model.predict_batch(X_forward)[0]
        
        feat_backward, _, _ = get_matchup_features(df, team_b, team_a, date)
        X_backward = np.ascontiguousarray([feat_backward], dtype=np.float64)
        probs_backward = model.predict_batch(X_backward)[0]
        
        p_a_win = (probs_forward[0] + probs_backward[2]) / 2.0
        p_draw = (probs_forward[1] + probs_backward[1]) / 2.0
        p_b_win = (probs_forward[2] + probs_backward[0]) / 2.0
        probs_sym = np.array([p_a_win, p_draw, p_b_win])
        
        # In knockout, a winner must advance (win = 90min win + 0.5 * draw)
        prob_a_advance = p_a_win + 0.5 * p_draw
        prob_b_advance = p_b_win + 0.5 * p_draw
        
        return prob_a_advance, prob_b_advance, probs_sym, state_a, state_b
        
    # ==================================================================
    # 5. Predict 2026 World Cup Standings & Brackets
    # ==================================================================
    print("\n==================================================================")
    print("      LIVE SIMULATION: 2026 WORLD CUP GROUPS AND STANDINGS        ")
    print("==================================================================")
    
    # Extract matches of 2026 World Cup group stage from results
    wc_2026_df = df[(df["date"] >= "2026-06-10") & (df["tournament"] == "FIFA World Cup")].copy()
    
    # Track team records: [wins, draws, losses, points, goals_scored, goals_conceded]
    standings = {}
    for letter, teams in GROUPS_2026.items():
        for team in teams:
            standings[team] = {"group": letter, "wins": 0, "draws": 0, "losses": 0, "points": 0, "gf": 0, "ga": 0}
            
    # Process already played matches
    print(f"Loading {len(wc_2026_df)} completed group stage matches...")
    for _, row in wc_2026_df.iterrows():
        home = row["home_team"]
        away = row["away_team"]
        h_score = int(row["home_score"])
        a_score = int(row["away_score"])
        
        if home in standings and away in standings:
            standings[home]["gf"] += h_score
            standings[home]["ga"] += a_score
            standings[away]["gf"] += a_score
            standings[away]["ga"] += h_score
            
            if h_score > a_score:
                standings[home]["wins"] += 1
                standings[home]["points"] += 3
                standings[away]["losses"] += 1
            elif h_score == a_score:
                standings[home]["draws"] += 1
                standings[home]["points"] += 1
                standings[away]["draws"] += 1
                standings[away]["points"] += 1
            else:
                standings[away]["wins"] += 1
                standings[away]["points"] += 3
                standings[home]["losses"] += 1
                
    # List of group matches to play/simulate (Matchday 3 for groups G to L)
    unplayed_matches = [
        # Group G
        ("Portugal", "Colombia"), ("Uzbekistan", "DR Congo"),
        # Group H
        ("England", "Panama"), ("Croatia", "Ghana"),
        # Group I
        ("France", "Norway"), ("Senegal", "Iraq"),
        # Group J
        ("Spain", "Uruguay"), ("Saudi Arabia", "Cape Verde"),
        # Group K
        ("Argentina", "Jordan"), ("Austria", "Algeria"),
        # Group L
        ("Belgium", "New Zealand"), ("Egypt", "Iran")
    ]
    
    print("\nSimulating unplayed Group Stage Matchday 3 matches (Symmetrized Neutral Venue):")
    sim_date = "2026-06-26"
    for ta, tb in unplayed_matches:
        # Symmetrize neutral matchup predictions for group stage
        feat_forward, state_a, state_b = get_matchup_features(df, ta, tb, sim_date)
        X_forward = np.ascontiguousarray([feat_forward], dtype=np.float64)
        probs_forward = model.predict_batch(X_forward)[0]
        
        feat_backward, _, _ = get_matchup_features(df, tb, ta, sim_date)
        X_backward = np.ascontiguousarray([feat_backward], dtype=np.float64)
        probs_backward = model.predict_batch(X_backward)[0]
        
        p_a_win = (probs_forward[0] + probs_backward[2]) / 2.0
        p_draw = (probs_forward[1] + probs_backward[1]) / 2.0
        p_b_win = (probs_forward[2] + probs_backward[0]) / 2.0
        probs = np.array([p_a_win, p_draw, p_b_win])
        
        # Determine match outcome based on highest predicted class probability
        outcome_idx = np.argmax(probs)
        if outcome_idx == 0: # Home Win
            h_score, a_score = 2, 0
            standings[ta]["wins"] += 1
            standings[ta]["points"] += 3
            standings[tb]["losses"] += 1
            res_str = f"{ta} Win"
        elif outcome_idx == 1: # Draw
            h_score, a_score = 1, 1
            standings[ta]["draws"] += 1
            standings[ta]["points"] += 1
            standings[tb]["draws"] += 1
            standings[tb]["points"] += 1
            res_str = "Draw"
        else: # Away Win
            h_score, a_score = 0, 2
            standings[tb]["wins"] += 1
            standings[tb]["points"] += 3
            standings[ta]["losses"] += 1
            res_str = f"{tb} Win"
            
        standings[ta]["gf"] += h_score
        standings[ta]["ga"] += a_score
        standings[tb]["gf"] += a_score
        standings[tb]["ga"] += h_score
        
        print(f"Simulating: {ta:12} vs {tb:12} -> Predicted: {res_str:15} (Home: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Away: {probs[2]*100:.1f}%)")
        
    # Calculate group standings
    groups_standings = {letter: [] for letter in GROUPS_2026.keys()}
    for team, stats in standings.items():
        stats["gd"] = stats["gf"] - stats["ga"]
        groups_standings[stats["group"]].append((team, stats))
        
    # Sort groups
    for letter in groups_standings.keys():
        # Sort key: points (desc), Goal Difference (desc), Goals For (desc)
        groups_standings[letter].sort(key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]), reverse=True)
        
    print("\n--- FINAL 2026 WORLD CUP GROUP STANDINGS ---")
    for letter, ranking in sorted(groups_standings.items()):
        print(f"\nGroup {letter}:")
        for pos, (team, s) in enumerate(ranking):
            print(f"  {pos+1}. {team:18} | Pts: {s['points']:d} | GD: {s['gd']:+d} | GF: {s['gf']:d}")
            
    # Determine the advancing teams: Top 2 from each group + 8 best 3rd places
    advancing_top2 = []
    third_place_candidates = []
    
    for letter, ranking in groups_standings.items():
        advancing_top2.append(ranking[0][0]) # 1st
        advancing_top2.append(ranking[1][0]) # 2nd
        third_place_candidates.append((ranking[2][0], ranking[2][1], letter)) # 3rd
        
    # Sort 3rd-place candidates
    third_place_candidates.sort(key=lambda x: (x[1]["points"], x[1]["gd"], x[1]["gf"]), reverse=True)
    best_8_third_places = [(x[0], x[2]) for x in third_place_candidates[:8]]
    
    print("\nBest 8 Third-Place Teams Advancing:")
    for idx, x in enumerate(third_place_candidates[:8]):
        print(f"  {idx+1}. {x[0]:15} | Pts: {x[1]['points']:d} | GD: {x[1]['gd']:+d} | GF: {x[1]['gf']:d}")
        
    # Combine all 32 advancing teams
    all_advancing = set(advancing_top2 + [x[0] for x in best_8_third_places])
    print(f"\nTotal Advancing Teams: {len(all_advancing)}")
    
    # Assign the 8 third-place teams dynamically based on official FIFA combinations
    allowed_map = {
        "1E": {"A", "B", "C", "D", "F"},
        "1I": {"C", "D", "F", "G", "H"},
        "1A": {"C", "E", "F", "H", "I"},
        "1L": {"E", "H", "I", "J", "K"},
        "1D": {"B", "E", "F", "I", "J"},
        "1G": {"A", "E", "H", "I", "J"},
        "1B": {"E", "F", "G", "I", "J"},
        "1K": {"D", "E", "I", "J", "L"}
    }
    
    def assign_third_places(best_8, allowed_map):
        keys = list(allowed_map.keys())
        def backtrack(idx, assigned_teams, assigned_slots):
            if idx == len(keys):
                return assigned_slots
            winner_pos = keys[idx]
            for team_name, group_letter in best_8:
                if team_name not in assigned_teams:
                    if group_letter in allowed_map[winner_pos]:
                        assigned_teams.add(team_name)
                        assigned_slots[winner_pos] = team_name
                        res = backtrack(idx + 1, assigned_teams, assigned_slots)
                        if res is not None:
                            return res
                        assigned_teams.remove(team_name)
                        del assigned_slots[winner_pos]
            return None
        return backtrack(0, set(), {})

    third_place_assignments = assign_third_places(best_8_third_places, allowed_map)
    if third_place_assignments is None:
        # Fallback greedy solver
        third_place_assignments = {}
        assigned = set()
        for winner_pos in allowed_map.keys():
            for team_name, group_letter in best_8_third_places:
                if team_name not in assigned:
                    third_place_assignments[winner_pos] = team_name
                    assigned.add(team_name)
                    break

    # Fetch teams dynamically
    teams_1st = {letter: groups_standings[letter][0][0] for letter in groups_standings}
    teams_2nd = {letter: groups_standings[letter][1][0] for letter in groups_standings}

    # Round of 32 Matchups (Official FIFA slots M73 to M88)
    r32_matches_def = {
        73: (teams_2nd["A"], teams_2nd["B"]),
        74: (teams_1st["E"], third_place_assignments["1E"]),
        75: (teams_1st["F"], teams_2nd["C"]),
        76: (teams_1st["C"], teams_2nd["F"]),
        77: (teams_1st["I"], third_place_assignments["1I"]),
        78: (teams_2nd["E"], teams_2nd["I"]),
        79: (teams_1st["A"], third_place_assignments["1A"]),
        80: (teams_1st["L"], third_place_assignments["1L"]),
        81: (teams_1st["D"], third_place_assignments["1D"]),
        82: (teams_1st["G"], third_place_assignments["1G"]),
        83: (teams_2nd["K"], teams_2nd["L"]),
        84: (teams_1st["H"], teams_2nd["J"]),
        85: (teams_1st["B"], third_place_assignments["1B"]),
        86: (teams_1st["J"], teams_2nd["H"]),
        87: (teams_1st["K"], third_place_assignments["1K"]),
        88: (teams_2nd["D"], teams_2nd["G"])
    }
    
    print(f"\nFinalized Round of 32 Brackets (Official Predefined Pairings):")
    for match_num in sorted(r32_matches_def.keys()):
        ta, tb = r32_matches_def[match_num]
        print(f"  Match {match_num:2d}: {ta:20} vs {tb:20}")
        
    # ==================================================================
    # 6. Simulate Knockout Stage
    # ==================================================================
    print("\n==================================================================")
    print("           SIMULATING 2026 WORLD CUP KNOCKOUT STAGE               ")
    print("==================================================================")
    
    # 2026 KO Dates
    r32_date = "2026-06-28"
    r16_date = "2026-07-04"
    qf_date = "2026-07-09"
    sf_date = "2026-07-14"
    final_date = "2026-07-19"
    
    print(f"\n[ROUND OF 32] (Dates: June 28 - July 3, 2026)")
    r32_winners = {}
    for match_num in sorted(r32_matches_def.keys()):
        ta, tb = r32_matches_def[match_num]
        pa, pb, probs, s_a, s_b = predict_ko_match(ta, tb, r32_date)
        winner = ta if pa >= pb else tb
        r32_winners[match_num] = winner
        print(f"  Match {match_num}: {ta:16} ({s_a['elo']:.0f}) vs {tb:16} ({s_b['elo']:.0f}) -> {winner} to ADVANCE (Prob A: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Prob B: {probs[2]*100:.1f}% | P({ta} Adv): {pa*100:.1f}%)")
        
    print(f"\n[ROUND OF 16] (Dates: July 4 - July 7, 2026)")
    r16_matchups_def = {
        89: (r32_winners[74], r32_winners[77]),
        90: (r32_winners[73], r32_winners[75]),
        91: (r32_winners[76], r32_winners[78]),
        92: (r32_winners[79], r32_winners[80]),
        93: (r32_winners[83], r32_winners[84]),
        94: (r32_winners[81], r32_winners[82]),
        95: (r32_winners[86], r32_winners[88]),
        96: (r32_winners[85], r32_winners[87])
    }
    
    r16_winners = {}
    for match_num in sorted(r16_matchups_def.keys()):
        ta, tb = r16_matchups_def[match_num]
        pa, pb, probs, s_a, s_b = predict_ko_match(ta, tb, r16_date)
        winner = ta if pa >= pb else tb
        r16_winners[match_num] = winner
        print(f"  Match {match_num}: {ta:16} ({s_a['elo']:.0f}) vs {tb:16} ({s_b['elo']:.0f}) -> {winner} to ADVANCE (Prob A: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Prob B: {probs[2]*100:.1f}% | P({ta} Adv): {pa*100:.1f}%)")
        
    print(f"\n[QUARTERFINALS] (Dates: July 9 - July 11, 2026)")
    qf_matchups_def = {
        97: (r16_winners[89], r16_winners[90]),
        98: (r16_winners[93], r16_winners[94]),
        99: (r16_winners[91], r16_winners[92]),
        100: (r16_winners[95], r16_winners[96])
    }
    
    qf_winners = {}
    for match_num in sorted(qf_matchups_def.keys()):
        ta, tb = qf_matchups_def[match_num]
        pa, pb, probs, s_a, s_b = predict_ko_match(ta, tb, qf_date)
        winner = ta if pa >= pb else tb
        qf_winners[match_num] = winner
        print(f"  Match {match_num}: {ta:16} ({s_a['elo']:.0f}) vs {tb:16} ({s_b['elo']:.0f}) -> {winner} to ADVANCE (Prob A: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Prob B: {probs[2]*100:.1f}% | P({ta} Adv): {pa*100:.1f}%)")
        
    print(f"\n[SEMIFINALS] (Dates: July 14 - July 15, 2026)")
    sf_matchups_def = {
        101: (qf_winners[97], qf_winners[98]),
        102: (qf_winners[99], qf_winners[100])
    }
    
    sf_winners = {}
    for match_num in sorted(sf_matchups_def.keys()):
        ta, tb = sf_matchups_def[match_num]
        pa, pb, probs, s_a, s_b = predict_ko_match(ta, tb, sf_date)
        winner = ta if pa >= pb else tb
        sf_winners[match_num] = winner
        print(f"  Match {match_num}: {ta:16} ({s_a['elo']:.0f}) vs {tb:16} ({s_b['elo']:.0f}) -> {winner} to ADVANCE (Prob A: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Prob B: {probs[2]*100:.1f}% | P({ta} Adv): {pa*100:.1f}%)")
        
    print(f"\n[WORLD CUP FINAL] (Date: July 19, 2026)")
    ta, tb = sf_winners[101], sf_winners[102]
    pa, pb, probs, s_a, s_b = predict_ko_match(ta, tb, final_date)
    champion = ta if pa >= pb else tb
    print(f"  Match 104: {ta:16} ({s_a['elo']:.0f}) vs {tb:16} ({s_b['elo']:.0f}) -> Predicted Champion: {champion.upper()} (Trophy probability: {max(pa, pb)*100:.1f}% | Prob A: {probs[0]*100:.1f}%, Draw: {probs[1]*100:.1f}%, Prob B: {probs[2]*100:.1f}%)")
    print("==================================================================")

if __name__ == "__main__":
    main()
