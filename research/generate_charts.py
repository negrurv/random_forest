import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure imports from parent directory and research directory work
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from research.data_pipeline import run_pipeline
from research.backtester import run_backtest, calculate_brier_score
import rf_cpp


def main():
    print("==========================================================")
    # Enforce deterministic reproducibility on the Python side
    np.random.seed(42)
    import random

    random.seed(42)

    # 1. Run Data Engineering Pipeline
    run_pipeline(start_year=1970)

    processed_path = os.path.join(parent_dir, "data", "processed_matches.csv")
    if not os.path.exists(processed_path):
        print(f"Error: processed matches file not found at {processed_path}")
        sys.exit(1)

    df = pd.read_csv(processed_path)
    df["date"] = pd.to_datetime(df["date"])

    # 2. Split into Train & Test sets chronologically
    train_cutoff = "2022-11-20"
    df_train = df[df["date"] < pd.to_datetime(train_cutoff)].copy()
    df_test = df[df["date"] >= pd.to_datetime(train_cutoff)].copy()

    features_cols = [
        "Home_Elo",
        "Away_Elo",
        "Home_Rest_Days",
        "Away_Rest_Days",
        "Is_Neutral_Venue",
    ]

    # STRICT float64 C-CONTIGUOUS NUMPY ARRAYS
    X_train = np.ascontiguousarray(df_train[features_cols].values, dtype=np.float64)
    y_train = np.ascontiguousarray(df_train["Target"].values, dtype=np.float64)

    X_test = np.ascontiguousarray(df_test[features_cols].values, dtype=np.float64)
    y_test = np.ascontiguousarray(df_test["Target"].values, dtype=np.float64)

    # 3. Train the C++ Random Forest Classifier (Zero-Copy)
    num_trees = 100
    max_depth = 10
    min_samples_split = 6
    feature_fraction = 0.8

    print("\n--- Training C++ Classification Forest (Deterministic Seed) ---")
    model = rf_cpp.RandomForest(
        num_trees, max_depth, min_samples_split, feature_fraction
    )
    model.train(X_train, y_train)
    print("Training complete.")

    # 4. Predict on Test Set
    print("\nPredicting on test set...")
    predictions = model.predict_batch(X_test)  # shape [N, 3]

    # 5. Run Betting Backtest
    print("Running backtest...")
    result = run_backtest(df_test, predictions, edge_threshold=0.05, margin=0.06)

    # 6. Calculate Risk Metrics
    pnl_hist = result["pnl_history"]
    odds = result["odds"]
    targets = y_test
    N = len(df_test)

    # Calculate Max Drawdown
    bankroll = 100.0 + np.array(pnl_hist)
    peaks = np.maximum.accumulate(bankroll)
    drawdowns = peaks - bankroll
    max_drawdown = np.max(drawdowns)

    # Calculate daily Sharpe Ratio
    # We trace PnL changes for each match, associate them with dates, and group by day
    pnl_changes = []
    prev_pnl = 0.0
    for val in pnl_hist:
        pnl_changes.append(val - prev_pnl)
        prev_pnl = val

    df_bets = df_test.copy()
    df_bets["pnl_change"] = pnl_changes
    daily_pnl = df_bets.groupby("date")["pnl_change"].sum()

    # Return rate as daily change relative to starting bankroll (100 units)
    daily_returns = daily_pnl / 100.0
    mean_daily_return = daily_returns.mean()
    std_daily_return = daily_returns.std()

    if std_daily_return > 0:
        sharpe_ratio = (mean_daily_return / std_daily_return) * np.sqrt(365)
    else:
        sharpe_ratio = 0.0

    print("\n================ QUANT RISK METRICS ================")
    print(f"Total Bets Placed : {result['total_bets']}")
    print(f"Betting Win Rate  : {result['win_rate'] * 100:.2f}%")
    print(f"Betting ROI       : {result['roi'] * 100:+.2f}%")
    print(f"Betting PnL       : {result['total_pnl']:+.2f} units")
    print(f"Max Drawdown      : {max_drawdown:.2f} units")
    print(f"Annualized Sharpe : {sharpe_ratio:.4f}")
    print("====================================================")

    # 7. Generate Baseline Strategy PnL (Always bet the favorite)
    baseline_pnl = 0.0
    baseline_history = []
    for i in range(N):
        odds_i = odds[i]
        # Favorite has lower odds between Home (0) and Away (2)
        fav_class = 0 if odds_i[0] < odds_i[2] else 2
        actual_class = 2
        y = targets[i]
        if y == 1.0:
            actual_class = 0
        elif y == 0.5:
            actual_class = 1

        if fav_class == actual_class:
            baseline_pnl += odds_i[fav_class] - 1.0
        else:
            baseline_pnl -= 1.0
        baseline_history.append(baseline_pnl)

    # 8. Plot Chart 1: Cumulative PnL
    plt.figure(figsize=(10, 6))
    plt.plot(
        pnl_hist,
        label="Model Strategy (Value Betting with 5% Edge)",
        color="#1abc9c",
        linewidth=2.5,
    )
    plt.plot(
        baseline_history,
        label="Baseline Strategy (Always Bet Favorite)",
        color="#e74c3c",
        linewidth=1.5,
        linestyle="--",
    )
    plt.axhline(0, color="gray", linestyle=":", linewidth=1)
    plt.title(
        "Cumulative Strategy PnL over Time (Post-2022 Backtest)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Matches Sorted Chronologically", fontsize=12)
    plt.ylabel("PnL (Units)", fontsize=12)
    plt.legend(fontsize=10, loc="upper left")
    plt.grid(True, linestyle=":", alpha=0.6)

    # Make design sleek and premium
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    pnl_path = os.path.join(parent_dir, "research", "cumulative_pnl.png")
    plt.savefig(pnl_path, dpi=300)
    plt.close()
    print(f"Saved Cumulative PnL chart to {pnl_path}")

    # 9. Plot Chart 2: Probability Calibration Curve
    all_preds = predictions.flatten()
    o = np.zeros((N, 3))
    for i, val in enumerate(targets):
        if val == 1.0:
            o[i, 0] = 1.0
        elif val == 0.5:
            o[i, 1] = 1.0
        elif val == 0.0:
            o[i, 2] = 1.0
    all_actuals = o.flatten()

    # Bucket into 10% bins
    bins = np.linspace(0.0, 1.0, 11)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    pred_win_rates = []
    actual_win_rates = []

    for j in range(10):
        low, high = bins[j], bins[j + 1]
        mask = (all_preds >= low) & (all_preds < high)
        if np.sum(mask) > 0:
            pred_win_rates.append(np.mean(all_preds[mask]))
            actual_win_rates.append(np.mean(all_actuals[mask]))
        else:
            pred_win_rates.append(bin_centers[j])
            actual_win_rates.append(np.nan)

    plt.figure(figsize=(7, 7))
    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        label="Perfect Calibration (y = x)",
    )
    plt.plot(
        pred_win_rates,
        actual_win_rates,
        marker="o",
        color="#3498db",
        label="Random Forest Model",
        linewidth=2,
    )
    plt.title(
        "Probability Calibration Curve (10% Bins)",
        fontsize=14,
        fontweight="bold",
        pad=15,
    )
    plt.xlabel("Average Predicted Probability", fontsize=12)
    plt.ylabel("Actual Win Rate / Outcome Frequency", fontsize=12)
    plt.xlim(0.0, 1.0)
    plt.ylim(0.0, 1.0)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle=":", alpha=0.6)

    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["right"].set_visible(False)
    plt.tight_layout()

    cal_path = os.path.join(parent_dir, "research", "calibration_curve.png")
    plt.savefig(cal_path, dpi=300)
    plt.close()
    print(f"Saved Calibration Curve to {cal_path}")




if __name__ == "__main__":
    main()
