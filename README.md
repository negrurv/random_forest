# C++ Random Forest Engine: World Cup Knockout Games Predictor

A high-performance Random Forest classifier built from the ground up in **C++** with a **zero-copy Python interface** (`pybind11`). Designed specifically to bypass Python ML library overhead for sub-millisecond real-time batch inference and backtesting, applied to simulating the FIFA World Cup knockout stages.

---

## 🏗️ Project Architecture

* **Quant Dev Layer (`cpp_core/`)**: High-performance C++ implementation using an array-backed, flattened `std::vector<TreeNode>` tree structure to eliminate pointer indirection, optimize memory locality, and run branchless traversal.
* **Quant Research Layer (`research/`)**: Python data pipelines computing running point-in-time Elo ratings and team rest days without lookahead bias, plus a historical backtester.
* **Execution Script (`run_predictions.py`)**: End-to-end simulator running the live group-stage results and predicting the knockout stage bracket for the 2026 World Cup.

---

## 🚀 Performance & Risk Profile

### ⚡ Execution Performance (M-Series Silicon)
* **Training:** ~65 ms (100 trees, 280x4 dataset)
* **Batch Inference:** 1.39 ms (280 samples)
* **Latency:** ~5 microseconds per sample
* **Architecture:** True zero-copy NumPy buffer access via `pybind11::array_t`.

### 📈 Backtest Performance & Quant Risk Metrics
Evaluating the model value-betting strategy (betting on outcomes with $>5\%$ edge vs. bookmaker vig & sentiment bias) on **2,773 matches** post-2022:
* **Betting Win Rate:** `53.95%`
* **Betting ROI:** `+15.59%`
* **Betting PnL:** **`+432.26 units`**
* **Max Drawdown:** **`27.24 units`** (from a 100-unit bankroll)
* **Annualized Sharpe Ratio:** **`4.0640`** (daily returns annualized)

---

## 📊 Backtest Tear Sheet Visualizations

Generate these charts dynamically by running:
```bash
python research/generate_charts.py
```

### Cumulative Strategy PnL ("The Money Shot")
Plots our value-betting strategy PnL over time against a baseline strategy that always bets 1 unit on the bookmaker's favorite. 

![Cumulative PnL](research/cumulative_pnl.png)

### Probability Calibration Curve
Plots predicted probabilities bucketed in 10% increments against actual outcome frequencies, proving probability calibration.

![Probability Calibration Curve](research/calibration_curve.png)

---

## 🔒 Deterministic Reproducibility
Random Forests are stochastic, but our engine is fully deterministic. We have hardcoded a fixed seed (`1337`) in both our C++ tree generator and our Python scripts. **Results are fully deterministic and reproducible via fixed RNG seeds.**

---

## 🧪 CI/CD (GitHub Actions)
The repository includes a GitHub Actions workflow (`.github/workflows/build.yml`) that automatically installs dependencies (Python, CMake, pybind11, NumPy, pandas) and compiles the C++ library on an Ubuntu runner on every push or PR. This ensures that the engine compiles cleanly, stays portable, and remains production-ready.

---

## 📦 Quick Start (Docker)
The easiest way to build the project and run the World Cup prediction engine inside a Docker container:
```bash
docker build -t rf-engine .
docker run --rm rf-engine
```

## 🐍 Python API
```python
import numpy as np
import rf_cpp # Compiled C++ engine

# Features: [Home_Elo, Away_Elo, Home_Rest_Days, Away_Rest_Days, Is_Neutral_Venue]
X = np.ascontiguousarray([[2100.0, 2000.0, 5.0, 6.0, 1.0]], dtype=np.float64)

# Initialize: (num_trees, max_depth, min_samples_split, feature_fraction)
model = rf_cpp.RandomForest(100, 10, 6, 0.8)

# Train natively (zero-copy)
model.train(X_train, y_train)

# Zero-copy memory buffer batch predictions
probs = model.predict_batch(X) # [p_home, p_draw, p_away]
```

---

## 🛠️ Compilation & Local Setup

To compile the high-performance C++ module locally using your Python virtual environment interpreter:

1. Ensure your virtual environment packages are installed.
2. Configure CMake passing the target Python executable path:
   ```bash
   cmake -S cpp_core -B build -DPython_EXECUTABLE=./venv/bin/python
   ```
3. Compile the Release shared library (which will output to the root directory as a `.so` module):
   ```bash
   cmake --build build --config Release
   ```

---

## 🔒 Memory Safety & Input Validation Layer

During the C++ execution bindings integration, we implemented a robust verification layer inside the `pybind11` wrapper ([bindings.cpp](file:///Users/negru/CLionProjects/sem2/random_forest/cpp_core/bindings.cpp)):
* **Dimensionality Verification**: Confirms that data matrix `X` is exactly 2D and targets `y` are 1D.
* **Length Alignment**: Asserts that `X.shape[0] == y.shape[0]` (matching number of samples/labels) before training.
* **Type Constraints**: Rejects non-compatible layouts.

This validation layer prevents out-of-bounds memory accesses and protects the C++ engine from segmentation faults due to mismatched NumPy configurations in Python.

