# C++ Random Forest Engine

A high-performance Random Forest implementation built from the ground up in **C++** with a **zero-copy Python interface** (`pybind11`). Designed specifically to bypass Python ML library overhead for sub-millisecond real-time batch inference.

### 🚀 Performance (M-Series Silicon)
* **Training:** ~65 ms (100 trees, 280x4 dataset)
* **Batch Inference:** 1.39 ms (280 samples)
* **Latency:** ~5 microseconds per sample
* **Architecture:** True zero-copy NumPy buffer access via `pybind11::array_t`.

### 📦 Quick Start (Docker)
The easiest way to reproduce the benchmarks without configuring C++ or CMake locally:

```bash
git clone [https://github.com/negrurv/random_forest.git](https://github.com/negrurv/random_forest.git)
cd random_forest
docker build -t rf-engine .
docker run --rm rf-engine
```

### 🐍 Python API

```python
import numpy as np
import rf_cpp # Compiled C++ engine

X, y = np.random.rand(280, 4), np.random.rand(280)

# Initialize: (num_trees, max_depth, min_samples_split, feature_fraction)
model = rf_cpp.RandomForest(100, 10, 2, 1.0)

# Train natively
model.train(X.flatten().tolist(), y.tolist(), 280, 4)

# Zero-copy memory buffer inference
predictions = model.predict_batch(X) 
```
