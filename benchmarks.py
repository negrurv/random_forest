import time
import numpy as np
import rf_cpp

def run_benchmarks():
    print("--- Random Forest C++ Engine Benchmark ---")
    
    num_samples = 280
    num_features = 4
    X = np.random.rand(num_samples, num_features).astype(np.float64)
    y = np.random.rand(num_samples).astype(np.float64)

    model = rf_cpp.RandomForest(100, 10, 2, 1.0)

    start_train = time.time()
    model.train(X.flatten().tolist(), y.tolist(), num_samples, num_features)
    train_time = (time.time() - start_train) * 1000
    print(f"Training Time (100 trees): {train_time:.2f} ms")

    start_infer = time.time()
    predictions = model.predict_batch(X)
    infer_time = (time.time() - start_infer) * 1000
    print(f"Batch Inference Time ({num_samples} samples): {infer_time:.4f} ms")
    print(f"Average Per-Sample Latency: {(infer_time / num_samples):.4f} ms")
    

if __name__ == "__main__":
    run_benchmarks()