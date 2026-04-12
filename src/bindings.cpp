#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include "../include/random_forest.hpp"

namespace py = pybind11;

// PYBIND11_MODULE: Defines Python extension module.
// High-level logic: Bridges C++ RandomForest to Python for efficient ML.
// Performance: Direct C++ calls avoid Python overhead.
// Non-obvious: Module name must match compiled library for import.
PYBIND11_MODULE(rf_cpp, m) {
    m.doc() = "C++ Random Forest Library for Predictions";

    // py::class_: Exposes RandomForest class to Python.
    // High-level logic: Allows Python instantiation and method calls.
    // Performance: Method forwarding is near-zero cost.
    py::class_<RandomForest>(m, "RandomForest")
        // Constructor: Initializes RandomForest with params.
        // High-level logic: Sets up ensemble hyperparameters.
        // Performance: Allocation happens in C++ for speed.
        .def(py::init<int, int, int, double>(), 
             py::arg("num_trees"), py::arg("max_depth"), 
             py::arg("min_samples_split"), py::arg("feature_fraction"))
        
        // Train: Fits model on provided data.
        // High-level logic: Builds trees with bootstrapping and subsampling.
        // Performance: Training is compute-intensive but done once.
        .def("train", &RandomForest::train, 
             "Train the random forest",
             py::arg("X"), py::arg("y"), py::arg("num_samples"), py::arg("num_features"))
        
        // Predict: Single-sample inference.
        // High-level logic: Traverses trees for prediction.
        // Performance: Fast O(depth) per tree.
        .def("predict", &RandomForest::predict, py::arg("sample_x"))
             
        // predict_batch: Batch prediction with numpy input.
        // High-level logic: Processes multiple samples efficiently.
        // Performance: Zero-copy via buffer protocol; ideal for large batches.
        // Non-obvious: Lambda captures numpy array shape and pointer.
        .def("predict_batch", [](RandomForest &self, py::array_t<double> X) {
            py::buffer_info buf = X.request();
            
            const double* ptr = static_cast<double*>(buf.ptr);
            int num_samples = buf.shape[0];
            int num_features = buf.shape[1];
            
            return self.predict_batch_optimized(ptr, num_samples, num_features);
        });
}