// backend/src/bindings.cpp
#include <pybind11/pybind11.h>
#include <pybind11/stl.h> 
#include "../include/random_forest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rf_cpp, m) {
    m.doc() = "C++ Random Forest Library for Football Predictions";

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int, double>(), 
             py::arg("num_trees"), py::arg("max_depth"), 
             py::arg("min_samples_split"), py::arg("feature_fraction"))
        
        .def("train", &RandomForest::train, 
             "Train the random forest",
             py::arg("X"), py::arg("y"), py::arg("num_samples"), py::arg("num_features"))
        
        .def("predict", &RandomForest::predict, 
             "Predict a single sample",
             py::arg("sample_x"))
             
        .def("predict_batch", &RandomForest::predict_batch, 
             "Predict a batch of samples",
             py::arg("X"), py::arg("num_samples"), py::arg("num_features"));
}
