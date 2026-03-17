#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include "../include/random_forest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rf_cpp, m) {
    m.doc() = "C++ Random Forest Library for Predictions";

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int, double>(), 
             py::arg("num_trees"), py::arg("max_depth"), 
             py::arg("min_samples_split"), py::arg("feature_fraction"))
        
        .def("train", &RandomForest::train, 
             "Train the random forest",
             py::arg("X"), py::arg("y"), py::arg("num_samples"), py::arg("num_features"))
        
        .def("predict", &RandomForest::predict, py::arg("sample_x"))
             
        .def("predict_batch", [](RandomForest &self, py::array_t<double> X) {
            py::buffer_info buf = X.request();
            
            const double* ptr = static_cast<double*>(buf.ptr);
            int num_samples = buf.shape[0];
            int num_features = buf.shape[1];
            
            return self.predict_batch_optimized(ptr, num_samples, num_features);
        });
}