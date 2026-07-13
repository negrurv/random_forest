#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h> 
#include "random_forest.hpp"

namespace py = pybind11;

PYBIND11_MODULE(rf_cpp, m) {
    m.doc() = "C++ Random Forest Library for Multi-class Match Predictions";

    py::class_<RandomForest>(m, "RandomForest")
        .def(py::init<int, int, int, double>(), 
             py::arg("num_trees"), py::arg("max_depth"), 
             py::arg("min_samples_split"), py::arg("feature_fraction"))
        
        .def("train", [](RandomForest &self, py::array_t<double> X, py::array_t<double> y) {
            py::buffer_info buf_X = X.request();
            py::buffer_info buf_y = y.request();
            
            if (buf_X.ndim != 2) {
                throw std::runtime_error("Input X must be a 2D numpy array");
            }
            if (buf_y.ndim != 1) {
                throw std::runtime_error("Input y must be a 1D numpy array");
            }
            if (buf_y.shape[0] != buf_X.shape[0]) {
                throw std::runtime_error("Input X and y must have the same number of samples (rows)");
            }
            
            const double* X_ptr = static_cast<double*>(buf_X.ptr);
            const double* y_ptr = static_cast<double*>(buf_y.ptr);
            
            int num_samples = buf_X.shape[0];
            int num_features = buf_X.shape[1];
            
            self.train_from_ptr(X_ptr, y_ptr, num_samples, num_features);
        }, "Train the random forest using C-contiguous NumPy arrays")
        
        .def("predict_batch", [](RandomForest &self, py::array_t<double> X) {
            py::buffer_info buf = X.request();
            
            if (buf.ndim != 2) {
                throw std::runtime_error("Input X must be a 2D numpy array");
            }
            
            const double* ptr = static_cast<double*>(buf.ptr);
            int num_samples = buf.shape[0];
            int num_features = buf.shape[1];
            
            // Allocate 2D NumPy array of shape [num_samples, 3] to write probabilities directly
            py::array_t<double> result({num_samples, 3});
            py::buffer_info res_buf = result.request();
            double* res_ptr = static_cast<double*>(res_buf.ptr);
            
            // Direct write to the numpy array's memory buffer - zero intermediate copy!
            self.predict_batch_optimized(ptr, num_samples, num_features, res_ptr);
            
            return result;
        }, "Predict match probabilities in a zero-copy handoff returning shape [num_samples, 3]");
}
