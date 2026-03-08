#include <pybind11/pybind11.h>

// 1. The actual C++ logic
int predict_match(float home_xg, float away_xg) {
    // A very dumb algorithm: whoever is expected to score more, wins.
    if (home_xg > away_xg) {
        return 1;  // Home Win
    }
    return -1; // Away Win
}

// 2. The Pybind11 Wrapper (The Bridge)
// "football_engine" is the name Python will use to import this
PYBIND11_MODULE(football_engine, m) {
    m.doc() = "C++ Football Prediction Engine"; // Optional docstring
    
    // Bind the C++ function to Python
    m.def("predict", &predict_match, "Predicts match outcome based on xG");
}
