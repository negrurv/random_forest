C++ Random Forest with Python Bindings (random_forest)

This project implements a Random Forest machine learning model entirely from scratch in C++. The goal is to understand the inner workings of ensemble learning by manually building decision trees, bootstrap aggregating (bagging), and calculating Gini and variance impurity metrics without relying on external ML libraries.

The core engine is highly optimized for algorithmic efficiency and fast execution. It is capable of training 100-tree ensembles on 280x4 datasets in approximately 40ms, while achieving sub-millisecond latency for real-time batch inference.

To make the engine usable in standard data science workflows, the project includes a zero-copy Python interface built with pybind11. The repository also includes a Dockerized Python API and frontend (currently under active development) to serve the model as a web service.

Project Structure
backend/src/ – C++ source files containing the decision tree logic and pybind11 bindings
backend/include/ – C++ header files for the core engine
backend/api.py – Python API wrapper for the C++ model
frontend/ – Web interface for model deployment (WIP)
Dockerfile – Containerization setup for the full stack

Build and Run

Navigate to the backend directory and compile the C++ pybind11 module using CMake:
cd backend
mkdir build
cd build
cmake ..
make

Return to the root directory and run the test script to evaluate the model:
cd ../..
python3 test_rf.py
