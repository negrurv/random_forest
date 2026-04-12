#include "../include/random_forest.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

// TreeNode constructor: Initializes a node as non-leaf with default values.
// Performance note: Using unique_ptr for children ensures automatic memory management,
// avoiding leaks in recursive tree structures.
TreeNode::TreeNode() 
    : is_leaf(false), split_feature_idx(-1), split_threshold(0.0), 
      prediction_value(0.0), left(nullptr), right(nullptr) {}

// DecisionTree constructor: Sets hyperparameters for tree growth.
// High-level logic: max_depth prevents overfitting by limiting tree size;
// min_samples_split ensures splits only occur on sufficiently large subsets.
DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : root_(nullptr), max_depth_(max_depth), min_samples_split_(min_samples_split) {}

// Train method: Builds the decision tree using bootstrapped samples and feature subsampling.
// High-level logic: Bootstrap indices introduce diversity; feature_fraction controls randomness.
// Performance: Shuffling features each time adds overhead but ensures unbiased subsampling.
void DecisionTree::train(const std::vector<double>& X, 
                         const std::vector<double>& y, 
                         const std::vector<int>& bootstrap_indices, 
                         int num_features,
                         double feature_fraction) {
    root_ = build_tree(X, y, bootstrap_indices, num_features, feature_fraction, 0);
}

// build_tree: Recursively constructs the tree by finding optimal splits.
// High-level logic: Uses variance reduction as the splitting criterion for regression.
// Non-obvious: Skips splits if feature values are identical to avoid redundant thresholds.
// Performance: Sorting feature values per feature is O(n log n) per feature, dominant cost.
std::unique_ptr<TreeNode> DecisionTree::build_tree(const std::vector<double>& X, 
                                                   const std::vector<double>& y, 
                                                   const std::vector<int>& sample_indices, 
                                                   int num_features, 
                                                   double feature_fraction,
                                                   int current_depth) {
    // Base case: Create leaf if depth limit or insufficient samples.
    if (current_depth >= max_depth_ || static_cast<int>(sample_indices.size()) < min_samples_split_) {
        auto node = std::make_unique<TreeNode>();
        node->is_leaf = true;
        double sum = 0.0;
        for (int idx : sample_indices) sum += y[idx];
        node->prediction_value = sum / sample_indices.size();
        return node;
    }

    // Randomly select subset of features to consider for split.
    // Performance: Shuffling is O(num_features), efficient for small num_features.
    // Note: RNG is reinitialized per node; could be optimized by reusing a generator
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> feature_indices(num_features);
    for (int i = 0; i < num_features; ++i) feature_indices[i] = i;
    std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    
    int num_features_to_check = std::max(1, static_cast<int>(feature_fraction * num_features));
    feature_indices.resize(num_features_to_check);
    // Tradeoff: randomness reduces overfitting but may weaken individual splits

    // Initialize best split tracking.
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_score = -1.0; 

    // Precompute total sum for efficient incremental variance calculation.
    // Non-obvious: Using sum of squares proxy for variance reduction avoids full recomputation.
    double total_sum = 0.0;
    for (int idx : sample_indices) {
        total_sum += y[idx];
    }
    
    double current_mean = total_sum / sample_indices.size();

    // Evaluate each candidate feature for best split.
    for (int f : feature_indices) {
        // X is stored as a flattened row-major matrix: [num_samples x num_features]
        // This layout improves cache locality compared to vector<vector<double>>
        // Collect and sort feature values with indices for threshold selection.
        std::vector<std::pair<double, int>> feature_vals;
        feature_vals.reserve(sample_indices.size());
        for (int idx : sample_indices) {
            feature_vals.push_back({X[idx * num_features + f], idx});
        }
        std::sort(feature_vals.begin(), feature_vals.end());

        // Incremental computation of left/right sums for efficiency.
        // Performance: O(n) per feature after sorting, avoids recomputing sums from scratch.
        double left_sum = 0.0;
        double right_sum = total_sum;
        int left_count = 0;
        int right_count = sample_indices.size();

        for (size_t i = 0; i < feature_vals.size() - 1; ++i) {
            double current_y = y[feature_vals[i].second];
            left_sum += current_y;
            right_sum -= current_y;
            left_count++;
            right_count--;

            // Skip if no split possible (identical values).
            if (feature_vals[i].first == feature_vals[i + 1].first) {
                continue;
            }

            // Score: Sum of squared means (proxy for variance reduction).
            // Equivalent to minimizing mean squared error (MSE) for regression trees
            // High-level logic: Higher score indicates better split.
            double score = (left_sum * left_sum / left_count) + 
                           (right_sum * right_sum / right_count);

            if (score > best_score) {
                best_score = score;
                best_feature = f;
                best_threshold = (feature_vals[i].first + feature_vals[i + 1].first) / 2.0;
            }
        }
    }

    // If no valid split found, create leaf.
    if (best_feature == -1) {
        auto node = std::make_unique<TreeNode>();
        node->is_leaf = true;
        node->prediction_value = current_mean; 
        return node;
    }

    // Create internal node and recurse on children.
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->split_feature_idx = best_feature;
    node->split_threshold = best_threshold;

    // Partition samples based on best split.
    std::vector<int> left_indices, right_indices;
    for (int idx : sample_indices) {
        if (X[idx * num_features + best_feature] <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    node->left = build_tree(X, y, left_indices, num_features, feature_fraction, current_depth + 1);
    node->right = build_tree(X, y, right_indices, num_features, feature_fraction, current_depth + 1);

    return node;
}

// Predict: Traverses tree to leaf for prediction.
// Performance: O(depth) traversal, fast for inference.
double DecisionTree::predict(const std::vector<double>& sample_x) const {
    const TreeNode* node = root_.get();
    while (!node->is_leaf) {
        if (sample_x[node->split_feature_idx] <= node->split_threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->prediction_value;
}

// RandomForest constructor: Initializes ensemble of trees.
// High-level logic: num_trees increases robustness; feature_fraction adds diversity.
RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, double feature_fraction)
    : num_trees_(num_trees), feature_fraction_(feature_fraction) {
    trees_.reserve(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        trees_.emplace_back(max_depth, min_samples_split);
    }
}

// Train: Builds each tree with bootstrapped samples.
// Performance: Parallelizable across trees; bootstrapping adds O(num_samples) per tree.
// Potential optimization: trees can be trained in parallel (embarrassingly parallel workload)
void RandomForest::train(const std::vector<double>& X, 
                         const std::vector<double>& y, 
                         int num_samples, 
                         int num_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::cout << "Training " << num_trees_ << " trees...\n";

    for (auto& tree : trees_) {
        // Generate bootstrap sample indices.
        std::vector<int> bootstrap_indices(num_samples);
        std::uniform_int_distribution<> dis(0, num_samples - 1);
        
        for (int i = 0; i < num_samples; ++i) {
            bootstrap_indices[i] = dis(gen);
        }

        tree.train(X, y, bootstrap_indices, num_features, feature_fraction_);
    }
}

// Predict: Averages predictions across trees.
// High-level logic: Ensemble averaging reduces variance.
double RandomForest::predict(const std::vector<double>& sample_x) const {
    double sum = 0.0;
    for (const auto& tree : trees_) {
        sum += tree.predict(sample_x);
    }
    return sum / trees_.size();
}

// predict_batch: Vectorized batch prediction.
// Performance: Uses pointer-based predict for efficiency, avoids vector copies.
std::vector<double> RandomForest::predict_batch(const std::vector<double>& X, 
                                                int num_samples, 
                                                int num_features) const {
    std::vector<double> predictions(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        const double* row_ptr = &X[i * num_features];
        predictions[i] = predict_from_ptr(row_ptr); 
    }
    return predictions;
}

// predict_from_ptr: Pointer-based prediction for DecisionTree.
// Performance: Direct pointer access avoids bounds checking in vectors.
double DecisionTree::predict_from_ptr(const double* sample_ptr) const {
    const TreeNode* node = root_.get();
    while (!node->is_leaf) {
        if (sample_ptr[node->split_feature_idx] <= node->split_threshold) {
            node = node->left.get();
        } else {
            node = node->right.get();
        }
    }
    return node->prediction_value;
}

// predict_from_ptr: Aggregates predictions from trees using pointers.
// Performance: Minimizes memory allocations in batch scenarios.
double RandomForest::predict_from_ptr(const double* sample_ptr) const {
    double sum = 0.0;
    for (const auto& tree : trees_) {
        sum += tree.predict_from_ptr(sample_ptr);
    }
    return sum / trees_.size();
}

// predict_batch_optimized: Highly optimized batch prediction.
// Performance: Direct array access via pointers, ideal for large batches.
std::vector<double> RandomForest::predict_batch_optimized(const double* X_ptr, 
                                                          int num_samples, 
                                                          int num_features) const {
    std::vector<double> predictions(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        predictions[i] = predict_from_ptr(X_ptr + (i * num_features));
    }
    return predictions;
}