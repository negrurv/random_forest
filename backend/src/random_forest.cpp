#include "../include/random_forest.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

TreeNode::TreeNode() 
    : is_leaf(false), split_feature_idx(-1), split_threshold(0.0), 
      prediction_value(0.0), left(nullptr), right(nullptr) {}

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : root_(nullptr), max_depth_(max_depth), min_samples_split_(min_samples_split) {}

// UPDATED: Now accepts the bootstrap indices from the Forest, and the feature fraction
void DecisionTree::train(const std::vector<double>& X, 
                         const std::vector<double>& y, 
                         const std::vector<int>& bootstrap_indices, 
                         int num_features,
                         double feature_fraction) {
    root_ = build_tree(X, y, bootstrap_indices, num_features, feature_fraction, 0);
}

std::unique_ptr<TreeNode> DecisionTree::build_tree(const std::vector<double>& X, 
                                                   const std::vector<double>& y, 
                                                   const std::vector<int>& sample_indices, 
                                                   int num_features, 
                                                   double feature_fraction,
                                                   int current_depth) {
    // 1. Check leaf conditions
    if (current_depth >= max_depth_ || static_cast<int>(sample_indices.size()) < min_samples_split_) {
        auto node = std::make_unique<TreeNode>();
        node->is_leaf = true;
        double sum = 0.0;
        for (int idx : sample_indices) sum += y[idx];
        node->prediction_value = sum / sample_indices.size();
        return node;
    }

    // 2. Select a random subset of features FOR THIS NODE specifically
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<int> feature_indices(num_features);
    for (int i = 0; i < num_features; ++i) feature_indices[i] = i;
    std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    
    int num_features_to_check = std::max(1, static_cast<int>(feature_fraction * num_features));
    feature_indices.resize(num_features_to_check);

    // 3. Setup variables
    int best_feature = -1;
    double best_threshold = 0.0;
    double best_score = -1.0; 

    // Calculate the total sum of y for this node ONCE
    double total_sum = 0.0;
    for (int idx : sample_indices) {
        total_sum += y[idx];
    }
    
    // THE FIX: Define current_mean here so the leaf node can use it later
    double current_mean = total_sum / sample_indices.size();

    // 4. Find the best split using the Running Sum optimization
    for (int f : feature_indices) {
        
        std::vector<std::pair<double, int>> feature_vals;
        feature_vals.reserve(sample_indices.size());
        for (int idx : sample_indices) {
            feature_vals.push_back({X[idx * num_features + f], idx});
        }

        std::sort(feature_vals.begin(), feature_vals.end());

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

            if (feature_vals[i].first == feature_vals[i + 1].first) {
                continue;
            }

            double score = (left_sum * left_sum / left_count) + 
                           (right_sum * right_sum / right_count);

            if (score > best_score) {
                best_score = score;
                best_feature = f;
                best_threshold = (feature_vals[i].first + feature_vals[i + 1].first) / 2.0;
            }
        }
    }

    // 5. If no valid split was found, make a leaf using the current_mean
    if (best_feature == -1) {
        auto node = std::make_unique<TreeNode>();
        node->is_leaf = true;
        node->prediction_value = current_mean; // Now this works perfectly!
        return node;
    }

    // 6. Otherwise, split the data and recurse
    auto node = std::make_unique<TreeNode>();
    node->is_leaf = false;
    node->split_feature_idx = best_feature;
    node->split_threshold = best_threshold;

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

// -------------------------------------------------------------

RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, double feature_fraction)
    : num_trees_(num_trees), feature_fraction_(feature_fraction) {
    trees_.reserve(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        trees_.emplace_back(max_depth, min_samples_split);
    }
}

void RandomForest::train(const std::vector<double>& X, 
                         const std::vector<double>& y, 
                         int num_samples, 
                         int num_features) {
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::cout << "Training " << num_trees_ << " trees...\n";

    for (auto& tree : trees_) {
        // 1. Bootstrap sampling (draw N samples WITH replacement)
        std::vector<int> bootstrap_indices(num_samples);
        std::uniform_int_distribution<> dis(0, num_samples - 1);
        
        for (int i = 0; i < num_samples; ++i) {
            bootstrap_indices[i] = dis(gen);
        }

        // 2. Train the tree passing ONLY the bootstrap indices
        tree.train(X, y, bootstrap_indices, num_features, feature_fraction_);
    }
}

double RandomForest::predict(const std::vector<double>& sample_x) const {
    double sum = 0.0;
    for (const auto& tree : trees_) {
        sum += tree.predict(sample_x);
    }
    return sum / trees_.size();
}

std::vector<double> RandomForest::predict_batch(const std::vector<double>& X, 
                                                int num_samples, 
                                                int num_features) const {
    std::vector<double> predictions(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        std::vector<double> sample_x(num_features);
        for (int j = 0; j < num_features; ++j) {
            sample_x[j] = X[i * num_features + j];
        }
        predictions[i] = predict(sample_x);
    }
    return predictions;
}