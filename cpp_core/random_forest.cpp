#include "random_forest.hpp"
#include <algorithm>
#include <random>
#include <cmath>
#include <iostream>

// DecisionTree constructor
DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : max_depth_(max_depth), min_samples_split_(min_samples_split) {}

// Train decision tree using raw pointers
void DecisionTree::train(const double* X, 
                         const double* y, 
                         const std::vector<int>& bootstrap_indices, 
                         int num_samples,
                         int num_features,
                         double feature_fraction) {
    tree_nodes_.clear();
    
    // Compute global class frequencies for parent_probs fallback
    double global_probs[3] = {0.0, 0.0, 0.0};
    if (num_samples > 0) {
        double counts[3] = {0.0, 0.0, 0.0};
        for (int i = 0; i < num_samples; ++i) {
            double target = y[i];
            if (target == 1.0) counts[0] += 1.0;
            else if (target == 0.5) counts[1] += 1.0;
            else if (target == 0.0) counts[2] += 1.0;
        }
        global_probs[0] = counts[0] / num_samples;
        global_probs[1] = counts[1] / num_samples;
        global_probs[2] = counts[2] / num_samples;
    } else {
        global_probs[0] = 0.33;
        global_probs[1] = 0.33;
        global_probs[2] = 0.33;
    }

    build_tree(X, y, bootstrap_indices, num_samples, num_features, feature_fraction, 0, global_probs);
}

// build_tree recursive construction with array-backed nodes
int DecisionTree::build_tree(const double* X, 
                             const double* y, 
                             const std::vector<int>& sample_indices, 
                             int num_samples,
                             int num_features, 
                             double feature_fraction, 
                             int current_depth,
                             const double* parent_probs) {
    
    // Create new node slot
    int node_idx = tree_nodes_.size();
    tree_nodes_.push_back(TreeNode());
    
    if (sample_indices.empty()) {
        tree_nodes_[node_idx].is_leaf = true;
        tree_nodes_[node_idx].split_feature_idx = -1;
        tree_nodes_[node_idx].split_threshold = 0.0;
        tree_nodes_[node_idx].prediction_probs[0] = parent_probs[0];
        tree_nodes_[node_idx].prediction_probs[1] = parent_probs[1];
        tree_nodes_[node_idx].prediction_probs[2] = parent_probs[2];
        tree_nodes_[node_idx].children[0] = -1;
        tree_nodes_[node_idx].children[1] = -1;
        return node_idx;
    }

    // Compute current node frequencies
    double current_probs[3] = {0.0, 0.0, 0.0};
    double counts[3] = {0.0, 0.0, 0.0};
    for (int idx : sample_indices) {
        double val = y[idx];
        if (val == 1.0) counts[0] += 1.0;
        else if (val == 0.5) counts[1] += 1.0;
        else if (val == 0.0) counts[2] += 1.0;
    }
    double total = sample_indices.size();
    current_probs[0] = counts[0] / total;
    current_probs[1] = counts[1] / total;
    current_probs[2] = counts[2] / total;

    // Base case: check depth or size limits
    if (current_depth >= max_depth_ || static_cast<int>(sample_indices.size()) < min_samples_split_) {
        tree_nodes_[node_idx].is_leaf = true;
        tree_nodes_[node_idx].split_feature_idx = -1;
        tree_nodes_[node_idx].split_threshold = 0.0;
        tree_nodes_[node_idx].prediction_probs[0] = current_probs[0];
        tree_nodes_[node_idx].prediction_probs[1] = current_probs[1];
        tree_nodes_[node_idx].prediction_probs[2] = current_probs[2];
        tree_nodes_[node_idx].children[0] = -1;
        tree_nodes_[node_idx].children[1] = -1;
        return node_idx;
    }

    // Deterministic random subset of features for reproducibility
    std::mt19937 gen(1337 + node_idx);
    std::vector<int> feature_indices(num_features);
    for (int i = 0; i < num_features; ++i) feature_indices[i] = i;
    std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    
    int num_features_to_check = std::max(1, static_cast<int>(feature_fraction * num_features));
    feature_indices.resize(num_features_to_check);

    int best_feature = -1;
    double best_threshold = 0.0;
    double best_score = -1.0;

    // Search for best split using Gini impurity reduction (maximized via Sum of Squares counts)
    for (int f : feature_indices) {
        std::vector<std::pair<double, int>> feature_vals;
        feature_vals.reserve(sample_indices.size());
        for (int idx : sample_indices) {
            feature_vals.push_back({X[idx * num_features + f], idx});
        }
        std::sort(feature_vals.begin(), feature_vals.end());

        // Track class counts on left and right sides
        double left_counts[3] = {0.0, 0.0, 0.0};
        double right_counts[3] = {counts[0], counts[1], counts[2]};
        
        int left_count = 0;
        int right_count = sample_indices.size();

        for (size_t i = 0; i < feature_vals.size() - 1; ++i) {
            int idx = feature_vals[i].second;
            double target = y[idx];
            int label = 2; // Default is Away Win (0.0)
            if (target == 1.0) label = 0;
            else if (target == 0.5) label = 1;

            left_counts[label] += 1.0;
            right_counts[label] -= 1.0;
            left_count++;
            right_count--;

            if (feature_vals[i].first == feature_vals[i + 1].first) {
                continue;
            }

            // Gini Score: sum_k(C_Lk^2)/N_L + sum_k(C_Rk^2)/N_R
            double left_sum_sq = (left_counts[0]*left_counts[0] + left_counts[1]*left_counts[1] + left_counts[2]*left_counts[2]);
            double right_sum_sq = (right_counts[0]*right_counts[0] + right_counts[1]*right_counts[1] + right_counts[2]*right_counts[2]);
            
            double score = (left_sum_sq / left_count) + (right_sum_sq / right_count);

            if (score > best_score) {
                best_score = score;
                best_feature = f;
                best_threshold = (feature_vals[i].first + feature_vals[i + 1].first) / 2.0;
            }
        }
    }

    if (best_feature == -1) {
        tree_nodes_[node_idx].is_leaf = true;
        tree_nodes_[node_idx].split_feature_idx = -1;
        tree_nodes_[node_idx].split_threshold = 0.0;
        tree_nodes_[node_idx].prediction_probs[0] = current_probs[0];
        tree_nodes_[node_idx].prediction_probs[1] = current_probs[1];
        tree_nodes_[node_idx].prediction_probs[2] = current_probs[2];
        tree_nodes_[node_idx].children[0] = -1;
        tree_nodes_[node_idx].children[1] = -1;
        return node_idx;
    }

    // Partition
    std::vector<int> left_indices, right_indices;
    for (int idx : sample_indices) {
        if (X[idx * num_features + best_feature] <= best_threshold) {
            left_indices.push_back(idx);
        } else {
            right_indices.push_back(idx);
        }
    }

    // Recurse (Note: tree_nodes_ indices can change sizes, so we save child indices rather than pointer references)
    int left_child = build_tree(X, y, left_indices, num_samples, num_features, feature_fraction, current_depth + 1, current_probs);
    int right_child = build_tree(X, y, right_indices, num_samples, num_features, feature_fraction, current_depth + 1, current_probs);

    // Update node details
    tree_nodes_[node_idx].is_leaf = false;
    tree_nodes_[node_idx].split_feature_idx = best_feature;
    tree_nodes_[node_idx].split_threshold = best_threshold;
    tree_nodes_[node_idx].prediction_probs[0] = current_probs[0];
    tree_nodes_[node_idx].prediction_probs[1] = current_probs[1];
    tree_nodes_[node_idx].prediction_probs[2] = current_probs[2];
    tree_nodes_[node_idx].children[0] = right_child; // index for cond == 0
    tree_nodes_[node_idx].children[1] = left_child;  // index for cond == 1

    return node_idx;
}

// Branchless traversal using children[cond] index mapping
void DecisionTree::predict_probs(const double* sample_ptr, double* out_probs) const {
    if (tree_nodes_.empty()) {
        out_probs[0] = 0.33;
        out_probs[1] = 0.33;
        out_probs[2] = 0.33;
        return;
    }

    int current_idx = 0;
    while (true) {
        const auto& node = tree_nodes_[current_idx];
        if (node.is_leaf) {
            out_probs[0] = node.prediction_probs[0];
            out_probs[1] = node.prediction_probs[1];
            out_probs[2] = node.prediction_probs[2];
            return;
        }

        // Branchless check
        bool cond = (sample_ptr[node.split_feature_idx] <= node.split_threshold);
        current_idx = node.children[cond];
    }
}


// RandomForest implementation
RandomForest::RandomForest(int num_trees, int max_depth, int min_samples_split, double feature_fraction)
    : num_trees_(num_trees), max_depth_(max_depth), min_samples_split_(min_samples_split), feature_fraction_(feature_fraction) {
    trees_.reserve(num_trees);
    for (int i = 0; i < num_trees; ++i) {
        trees_.emplace_back(max_depth, min_samples_split);
    }
}

// Train ensemble on raw pointer data
void RandomForest::train_from_ptr(const double* X_ptr, const double* y_ptr, int num_samples, int num_features) {
    // Deterministic random generator for reproducibility
    std::mt19937 gen(1337);
    
    std::cout << "Training " << num_trees_ << " classification trees...\n";

    for (auto& tree : trees_) {
        // Bootstrap samples
        std::vector<int> bootstrap_indices(num_samples);
        std::uniform_int_distribution<> dis(0, num_samples - 1);
        for (int i = 0; i < num_samples; ++i) {
            bootstrap_indices[i] = dis(gen);
        }
        tree.train(X_ptr, y_ptr, bootstrap_indices, num_samples, num_features, feature_fraction_);
    }
}

// Predict probability distributions in a zero-copy batch write
void RandomForest::predict_batch_optimized(const double* X_ptr, int num_samples, int num_features, double* out_ptr) const {
    for (int i = 0; i < num_samples; ++i) {
        const double* sample_ptr = X_ptr + (i * num_features);
        double avg_probs[3] = {0.0, 0.0, 0.0};

        for (const auto& tree : trees_) {
            double tree_probs[3] = {0.0, 0.0, 0.0};
            tree.predict_probs(sample_ptr, tree_probs);
            avg_probs[0] += tree_probs[0];
            avg_probs[1] += tree_probs[1];
            avg_probs[2] += tree_probs[2];
        }

        out_ptr[i * 3 + 0] = avg_probs[0] / trees_.size();
        out_ptr[i * 3 + 1] = avg_probs[1] / trees_.size();
        out_ptr[i * 3 + 2] = avg_probs[2] / trees_.size();
    }
}
