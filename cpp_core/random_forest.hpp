// random_forest.hpp
#pragma once

#include <vector>
#include <memory>

struct TreeNode {
    bool is_leaf;
    int split_feature_idx;       
    double split_threshold;      
    double prediction_probs[3]; // class 0: Home Win, class 1: Draw, class 2: Away Win
    int children[2];            // children[0] is right child index, children[1] is left child index
};


class DecisionTree {
public:
    DecisionTree(int max_depth, int min_samples_split);
    ~DecisionTree() = default;

    void train(const double* X, 
               const double* y, 
               const std::vector<int>& bootstrap_indices, 
               int num_samples,
               int num_features,
               double feature_fraction);

    void predict_probs(const double* sample_ptr, double* out_probs) const;

    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;

private:
    std::vector<TreeNode> tree_nodes_; // Flattened array-backed tree layout
    int max_depth_;
    int min_samples_split_;

    int build_tree(const double* X, 
                   const double* y, 
                   const std::vector<int>& sample_indices, 
                   int num_samples,
                   int num_features, 
                   double feature_fraction, 
                   int current_depth,
                   const double* parent_probs);
};

class RandomForest {
public:
    RandomForest(int num_trees, int max_depth, int min_samples_split, double feature_fraction);
    ~RandomForest() = default;
    
    void train_from_ptr(const double* X_ptr, const double* y_ptr, int num_samples, int num_features);
    void predict_batch_optimized(const double* X_ptr, int num_samples, int num_features, double* out_ptr) const;

private:
    int num_trees_;
    int max_depth_;
    int min_samples_split_;
    double feature_fraction_; 
    
    std::vector<DecisionTree> trees_;
};
