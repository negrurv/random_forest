// random_forest.hpp
#pragma once

#include <vector>
#include <memory>

// ---------------------------------------------------------
// 1. Node Structure
// ---------------------------------------------------------
struct TreeNode {
    bool is_leaf;
    int split_feature_idx;       
    double split_threshold;      
    double prediction_value;     

    std::unique_ptr<TreeNode> left;  
    std::unique_ptr<TreeNode> right;

    TreeNode();
    ~TreeNode() = default;
};

// ---------------------------------------------------------
// 2. Decision Tree Class
// ---------------------------------------------------------
class DecisionTree {
public:
    DecisionTree(int max_depth, int min_samples_split);
    ~DecisionTree() = default;

    // --- ADD THESE TWO LINES to enable moving in std::vector ---
    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;

    void train(const std::vector<double>& X, 
               const std::vector<double>& y, 
               const std::vector<int>& bootstrap_indices, 
               int num_features,
               double feature_fraction);

    double predict(const std::vector<double>& sample_x) const;

private:
    std::unique_ptr<TreeNode> root_;
    int max_depth_;
    int min_samples_split_;

    std::unique_ptr<TreeNode> build_tree(const std::vector<double>& X, 
                                         const std::vector<double>& y, 
                                         const std::vector<int>& sample_indices, 
                                         int num_features, 
                                         double feature_fraction,
                                         int current_depth);
};

// ---------------------------------------------------------
// 3. Random Forest Class
// ---------------------------------------------------------
class RandomForest {
public:
    RandomForest(int num_trees, int max_depth, int min_samples_split, double feature_fraction);
    ~RandomForest() = default;

    // The signature for the forest stays exactly the same
    void train(const std::vector<double>& X, 
               const std::vector<double>& y, 
               int num_samples, 
               int num_features);

    double predict(const std::vector<double>& sample_x) const;

    std::vector<double> predict_batch(const std::vector<double>& X, 
                                      int num_samples, 
                                      int num_features) const;

private:
    int num_trees_;
    // REMOVED max_depth_ and min_samples_split_ from here!
    double feature_fraction_; 
    
    std::vector<DecisionTree> trees_;
};