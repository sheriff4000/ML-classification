from numpy.random import Generator
import numpy as np

from dataset import Dataset, holdout_fold, shuffle_dataset
from evaluation import eval_and_update, evaluate, evaluate_acc, TypeEvaluationMetrics
from tree import TreeNode, SplitCondition, copy_tree


# The following can be used to render an image of a tree when given its start node
# from viz import TreeViz
# TreeViz(tree_start_node).render()


K_FOLDS = 10
HYPERPARAM_COMPLEXITY_PREFER_PRUNING_WHEN_SAME_ACCURACY = 850


class Model:
    def __init__(self, dataset: Dataset, rng: Generator):
        """Constructs a shuffled dataset from the provided set and returns a new 
        untrained model.

        Args:
            dataset (Dataset): The raw dataset.
            rng (Generator): A random number generator to control randomness.
        """
        self.dataset = dataset
        self.rng = rng
        self.shuffled_dataset = shuffle_dataset(
            self.dataset, random_generator=rng)
    

    def run(self) -> TypeEvaluationMetrics:
        """Performs K_FOLDS cross validation learning and evaluation and returns the
        calcuated TypeEvaluationMetrics.

        Returns:
            TypeEvaluationMetrics: the metrics for generated all trees.
        """
        all_metrics = TypeEvaluationMetrics()

        for i in range(K_FOLDS):
            # Perform the k-folds inside this loop
            unpruned_trees = []
            pruned_trees = []
            pruned_trees_accs = []
            
            # Separate this fold's test data from the rest of the data
            test_data, remaining_data = holdout_fold(self.shuffled_dataset, K_FOLDS, i)
            
            # Train on the remaining data to get the no-pruning tree
            tree_start_node_no_prune, _ = \
                self.decision_tree_learning(Dataset(remaining_data), 0)
            
            # Update the no-pruning metrics with the evaluation of the tree 
            eval_and_update(tree_start_node_no_prune, test_data, all_metrics.no_pruning)

            validation_idxs = set()
            for j in range(K_FOLDS - 1):
                # Loop through each possible validation data set for the current
                # fold.
                validation_idx = (i+j) % 9
                validation_idxs.add(validation_idx)

                # Separate the validation data and training data
                validation_data, training_data = holdout_fold(remaining_data, K_FOLDS - 1, validation_idx)
                # Train on the training data to get the pre-pruned tree
                tree_start_node, _ = self.decision_tree_learning(Dataset(training_data), 0)
                # Save a copy of the tree before it is pruned
                unpruned_trees.append(copy_tree(tree_start_node))
                pruned_tree = prune_tree(validation_data, tree_start_node, tree_start_node)
                
                # Save the pruned tree and the accuracy of the pruned tree
                pruned_trees.append(pruned_tree)
                pruned_trees_accs.append(evaluate_acc(validation_data, pruned_tree))
            
            # Check we actually cycled through all possible validation sets
            assert len(validation_idxs) == K_FOLDS - 1
            
            # Find the best pruned tree
            best_acc = 0
            for i, acc in enumerate(pruned_trees_accs):
                if acc > best_acc:
                    best_pruned_tree = pruned_trees[i]
                    unpruned_best_tree = unpruned_trees[i]
                    best_acc = acc
            
            # Evaluate the best pruned tree and its unpruned version and update the corresponding metrics
            eval_and_update(unpruned_best_tree, test_data, all_metrics.pre_pruning)
            eval_and_update(best_pruned_tree, test_data, all_metrics.post_pruning)
        
        return all_metrics
    

    def find_split(self, node_data: Dataset) -> tuple[SplitCondition, Dataset, Dataset]:
        """Finds the best way to split the dataset to maximise information gain,
        returning the split_condition and each of the corresponding subsets.

        Args:
            node_data (Dataset): The dataset to split on.

        Returns:
            tuple[SplitCondition, Dataset, Dataset]: A tuple of the condition to
            split on and the left and right datasets.
        """
        max_information_gain = 0
        dataset_entropy = node_data.label_entropy()
        
        
        for i in range(node_data.attributes().shape[1]):
            # Loop through all attributes to test them all
            attribute = node_data.dataset[:, i]
            
            for j in range(int(np.amin(attribute)), int(np.amax(attribute)+1)):
                # Range between the min and max of the attribute to test
                # splitting on every possible value.
                subset_left = Dataset(node_data.dataset[attribute < j])
                subset_right = Dataset(node_data.dataset[attribute >= j])

                # Calculate the entropy of the subsets
                subsets_entropy = ((len(subset_left) / len(node_data)) * subset_left.label_entropy()) + \
                    ((len(subset_right) / len(node_data)) * subset_right.label_entropy())

                # Calculate the information gain
                information_gain = dataset_entropy - subsets_entropy
                if information_gain < max_information_gain:
                    # If this split's gain is lower, ignore it
                    continue
                
                # Store the best gain
                max_information_gain = information_gain
                best_split_condition = SplitCondition(i, j)
                best_subset_left = subset_left
                best_subset_right = subset_right
        
        assert max_information_gain != 0
        
        return best_split_condition, best_subset_left, best_subset_right
    

    def decision_tree_learning(self, training_dataset: Dataset, depth=0) -> tuple[TreeNode, int]:
        """Recursively creates a binary classification tree using the training_dataset.

        Args:
            training_dataset (Dataset): The dataset to train on.
            depth (int, optional): A parameter used during recursion; should be 0 on the initial call.

        Returns:
            tuple[TreeNode, int]: A tuple of the decision tree and it's max depth.
        """
        assert len(training_dataset) != 0

        labels = training_dataset.unique_labels()
        if len(labels[0]) == 1:
            return TreeNode(None, None, labels[0][0], None, False), depth

        split_cond, l_dataset, r_dataset = self.find_split(training_dataset)
        l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
        node = TreeNode(l_branch, r_branch, None, split_cond, False)
        return (node, max(l_depth, r_depth))


def clip_tree(dataset, node: TreeNode, top_node: TreeNode) -> TreeNode:
    """Evaluates the each case of replacing a node with one of its leaves, or
    leaving it untouched and picks the resulting tree with the best accuracy.

    Args:
        dataset (_type_): The validation set to prune with.
        node (TreeNode): The node to consider pruning.
        top_node (TreeNode): The root of the tree to perform a final evaluation after pruning.

    Returns:
        TreeNode: the node to replace the originally passed in node with.
    """
    if node.left.is_leaf() and node.right.is_leaf() and node.left.leaf == node.right.leaf:
        # Clip the tree as we have the same value in both leaves.
        node.left.pruned = True
        return node.left

    LEFT_LEAF_ACC, RIGHT_LEAF_ACC, ORIGINAL_ACC = 0, 1, 2
    accuracies = [0] * 3

    # Evaluate existing accuracy
    accuracies[ORIGINAL_ACC], _, original_nodes_touched = evaluate(
        dataset, top_node, confusion_matrix_enabled=False)

    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruned]
    node.left, node.right, node.condition, node.pruned = None, None, None, False

    # Evaluate accuracy with `node` replaced with left leaf
    node.leaf = tmp_node[LEFT_LEAF_ACC].leaf
    accuracies[LEFT_LEAF_ACC] = evaluate_acc(dataset, top_node)

    # Evaluate accuracy with `node` replaced with right leaf
    node.leaf = tmp_node[RIGHT_LEAF_ACC].leaf
    accuracies[RIGHT_LEAF_ACC] = evaluate_acc(dataset, top_node)

    best_acc_arg, best_acc = 100, 0
    for i, acc in enumerate(accuracies):
        if acc < best_acc:
            continue
        if acc <= best_acc and original_nodes_touched > HYPERPARAM_COMPLEXITY_PREFER_PRUNING_WHEN_SAME_ACCURACY:
            # Since the original node is at the end of the list, by only
            # allowing this to be chosen when the original tree is complex and
            # has the same accuracy as the pruned version, we limit un-necessary
            # pruning of non-complex trees. This improves our post-pruning
            # accuracy for clean data, at a negligible cost for noisy data.
            continue
        best_acc_arg, best_acc = i, acc

    assert best_acc_arg >= 0 and best_acc_arg <= 2
    assert tmp_node[0].leaf is not None
    assert tmp_node[1].leaf is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    if best_acc_arg == ORIGINAL_ACC:
        node.left, node.right, node.leaf, node.condition, node.pruned = \
            tmp_node[0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
        assert node.left is not None or node.leaf is not None
        assert node.pruned is False
        return node

    # Prune with the most optimal leaf
    node.leaf = tmp_node[best_acc_arg].leaf
    node.pruned = True
    return node


def prune_tree(dataset, node: TreeNode, top_node: TreeNode):
    """Recursively prunes the given node, returning the pruned tree.

    Args:
        dataset (_type_): The validation dataset to prune with.
        node (TreeNode): The node to recursively prune.
        top_node (TreeNode): The root of the tree to perform a final evaluation after pruning. 

    Returns:
        _type_: _description_
    """
    if node.is_leaf():
        # Can't recuse any further than this.
        return node
    
    if node.left.is_leaf() and node.right.is_leaf():
        # If the current node has a leaf node on both right and left branches,
        # attempt to prune the tree.
        return clip_tree(dataset, node, top_node)

    # Traverse down both left and right branches
    node.left = prune_tree(dataset, node.left, top_node)
    node.right = prune_tree(dataset, node.right, top_node)

    # Checks if the child nodes were pruned and then re-prunes the root if so.
    if node.left.pruned or node.right.pruned:
        node.left.pruned = False
        node.right.pruned = False
        node = prune_tree(dataset, node, top_node)
    return node
