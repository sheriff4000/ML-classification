# MAKE SURE

from turtle import pos, shape
from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt

import unit_test

from dataset import Dataset
from tree import TreeNode, SplitCondition
from viz import TreeViz

HYPERPARAM_COMPLEXITY_PREFER_PRUNING_WHEN_SAME_ACCURACY = 850

class Model:
    def __init__(self, dataset: Dataset):
        pass

    def train(self):
        pass

    def find_split(self, training_set: Dataset) -> tuple[SplitCondition, Dataset, Dataset]:
        max_gain = 0
        entropy = training_set.label_entropy()
        for i in range(training_set.attributes().shape[1]):
            attribute = training_set.dataset[:, i]
            for j in range(int(np.amin(attribute)), int(np.amax(attribute)+1)):
                a = Dataset(training_set.dataset[attribute < j])
                b = Dataset(training_set.dataset[attribute >= j])

                remainder = ((len(a) / len(training_set)) * a.label_entropy()) + \
                    ((len(b) / len(training_set)) * b.label_entropy())

                gain = entropy - remainder

                if gain < max_gain:
                    continue

                max_gain = gain
                split_idx = i
                split_val = j
                out_a = a
                out_b = b
        
        assert max_gain != 0
        return SplitCondition(split_idx, split_val), out_a, out_b

    def decision_tree_learning(self, training_dataset: Dataset, depth=0):
        assert len(training_dataset) != 0
        
        labels = training_dataset.unique_labels()
        if len(labels[0]) == 1:
            return TreeNode(None, None, labels[0][0], None, False), depth


        split_cond, l_dataset, r_dataset = self.find_split(training_dataset)
        l_branch, l_depth = self.decision_tree_learning(l_dataset, depth + 1)
        r_branch, r_depth = self.decision_tree_learning(r_dataset, depth + 1)
        node = TreeNode(l_branch, r_branch, None, split_cond, False)
        return (node, max(l_depth, r_depth))

def max_depth_finder(node: TreeNode, depth):
    if node.is_leaf():
        return depth
    l_depth = max_depth_finder(node.left, depth + 1)
    r_depth = max_depth_finder(node.right, depth + 1)
    return max(l_depth, r_depth)

def mean_depth_finder(node: TreeNode, depth):
    if node.is_leaf():
        return depth, 1
    l_depth, l_num_of_leafs = mean_depth_finder(node.left, depth + 1)
    r_depth, r_num_of_leafs = mean_depth_finder(node.right, depth + 1)
    if depth == 0 :
        return (l_depth + r_depth) / (l_num_of_leafs + r_num_of_leafs)
    else:
        return l_depth + r_depth, l_num_of_leafs + r_num_of_leafs

def clip_tree(dataset, node: TreeNode, top_node: TreeNode):
    if node.left.is_leaf() and node.right.is_leaf() and node.left.leaf == node.right.leaf:
        node.left.pruning = True
        return node.left

    accuracies = [0] * 3
    
    # Evaluate existing accuracy
    existing_metrics = EvalMetrics()
    accuracies[2] = evaluate(dataset, top_node, False, metrics=existing_metrics)
    
    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruning]
    node.left = None
    node.right = None
    node.condition = None
    node.pruning = False

    # Evaluate accuracy with `node` replaced with left leaf
    node.leaf = tmp_node[0].leaf
    accuracies[0] = evaluate(dataset, top_node, False)

    # Evaluate accuracy with `node` replaced with right leaf
    node.leaf = tmp_node[1].leaf
    accuracies[1] = evaluate(dataset, top_node, False)

    best_acc_arg, best_acc = 100, 0
    for i, acc in enumerate(accuracies):
        if acc < best_acc:
            continue
        if acc <= best_acc and existing_metrics.nodes_touched > HYPERPARAM_COMPLEXITY_PREFER_PRUNING_WHEN_SAME_ACCURACY:
            continue
        best_acc_arg, best_acc = i, acc

    assert best_acc_arg >= 0 and best_acc_arg <= 2
    assert tmp_node[0].leaf is not None
    assert tmp_node[1].leaf is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    if best_acc_arg == 2:
        node.left, node.right, node.leaf, node.condition, node.pruning = tmp_node[
            0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
        assert node.left is not None or node.leaf is not None
        assert node.pruning is False
        return node
    
    # Prune with the most optimal leafg
    node.leaf = tmp_node[best_acc_arg].leaf
    node.pruning = True
    return node

def pruning(dataset, node: TreeNode, top_node: TreeNode):
    if node.is_leaf():
        return node

    if node.left.is_leaf() and node.right.is_leaf():
        # Fruit node
        return clip_tree(dataset, node, top_node)

    node.left = pruning(dataset, node.left, top_node)
    node.right = pruning(dataset, node.right, top_node)
    if node.left.pruning or node.right.pruning:
        node.left.pruning = False
        node.right.pruning = False
        node = pruning(dataset, node, top_node)
    return node

def copy_tree(node: TreeNode):
    if node is None:
        return None

    new_tree = node.shallow_copy()    
    new_tree.left = copy_tree(node.left)
    new_tree.right = copy_tree(node.right)
    return new_tree

# split 80/10/10
def shuffle_dataset(dataset, random_generator=default_rng()):
    shuffled_indecies = random_generator.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indecies]

    return shuffled_dataset

def split_dataset(dataset, test_idx, validation_offset):
    subsets = np.split(dataset, 10)
    
    validation_idx = (test_idx+validation_offset) % 10
    assert test_idx != validation_idx
    
    test_data = subsets[test_idx]
    validation_data = subsets[validation_idx]

    remaining_data = [subsets[i] for i in range(len(subsets)) if i != test_idx and i != validation_idx]
    training_data = np.concatenate(remaining_data)
    return training_data, test_data, validation_data

class EvalMetrics:
    def __init__(self):
        self.nodes_touched = 0

def evaluate(test_db, tree_start, confusion_matrix_enabled, metrics:EvalMetrics=None):
    test_data = Dataset(test_db)
    y_classified = []
    
    assert len(test_data) != 0
    
    counter = 0
    for row in test_data.attributes():
        current_node = tree_start
        
        while not current_node.is_leaf():
            counter += 1
            assert current_node.left is not None, [current_node]
            assert current_node.right is not None
            if row[current_node.condition.attribute] < current_node.condition.less_than:
                current_node = current_node.left
            else:
                current_node = current_node.right

        assert current_node.is_leaf()
        y_classified.append(current_node.leaf)

    if metrics:
        metrics.nodes_touched = counter

    y_classified_nparray = np.array(y_classified)
    accuracy = np.sum(y_classified_nparray == test_data.labels())/len(y_classified)

    if confusion_matrix_enabled:
        confusion_matrix = np.zeros((4, 4))
        for i in range(len(y_classified_nparray)):
            confusion_matrix[int(test_data.labels()[i])-1, int(y_classified_nparray[i])-1] += 1
        return (accuracy, confusion_matrix)

    return (accuracy)

class EvaluationMetrics:
    def __init__(self):
        self.accuracy = []
        self.max_depth = []
        self.mean_depth = []
        self.confusion_matrix = []
    
    def update(self, accuracy, max_depth, mean_depth, confusion_matrix):
        self.accuracy.append(accuracy)
        self.max_depth.append(max_depth)
        self.mean_depth.append(mean_depth)
        self.confusion_matrix.append(confusion_matrix) 
        assert np.sum(confusion_matrix) == 200
    
    def avg(self, vals):
        return sum(vals) / len(vals)
    
    def sum_confusion_matrices(self):
        if len(self.confusion_matrix) == 0:
            return None
        res = self.confusion_matrix[0]
        for mat in self.confusion_matrix[1:]:
            res += mat
        return res
    
    def confusion_metrics(self):
        confusion_matrix = self.sum_confusion_matrices()
        
        TPs = np.zeros(4)
        FPs = np.zeros(4)
        FNs = np.zeros(4)
        
        for i in range(4):
            current_row = confusion_matrix[i, :]
            current_col = confusion_matrix[:, i]
            for j in range(4):
                if i == j :
                    TPs[i] += confusion_matrix[i, j]
                else:
                    FNs[i] += current_row[j]
                    FPs[i] += current_col[j]
        
        class_precisions = np.zeros(4)
        class_recalls = np.zeros(4)

        for i in range(4):
            class_precisions[i] = TPs[i]/(TPs[i]+FPs[i])
            class_recalls[i] = TPs[i]/(TPs[i]+FNs[i])

        F1s = np.zeros(4)

        for i in range(4):
            F1s[i] = (2*class_precisions[i]*class_recalls[i])/(class_precisions[i]+class_recalls[i])

        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

        return confusion_matrix, class_precisions, class_recalls, F1s, accuracy    

    def print(self, prefix):
        print(f"[{prefix}] Max Tree Depth Avg:", self.avg(self.max_depth))
        print(f"[{prefix}] Mean Tree Depth avg", self.avg(self.mean_depth))
        print(f"[{prefix}] Ten-fold Accuracy:", self.avg(self.accuracy))
        
        confusion_matrix, class_precisions, class_recalls, F1s, _ = self.confusion_metrics()
        print(f"[{prefix}] Per-class Metrics:", self.avg(self.accuracy))
        for i in range(class_precisions.shape[0]):
            print("\tClass ", i+1, " Precision = ", class_precisions[i])
            print("\tClass ", i+1, " Recall = ", class_recalls[i])
            print("\tClass ", i+1, " F1 = ", F1s[i])
        print(f"[{prefix}] Confusion Matrix \n", confusion_matrix)

class TypeEvaluationMetrics:
    def __init__(self):
        self.no_pruning = EvaluationMetrics()
        self.pre_pruning = EvaluationMetrics()
        self.post_pruning = EvaluationMetrics()
    
    def print(self):
        self.no_pruning.print("No Pruning")
        self.pre_pruning.print("Pre-Pruning")
        self.post_pruning.print("Post-Pruning")

def eval_and_update(tree: TreeNode, test_data: Dataset, metrics: EvaluationMetrics):
    accuracy, confusion_matrix = evaluate(test_data, tree, True)
    metrics.update(
        accuracy, 
        max_depth_finder(tree, 0),
        mean_depth_finder(tree, 0),
        confusion_matrix
    )
    return confusion_matrix

def machine_learn(dataset, rg=default_rng()):
    shuffled_dataset = shuffle_dataset(dataset, random_generator=rg)
    
    all_metrics = TypeEvaluationMetrics()
    model = Model(None)
    
    for i in range(10):
        unpruned_trees = []
        pruned_trees = []
        tree_accs = []
        for j in range(1, 10):
            training_data, test_data, validation_data = split_dataset(shuffled_dataset, i, j)
            if j == 1:
                training_data_no_prune = np.concatenate((training_data, validation_data))
                tree_start_node_no_prune, no_prune_max_depth = model.decision_tree_learning(Dataset(training_data_no_prune), 0)
                no_prune_eval = evaluate(test_data, tree_start_node_no_prune, True)
                
                all_metrics.no_pruning.update(
                    no_prune_eval[0], 
                    no_prune_max_depth,
                    mean_depth_finder(tree_start_node_no_prune, 0),
                    no_prune_eval[1]
                )
                
                TreeViz(tree_start_node_no_prune).render()
            tree_start_node = model.decision_tree_learning(Dataset(training_data), 0)
            
            unpruned_trees.append(copy_tree(tree_start_node[0]))
            pruned_tree = pruning(validation_data, tree_start_node[0], tree_start_node[0])
            pruned_trees.append(pruned_tree)
            tree_accs.append(evaluate(validation_data, pruned_tree, False))
    
        best_acc = 0
        for i, acc in enumerate(tree_accs):
            if acc > best_acc:
                best_pruned_tree = pruned_trees[i]
                unpruned_best_tree = unpruned_trees[i]
                best_acc = acc
        
        eval_and_update(unpruned_best_tree, test_data, all_metrics.pre_pruning)
        eval_and_update(best_pruned_tree, test_data, all_metrics.post_pruning)
    
    all_metrics.print()

    return

unit_test.Test()

clean_data = np.loadtxt("intro2ML-coursework1/wifi_db/clean_dataset.txt", delimiter="\t")
noisy_data = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")

seed = 1030
rg = default_rng(seed)

print("CLEAN DATA")
machine_learn(clean_data, rg=rg)

print("\n\nNOISY DATA")
machine_learn(noisy_data, rg=rg)
