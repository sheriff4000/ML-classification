from numpy.random import default_rng
import numpy as np

from dataset import Dataset
from tree import TreeNode, SplitCondition
from viz import TreeViz

# TreeViz(tree_start_node).render()
# this command can be used to render an image of a tree when given its start node

K_FOLDS = 10
HYPERPARAM_COMPLEXITY_PREFER_PRUNING_WHEN_SAME_ACCURACY = 850


class Model:
    # the constructor is given the dataset and random generator and initilises the Model with these and a shuffled version of the dataset
    def __init__(self, dataset: Dataset, rng):
        self.dataset = dataset
        self.rng = rng
        self.shuffled_dataset = shuffle_dataset(
            self.dataset, random_generator=rng)
    
    # runs k-fold cross validation learning and evaluation and returns the calcuated TypeEvaluationMetrics
    def run(self):
        # initilise evaluation metrics for the run instance
        all_metrics = TypeEvaluationMetrics()

        # outer loop 1 iteration per fold 
        for i in range(K_FOLDS):
            # initilise lists for this fold for storing the trees and acuracys created with each option of validation data
            unpruned_trees = []
            pruned_trees = []
            pruned_trees_accs = []
            
            # seperate this folds test data from the rest of the data
            test_data, remaining_data = holdout_fold(self.shuffled_dataset, K_FOLDS, i)
            # learn on the remaining data to get the no-pruning tree
            tree_start_node_no_prune, _ = \
                self.decision_tree_learning(Dataset(remaining_data), 0)
            #call to update the no-pruning metrics with the evaluation of the tree 
            eval_and_update(tree_start_node_no_prune,
                            test_data, all_metrics.no_pruning)

            validation_idxs = set()
            # inner loop with k-1 iterations 1 per possible validation data for current fold
            for j in range(K_FOLDS - 1):
                validation_idx = (i+j) % 9
                validation_idxs.add(validation_idx)

                # seperate the validation data and training data
                validation_data, training_data = holdout_fold(
                    remaining_data, K_FOLDS - 1, validation_idx)
                # learn on the training data to get the pre pruned tree
                tree_start_node, _ = self.decision_tree_learning(
                    Dataset(training_data), 0)
                # save a copy of the tree before it is pruned
                unpruned_trees.append(copy_tree(tree_start_node))
                # prune the tree
                pruned_tree = pruning(
                    validation_data, tree_start_node, tree_start_node)
                # save the pruned tree and the acuracy of the pruned tree
                pruned_trees.append(pruned_tree)
                pruned_trees_accs.append(evaluate_acc(validation_data, pruned_tree))

            assert len(validation_idxs) == K_FOLDS - 1
            
            # find the tree with the best accuracy of the inner loop trees
            best_acc = 0
            for i, acc in enumerate(pruned_trees_accs):
                if acc > best_acc:
                    best_pruned_tree = pruned_trees[i]
                    unpruned_best_tree = unpruned_trees[i]
                    best_acc = acc
            # evaluate the best pruned tree and its unpruned version and update the corresponding metrics
            eval_and_update(unpruned_best_tree, test_data,
                            all_metrics.pre_pruning)
            eval_and_update(best_pruned_tree, test_data,
                            all_metrics.post_pruning)

        return all_metrics
    
    # taking the dataset at a node find the best way to split it to maximise information gain, returning the split_condition and each of the corosponding subsets
    def find_split(self, node_data: Dataset) -> tuple[SplitCondition, Dataset, Dataset]:
        max_information_gain = 0
        dataset_entropy = node_data.label_entropy()
        # one iteration per atribute to check everyone
        for i in range(node_data.attributes().shape[1]):
            attribute = node_data.dataset[:, i]
            # range j between the range of the atribute to test spiltting on every possible value 
            for j in range(int(np.amin(attribute)), int(np.amax(attribute)+1)):
                # split the dataset on less than j
                subset_left = Dataset(node_data.dataset[attribute < j])
                subset_right = Dataset(node_data.dataset[attribute >= j])

                # calculate the entropy of the subsets
                subsets_entropy = ((len(subset_left) / len(node_data)) * subset_left.label_entropy()) + \
                    ((len(subset_right) / len(node_data)) * subset_right.label_entropy())

                # calculate the information gain 
                information_gain = dataset_entropy - subsets_entropy

                if information_gain < max_information_gain:
                    continue
                # if the gain is higher than the current best update the best
                max_information_gain = information_gain
                best_split_condition = SplitCondition(i, j)
                best_subset_left = subset_left
                best_subset_right = subset_right
        
        assert max_information_gain != 0
        
        return best_split_condition, best_subset_left, best_subset_right

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
    if depth == 0:
        return (l_depth + r_depth) / (l_num_of_leafs + r_num_of_leafs)
    else:
        return l_depth + r_depth, l_num_of_leafs + r_num_of_leafs


def clip_tree(dataset, node: TreeNode, top_node: TreeNode):
    if node.left.is_leaf() and node.right.is_leaf() and node.left.leaf == node.right.leaf:
        # Clip the tree as we have the same value in both leaves.
        node.left.pruning = True
        return node.left

    LEFT_LEAF_ACC, RIGHT_LEAF_ACC, ORIGINAL_ACC = 0, 1, 2
    accuracies = [0] * 3

    # Evaluate existing accuracy
    accuracies[ORIGINAL_ACC], _, original_nodes_touched = evaluate(
        dataset, top_node, confusion_matrix_enabled=False)

    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruning]
    node.left, node.right, node.condition, node.pruning = None, None, None, False

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
        node.left, node.right, node.leaf, node.condition, node.pruning = \
            tmp_node[0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
        assert node.left is not None or node.leaf is not None
        assert node.pruning is False
        return node

    # Prune with the most optimal leaf
    node.leaf = tmp_node[best_acc_arg].leaf
    node.pruning = True
    return node


def pruning(dataset, node: TreeNode, top_node: TreeNode):
    # checks if the current node is a leaf node
    if node.is_leaf():
        # immeditately return the node as no traversing downwards can be done
        return node

    # if the current node has a leaf node on both right and left branches
    if node.left.is_leaf() and node.right.is_leaf():
        # attempts to prune the tree
        return clip_tree(dataset, node, top_node)

    # traversing down both left and right branches
    node.left = pruning(dataset, node.left, top_node)
    node.right = pruning(dataset, node.right, top_node)

    # checks if self.pruning on node is True to re-prune the node
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

# shuffles dataset before split is done so data is in a random order


def shuffle_dataset(dataset, random_generator=default_rng()):
    return dataset[
        random_generator.permutation(len(dataset))
    ]


def holdout_fold(dataset, num_splits, holdout_idx):
    subsets = np.split(dataset, num_splits)
    holdout = subsets[holdout_idx]
    remaining_data = [subsets[i]
                      for i in range(len(subsets)) if i != holdout_idx]
    return holdout, np.concatenate(remaining_data)


class EvalMetrics:
    def __init__(self):
        self.nodes_touched = 0


def evaluate_acc(test_db, tree_start):
    return evaluate(test_db, tree_start, confusion_matrix_enabled=False)[0]


def evaluate(test_db, tree_start, confusion_matrix_enabled=True):
    test_data = Dataset(test_db)
    y_classified = []

    assert len(test_data) != 0

    accesses = 0
    for row in test_data.attributes():
        current_node = tree_start

        while not current_node.is_leaf():
            accesses += 1
            assert current_node.left is not None, [current_node]
            assert current_node.right is not None
            if row[current_node.condition.attribute] < current_node.condition.less_than:
                current_node = current_node.left
            else:
                current_node = current_node.right

        assert current_node.is_leaf()
        y_classified.append(current_node.leaf)

    y_classified_nparray = np.array(y_classified)
    accuracy = np.sum(y_classified_nparray ==
                      test_data.labels())/len(y_classified)

    confusion_matrix = None
    if confusion_matrix_enabled:
        confusion_matrix = np.zeros((4, 4))
        for i in range(len(y_classified_nparray)):
            confusion_matrix[int(test_data.labels()[i])-1,
                             int(y_classified_nparray[i])-1] += 1
    return (accuracy, confusion_matrix, accesses)


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
                if i == j:
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
            F1s[i] = (2*class_precisions[i]*class_recalls[i]) / \
                (class_precisions[i]+class_recalls[i])

        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

        return confusion_matrix, class_precisions, class_recalls, F1s, accuracy

    def print(self, prefix):
        print(f"[{prefix}] Max Tree Depth Avg:", self.avg(self.max_depth))
        print(f"[{prefix}] Mean Tree Depth avg", self.avg(self.mean_depth))
        print(f"[{prefix}] Ten-fold Accuracy:", self.avg(self.accuracy))

        confusion_matrix, class_precisions, class_recalls, F1s, _ = self.confusion_metrics()
        print(f"[{prefix}] Per-class Metrics:")
        for i in range(class_precisions.shape[0]):
            print("\tClass ", i+1, " Precision = ", class_precisions[i])
            print("\tClass ", i+1, " Recall = ", class_recalls[i])
            print("\tClass ", i+1, " F1 = ", F1s[i])
        print(f"[{prefix}] Confusion Matrix:")

        str_cf = str(confusion_matrix).split('\n')
        for line in str_cf:
            print(f"\t{line}")


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
    accuracy, confusion_matrix, _ = evaluate(test_data, tree)
    metrics.update(
        accuracy,
        max_depth_finder(tree, 0),
        mean_depth_finder(tree, 0),
        confusion_matrix
    )
    return confusion_matrix