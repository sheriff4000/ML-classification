import numpy as np

from dataset import Dataset
from tree import TreeNode, max_depth_finder, mean_depth_finder


class EvaluationMetrics:
    def __init__(self):
        """Creates a new metrics object used for evaluating tree performance.
        """
        self.accuracy = []
        self.max_depth = []
        self.mean_depth = []
        self.confusion_matrix = []
    
    def update(self, accuracy, max_depth, mean_depth, confusion_matrix):
        """Appends the provided values to an internal list of tree metrics.
        """
        self.accuracy.append(accuracy)
        self.max_depth.append(max_depth)
        self.mean_depth.append(mean_depth)
        self.confusion_matrix.append(confusion_matrix)
        assert np.sum(confusion_matrix) == 200


    def avg(self, vals):
        """Returns the average of the vals parameter
        """
        return sum(vals) / len(vals)


    def sum_confusion_matrices(self):
        """Returns the confusion matrix created by summing all of the stored
        confusion matrices.
        """
        if len(self.confusion_matrix) == 0:
            return None
        res = self.confusion_matrix[0]
        for mat in self.confusion_matrix[1:]:
            res += mat
        return res


    def confusion_metrics(self):
        """Calulates the sum of all the stored confusion matrices and then computes
        the accuracy, precision, recall and f1 score for the summed matrix to get
        the 10-fold cross validated average for these metrics.
        """
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
            class_precisions[i] = TPs[i] / (TPs[i]+FPs[i])
            class_recalls[i] = TPs[i] / (TPs[i]+FNs[i])

        F1s = np.zeros(4)

        for i in range(4):
            F1s[i] = (2*class_precisions[i]*class_recalls[i]) / (class_precisions[i]+class_recalls[i])

        accuracy = np.trace(confusion_matrix)/np.sum(confusion_matrix)

        return confusion_matrix, class_precisions, class_recalls, F1s, accuracy


    def print(self, prefix):
        """Prints the k-fold average for all metrics and sum of all the confusion
        matrices from each fold.

        Args:
            prefix (_type_): a prefix to be used before printing; usually the name of the metrics.
        """
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


def evaluate(test_db, tree_start: TreeNode, confusion_matrix_enabled=True):
    """Evaluates how well the tree_start classifies the test_db dataset, returning the
    classification accuracy and the confusion matrix. Returns the accuracy, a confusion
    matrix and an estimate of the complexity of the tree.

    Args:
        test_db (_type_): The test dataset to evaluate the tree with.
        tree_start (TreeNode): The tree to evaluate
        confusion_matrix_enabled (bool, optional): Whether to produce a confusion matrix.
    """
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
    accuracy = np.sum(y_classified_nparray == test_data.labels()) / len(y_classified)
    
    confusion_matrix = None
    if confusion_matrix_enabled:
        confusion_matrix = np.zeros((4, 4))
        for i in range(len(y_classified_nparray)):
            confusion_matrix[int(test_data.labels()[i])-1, int(y_classified_nparray[i])-1] += 1
    return (accuracy, confusion_matrix, accesses)


def evaluate_acc(test_db, tree_start: TreeNode):
    """Evaluates the provided tree on the test_db dataset and returns the accuracy.
    """
    return evaluate(test_db, tree_start, confusion_matrix_enabled=False)[0]


def eval_and_update(tree: TreeNode, test_data: Dataset, metrics: EvaluationMetrics):
    """Evaluates the performance of the provided tree on the test_data and then
    update the metrics instance with the results. Returns the confusion matrix.

    Args:
        tree (TreeNode): The tree to evaluate.
        test_data (Dataset): The test data to evaluate the tree on.
        metrics (EvaluationMetrics): The metrics instance to update.
    """
    accuracy, confusion_matrix, _ = evaluate(test_data, tree)
    metrics.update(
        accuracy,
        max_depth_finder(tree, 0),
        mean_depth_finder(tree, 0),
        confusion_matrix
    )
    return confusion_matrix
