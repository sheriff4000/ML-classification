# MAKE SURE

from numpy.random import default_rng
import numpy as np
import matplotlib.pyplot as plt


class Test:
    def __init__(self) -> None:
        data = np.loadtxt(
            "intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
        set = Dataset(data)
        assert (len(set.unique_labels()[0]) == 4)
        assert (set.unique_labels()[0][0] == 1)


class TreeNode:
    def __init__(self, left, right, leaf, condition, pruned) -> None:
        self.left = left
        self.right = right
        self.leaf = leaf
        self.condition = condition
        self.pruning = pruned
        
        self.x = 0
        self.y = 0
        self.mod = 0
        self.thread = None
        
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def shallow_copy(self):
        new_node = TreeNode(self.left, self.right, self.leaf, self.condition, self.pruning)
        new_node.x = self.x
        new_node.y = self.y
        new_node.mod = self.mod
        new_node.thread = self.thread
        return new_node

class SplitCondition:
    def __init__(self, attribute, less_than) -> None:
        self.attribute = attribute
        self.less_than = less_than

    def __repr__(self):
        return f"[X{self.attribute} < {self.less_than}]"

class Dataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.LABEL_COL_INDEX = -1

    def attributes(self):
        return self.dataset[:, :self.LABEL_COL_INDEX]

    def labels(self):
        return self.dataset[:, self.LABEL_COL_INDEX]

    def unique_labels(self):
        return np.unique(self.labels(), return_counts=True)

    def find_entropy(self) -> float:
        attributes = self.attributes()
        labels = self.labels()

        # print(attributes)
        # print(labels)
        unique_labels, label_count = self.unique_labels()
        # print(unique_labels)
        # print(label_count)
        # print(self.dataset)

        # make sure stuffs not broken i guess
        assert (np.sum(label_count) == len(labels))
        entropy = 0
        for i in range(len(unique_labels)):
            prob = float(label_count[i]) / float(len(labels))
            entropy -= prob * np.log2(prob)
            # print(entropy)

        return entropy

    def __len__(self):
        return len(self.dataset)

class TreeViz:
    """
    Implements the Reingold-Tilford [1] Algorithm to render the given binary tree.
    
    [1] https://reingold.co/tidier-drawings.pdf
    
    The implementation takes inspiration from:
        - https://rachel53461.wordpress.com/2014/04/20/algorithm-for-drawing-trees/
        - https://llimllib.github.io/pymag-trees/
    """
    def __init__(self, tree: TreeNode):
        self.tree = tree

    def render(self):
        """Call this function to initialise and render the provided binary tree. This is the only function an external user should call.
        """
        self.init(self.tree)
        self.plot(self.tree)
        plt.axis('off')
        print("about to plot")
        plt.show()
        print("plotting")
    
    def plot(self, t: TreeNode, depth=0):
        """Plots the diagram of a pre-initialised tree.

        Args:
            t (TreeNode): The pre-initialised tree to plot.
            depth (int, optional): The current plotting depth (node level). Defaults to 0.
        """
        if t is None:
            return
        
        scale_y_by = 3
        
        circle_with_line = "-o"
        
        if t.left:
            plt.plot(np.array([t.x, t.left.x]), np.array([-t.y * scale_y_by, -t.left.y * scale_y_by]), circle_with_line)
        
        if t.right:
            plt.plot(np.array([t.x, t.right.x]), np.array([-t.y * scale_y_by, -t.right.y * scale_y_by]), circle_with_line)
        
        if not t.left and not t.right:
            # Display leaf value
            plt.text(t.x, -t.y * scale_y_by - 0.33, int(t.leaf), horizontalalignment='center', verticalalignment='top')
        else:
            # Display decision node
            bbox = dict(boxstyle='round', facecolor='wheat', alpha=0.95)
            plt.text(t.x, -t.y * scale_y_by, t.condition, size=9.0, horizontalalignment='center', verticalalignment='center', bbox=bbox)

        self.plot(t.left)
        self.plot(t.right)
    
    def init(self, tree: TreeNode, init_tree=True, curmod=0):
        """Recursively configures the tree positions if init_tree is True and then moves the
        sub-trees based on the newly calculated node mod values.
        
        Args:
            tree (TreeNode): The tree to initialise.
            init_tree (bool, optional): Calculates the initial x positions of the nodes. Defaults to True.
            curmod (int, optional): A rolling count of the mod value to add to the x coordinates of the nodes. Defaults to 0.
        """
        if init_tree:
            self.init_tree(tree)

        tree.x += curmod
        curmod += tree.mod
        if tree.left:
            self.init(tree.left, init_tree=False, curmod=curmod)
        if tree.right:
            self.init(tree.right, init_tree=False, curmod=curmod)
    
    def init_tree(self, tree: TreeNode, depth=0):
        """Recursively resets tree values to good defaults before computing the initial x positions of nodes.

        Args:
            tree (TreeNode): The tree to modify.
            depth (int, optional): The current computation depth (node level). Defaults to 0.
        """
        if not tree:
            return

        # Reset to defaults, so they can be computed correctly.
        tree.x = 0
        tree.y = depth
        tree.mod = 0
        tree.thread = None

        if tree.is_leaf():
            return

        # Post-order traversal of the tree
        self.init_tree(tree.left, depth+1)
        self.init_tree(tree.right, depth+1)
        tree.x = self.calc_parent_x(tree.left, tree.right)

    def calc_parent_x(self, left: TreeNode, right: TreeNode, 
                      left_root: TreeNode=None, right_root: TreeNode=None,
                      left_outer: TreeNode=None, right_outer: TreeNode=None,
                      move_by=0, left_x_offset=0, right_x_offset=0) -> float:
        """Implements the Reingold-Tilford algorithm to compute the x position of the left and right node's root.

        Args:
            left (TreeNode): The left node in the sub-tree.
            right (TreeNode): The right node in the sub-tree.
            left_root (TreeNode, optional): The original root of the left node before recursing. Defaults to None.
            right_root (TreeNode, optional): The original root of the right node before recursing. Defaults to None.
            left_outer (TreeNode, optional): The left contour node of the subtree. Defaults to None.
            right_outer (TreeNode, optional): The right contour node of the subtree. Defaults to None.
            move_by (int, optional): The minimum value to move the right subtree by. Defaults to 0.
            left_x_offset (int, optional): The left subtree offset, used to compute the actual left node x value. Defaults to 0.
            right_x_offset (int, optional): The right subtree offset, used to compute the actual right node x value. Defaults to 0.

        Returns:
            float: The midpoint of the left and right subtrees, which can then be updated on the parent. 
        """

        # Set the default values of the original left and right nodes as we will
        # need these after recursing.
        left_root = left if not left_root else left_root
        right_root = right if not right_root else right_root
        
        # Set the default values of the outer-most nodes from the left and right
        # branch respectively.
        left_outer = left if not left_outer else left_outer
        right_outer = right if not right_outer else right_outer
        
        # Compute the next inner and outer nodes for the left side of the tree and
        # the left contour respectively.
        next_left_outer = left_outer.thread if left_outer.thread else left_outer.left
        next_left_inner = left.thread if left.thread else left.right

        # Compute the next inner and outer nodes for the right side of the tree and
        # the right contour respectively.
        next_right_inner = right.thread if right.thread else right.left
        next_right_outer = right_outer.thread if right_outer.thread else right_outer.right
        
        # Compute the actual x coodinates of the left and right node (taking into
        # account the offset based on the parents).
        left_x = left.x + left_x_offset
        right_x = right.x + right_x_offset
        delta_x = left_x - right.x
        
        # Force some constant spacing between the tree
        min_delta = 10
        delta_x += min_delta
        
        # Make sure we always move by at least move_by (or delta_x if bigger).
        if move_by == 0 or delta_x > move_by:
            move_by = delta_x

        if next_left_inner and next_right_inner:
            # Recurse to continue computing the x offset at the bottom of the tree and
            # setup the "threads" before returning.
            left_x_offset += left.mod
            right_x_offset += right.mod
            return self.calc_parent_x(next_left_inner, next_right_inner, left_root, right_root, next_left_outer, next_right_outer, move_by, left_x_offset, right_x_offset)

        if next_right_inner:
            # Configure the "thread" to point to the contour, skipping later
            # computation.
            left_outer.thread = next_right_inner

        # Sub-trees are now initialised correctly, so we can set the sub-tree
        # root to it's correct location.
        right_root.mod += move_by
        right_root.x += move_by
        
        if next_left_inner:
            if not right_root.is_leaf():
                # If the right tree is not a leaf node, then shift it over by
                # the max change.
                right_x_offset += move_by

            # Move the outer contour.
            right_outer.mod = left_x_offset - right_x_offset

            # Configure the "thread" to point to the contour, skipping later
            # computation.
            right_outer.thread = next_left_inner

        midpoint = (left_root.x + right_root.x) / 2
        return midpoint

def find_split(training_set: Dataset):
    max_gain = 0
    entropy = training_set.find_entropy()
    for i in range(training_set.attributes().shape[1]):
        attribute = training_set.dataset[:, i]
        for j in range(int(np.amin(attribute)), int(np.amax(attribute)+1)):
            a = Dataset(training_set.dataset[attribute < j])
            b = Dataset(training_set.dataset[attribute >= j])

            remainder = ((len(a) / len(training_set)) * a.find_entropy()) + \
                ((len(b) / len(training_set)) * b.find_entropy())

            gain = entropy - remainder

            if gain >= max_gain:
                max_gain = gain
                split_idx = i
                split_val = j
                out_a = a
                out_b = b

    assert max_gain != 0

    return split_idx, split_val, out_a.dataset, out_b.dataset


def decision_tree_learning(training_dataset, depth):
    data = Dataset(training_dataset)
    assert len(data) != 0
    labels = data.unique_labels()
    if len(labels[0]) == 1:
        # TODO: bug: what about if 0?
        return TreeNode(None, None, labels[0][0], None, False), depth
    attribute, split_val, l_dataset, r_dataset = find_split(data)

    split_cond = SplitCondition(attribute, split_val)

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    node = TreeNode(l_branch, r_branch, None, split_cond, False)

    return (node, max(l_depth, r_depth))


def clip_tree(dataset, node: TreeNode, top_node: TreeNode):
    if node.left.is_leaf() and node.right.is_leaf() and node.left.leaf == node.right.leaf:
        node.left.pruning = True
        return node.left

    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruning]
    accuracies = np.ndarray((3, 1))
    accuracies[2] = evaluate(dataset, top_node, False)
    node.leaf = tmp_node[0].leaf
    node.left = None
    node.right = None
    node.condition = None
    node.pruning = False
    left_node = [node.left, node.right,
                 node.leaf, node.condition, node.pruning]
    accuracies[0] = evaluate(dataset, top_node, False)

    node.leaf = tmp_node[1].leaf
    node.left = None
    node.right = None
    node.condition = None
    node.pruning = False
    right_node = [node.left, node.right,
                  node.leaf, node.condition, node.pruning]
    accuracies[1] = evaluate(dataset, top_node, False)

    best_acc_arg = accuracies.argmax()
    assert left_node[0] is not None or left_node[2] is not None
    assert right_node[0] is not None or right_node[2] is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    if best_acc_arg == 2:
        node.left, node.right, node.leaf, node.condition, node.pruning = tmp_node[
            0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
        assert node.left is not None or node.leaf is not None
        assert node.pruning is False
    elif best_acc_arg == 0:
        node.left, node.right, node.leaf, node.condition, node.pruning = left_node[
            0], left_node[1], left_node[2], left_node[3], True
        assert node.pruning is True
    elif best_acc_arg == 1:
        node.left, node.right, node.leaf, node.condition, node.pruning = right_node[
            0], right_node[1], right_node[2], right_node[3], True
        assert node.pruning is True
    else:
        print("Failed to match on best_acc_arg")
        exit(1)

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

def find_prune(node: TreeNode, validation_data, best_acc):
    prune_tree = copy_tree(node)
    test_prune = pruning(validation_data, prune_tree, prune_tree)

    current_acc = evaluate(validation_data, test_prune, False)

    if current_acc > best_acc:
        return test_prune, current_acc
    else:
        return node, best_acc

# split 80/10/10
def shuffle_dataset(dataset, random_generator=default_rng()):
    shuffled_indecies = random_generator.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indecies]

    return shuffled_dataset


# def generate_data_sets(dataset):
#     splits = 10
#     subsets = np.split(dataset, splits)
    
#     for i in range(splits):
#         test_data = subsets[i]
#         validation_data = subsets[i + 1]
        


def split_dataset(dataset, test_idx, validation_offset):
    subsets = np.split(dataset, 10)
    
    validation_idx = (test_idx+validation_offset) % 10
    assert test_idx != validation_idx
    
    test_data = subsets[test_idx]
    validation_data = subsets[validation_idx]

    remaining_data = [subsets[i] for i in range(len(subsets)) if i != test_idx and i != validation_idx]
    training_data = np.concatenate(remaining_data)
    return training_data, test_data, validation_data

def evaluate(test_db, tree_start, confusion_matrix_enabled):
    test_data = Dataset(test_db)
    y_classified = []
    
    assert len(test_data) != 0
    
    for row in test_data.attributes():
        current_node = tree_start
        
        while not current_node.is_leaf():
            assert current_node.left is not None, [current_node]
            assert current_node.right is not None
            if row[current_node.condition.attribute] < current_node.condition.less_than:
                current_node = current_node.left
            else:
                current_node = current_node.right

        assert current_node.is_leaf()
        y_classified.append(current_node.leaf)

    y_classified_nparray = np.array(y_classified)
    accuracy = np.sum(y_classified_nparray == test_data.labels())/len(y_classified)

    if confusion_matrix_enabled:
        confusion_matrix = np.zeros((4, 4))
        for i in range(len(y_classified_nparray)):
            confusion_matrix[int(test_data.labels()[i])-1, int(y_classified_nparray[i])-1] += 1
        return (accuracy, confusion_matrix)

    return (accuracy)

def precision_and_recall(confusion_matrix):
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
    
    return class_precisions, class_recalls

def machine_learn(dataset, rg=default_rng()):
    no_prune_accs = 0
    pre_prune_accs = 0
    post_prune_accs = 0
    shuffled_dataset = shuffle_dataset(dataset, random_generator=rg)
    confusion_matrix = np.zeros((4, 4))
    
    
    for i in range(10):
        for j in range(1, 10):
            training_data, test_data, validation_data = split_dataset(shuffled_dataset, i, j)
            if j == 1:
                training_data_no_prune = np.concatenate((training_data, validation_data))
                tree_start_node_no_prune = decision_tree_learning(training_data_no_prune, 0)
                no_prune_accs += evaluate(test_data, tree_start_node_no_prune[0], False)
                
            tree_start_node = decision_tree_learning(training_data, 0)

            if j == 1:
                TreeViz(tree_start_node[0]).render()

                pre_prune_accs += evaluate(test_data, tree_start_node[0], False)
                best_pruned_tree = tree_start_node[0]
                best_acc = evaluate(validation_data, tree_start_node[0], False)
        
            best_pruned_tree, best_acc = find_prune(best_pruned_tree, validation_data, best_acc)
    

        TreeViz(best_pruned_tree).render()

        post_prune_eval = evaluate(test_data, best_pruned_tree, True)
        post_prune_accs += post_prune_eval[0]
        confusion_matrix += post_prune_eval[1]

    post_prune_precisions, post_prune_recalls = precision_and_recall(confusion_matrix)
    
    print("no prune 10 fold average accuracy ", no_prune_accs/10)
    print("pre prune 10 fold average accuracy ", pre_prune_accs/10)
    print("post prune 10 fold average accuracy ", post_prune_accs/10)
    print("post prune confusion matrix \n", confusion_matrix)
    for i in range(4):
        print("class ", i+1, " precision = ", post_prune_precisions[i])
        print("class ", i+1, " recall = ", post_prune_recalls[i])
        print("class ", i+1, " F1 = ", (2*post_prune_precisions[i]*post_prune_recalls[i])/(post_prune_precisions[i]+post_prune_recalls[i]))
    
    return

Test()

clean_data = np.loadtxt("intro2ML-coursework1/wifi_db/clean_dataset.txt", delimiter="\t")
noisy_data = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")

seed = 6969
rg = default_rng(seed)

print("CLEAN DATA")
machine_learn(clean_data, rg=rg)

print("NOISY DATA")
machine_learn(noisy_data, rg=rg)


# TODO: visualise tree or something
