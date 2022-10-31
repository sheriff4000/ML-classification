# MAKE SURE

from wsgiref import validate
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
    def __init__(self, left, right, leaf, condition) -> None:
        self.left = left
        self.right = right
        self.leaf = leaf
        self.condition = condition
        self.x = 0
        self.y = 0
        self.mod = 0
        self.thread = None
        
    def is_leaf(self):
        return self.left is None and self.right is None


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

    # def __getitem__(self, i):
    #     return self.dataset[i]

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
        plt.show()
    
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
        # TODO: find better way to split
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
        return TreeNode(None, None, labels[0][0], None), depth
    attribute, split_val, l_dataset, r_dataset = find_split(data)

    split_cond = SplitCondition(attribute, split_val)

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    node = TreeNode(l_branch, r_branch, None, split_cond)

    return (node, max(l_depth, r_depth))


def clip_tree(dataset, node, top_node):
    tmp_node = [node.left, node.right, node.leaf, node.condition]
    #print("tmp_node", tmp_node)
    accuracies = np.ndarray((3, 1))
    #print("node = ", node)
    #print("top node: ", top_node)
    #print("node leaf = ", node.leaf)
    # print("clipping")
    accuracies[0] = evaluate(dataset, top_node)[0]
    node.leaf = tmp_node[0].leaf
    node.left = None
    node.right = None
    node.condition = None
    left_node = [node.left, node.right, node.leaf, node.condition]
    accuracies[1] = evaluate(dataset, top_node)[0]

    node.leaf = tmp_node[1].leaf
    node.left = None
    node.right = None
    node.condition = None
    right_node = [node.left, node.right, node.leaf, node.condition]
    accuracies[2] = evaluate(dataset, top_node)[0]
    # print(accuracies)

    best_acc_arg = accuracies.argmax()
    assert left_node[0] is not None or left_node[2] is not None
    assert right_node[0] is not None or right_node[2] is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    if best_acc_arg == 0:
        node.left, node.right, node.leaf, node.condition = tmp_node[
            0], tmp_node[1], tmp_node[2], tmp_node[3]
        assert node.left is not None or node.leaf is not None
    elif best_acc_arg == 1:
        node.left, node.right, node.leaf, node.condition = left_node[
            0], left_node[1], left_node[2], left_node[3]
    elif best_acc_arg == 2:
        node.left, node.right, node.leaf, node.condition = right_node[
            0], right_node[1], right_node[2], right_node[3]
    else:
        print("Couldn't find matching best_acc_arg")
        exit(1)
    return node

def pruning(dataset, node, top_node):
    if node.leaf is not None:
        return node
    else:
        if node.left.leaf is not None and node.right.leaf is not None:  # is this a fruit
            node = clip_tree(dataset, node, top_node)
        else:
            node.left = pruning(dataset, node.left, top_node)
            node.right = pruning(dataset, node.right, top_node)
    return node


# split 80/10/10
def shuffle_dataset(dataset, random_generator=default_rng()):
    shuffled_indecies = random_generator.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indecies]

    return shuffled_dataset


def split_dataset(dataset, test_idx):
    subsets = np.split(dataset, 10)

    test_data = subsets.pop(test_idx)
    validation_data = subsets.pop((test_idx+1) % 9)
    training_data = np.concatenate(
        (subsets[0], subsets[1], subsets[2], subsets[3], subsets[4], subsets[5], subsets[6], subsets[7]))

    return training_data, test_data, validation_data


# split with 60/20/20
# def split_dataset(dataset, test_idx, random_generator=default_rng()):
#     shuffled_indecies = random_generator.permutation(len(dataset))
#     shuffled_dataset = dataset[shuffled_indecies]

#     subsets = np.split(shuffled_dataset, 10)

#     test_data1 = subsets.pop(test_idx)
#     test_data2 = subsets.pop(test_idx%9)
#     test_data = np.concatenate((test_data1, test_data2))
#     validation_data1 = subsets.pop(0)
#     validation_data2 = subsets.pop(0)
#     validation_data = np.concatenate((validation_data1, validation_data2))
#     training_data = np.concatenate((subsets[0], subsets[1], subsets[2], subsets[3], subsets[4], subsets[5]))

#     return training_data, test_data, validation_data


def evaluate(test_db, tree_start):
    test_data = Dataset(test_db)
    current_node = tree_start
    y_classified = []
    # print(test_db.shape)
    assert (len(test_db) != 0)
    for row in test_data.attributes():
        #print("current node: ", current_node.left)
        while current_node.leaf is None:
            #print(current_node.left, current_node.right)
            assert current_node.left is not None, [current_node]
            assert current_node.right is not None
            if row[current_node.condition.attribute] < current_node.condition.less_than:
                current_node = current_node.left
            else:
                current_node = current_node.right
        assert current_node.leaf != None
        y_classified.append(current_node.leaf)
        current_node = tree_start

    y_classified_nparray = np.array(y_classified)

    accuracy = np.sum(y_classified_nparray ==
                      test_data.labels())/len(y_classified)

    confusion_matrix = np.zeros((4, 4))

    for i in range(len(y_classified_nparray)):
        confusion_matrix[int(test_data.labels()[i])-1,
                         int(y_classified_nparray[i])-1] += 1

    return accuracy, confusion_matrix


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

Test()

clean_data = np.loadtxt("intro2ML-coursework1/wifi_db/clean_dataset.txt", delimiter="\t")
noisy_data = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")

seed = 6969
rg = default_rng(seed)
no_prune_accs = 0
pre_prune_accs = 0
post_prune_accs = 0
shuffled_dataset = shuffle_dataset(clean_data, random_generator=rg)
confusion_matrix = np.zeros((4, 4))


for i in range(10):
    training_data, test_data, validation_data = split_dataset(
        shuffled_dataset, i)
    training_data_no_prune = np.concatenate((training_data, validation_data))

    tree_start_node_no_prune = decision_tree_learning(
        training_data_no_prune, 0)
    tree_start_node = decision_tree_learning(training_data, 0)

    no_prune_accs += evaluate(test_data, tree_start_node_no_prune[0])[0]
    pre_prune_accs += evaluate(test_data, tree_start_node[0])[0]
    
    if i == 0:
        TreeViz(tree_start_node[0]).render()

    pruning(validation_data, tree_start_node[0], tree_start_node[0])

    post_prune_eval = evaluate(test_data, tree_start_node[0])
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

# TODO: visualise tree or something
