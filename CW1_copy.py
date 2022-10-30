# MAKE SURE

from wsgiref import validate
from numpy.random import default_rng
import numpy as np


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


class SplitCondition:
    def __init__(self, attribute, less_than) -> None:
        self.attribute = attribute
        self.less_than = less_than


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
        return TreeNode(None, None, labels[0][0], None, False), depth
    attribute, split_val, l_dataset, r_dataset = find_split(data)

    split_cond = SplitCondition(attribute, split_val)

    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    node = TreeNode(l_branch, r_branch, None, split_cond, False)

    return (node, max(l_depth, r_depth))


def clip_tree(dataset, node, top_node):
    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruning]
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
    node.pruning = False
    left_node = [node.left, node.right,
                 node.leaf, node.condition, node.pruning]
    accuracies[1] = evaluate(dataset, top_node)[0]

    node.leaf = tmp_node[1].leaf
    node.left = None
    node.right = None
    node.condition = None
    node.pruning = False
    right_node = [node.left, node.right,
                  node.leaf, node.condition, node.pruning]
    accuracies[2] = evaluate(dataset, top_node)[0]
    # print(accuracies)

    best_acc_arg = accuracies.argmax()
    assert left_node[0] is not None or left_node[2] is not None
    assert right_node[0] is not None or right_node[2] is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    match best_acc_arg:
        case 0:

            node.left, node.right, node.leaf, node.condition, node.pruning = tmp_node[
                0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
            #print("node_change case 0: ", node)
            #print("node leaf = ", node.leaf)
            #print("node left = ", node.left)
            #print("node right = ", node.right)
            #print("node left leaf = ", node.left.leaf)
            #print("node right leaf = ", node.right.leaf)
            #print("node right = ", node.right)
            #print('tmp_node: ', tmp_node)
            assert node.left is not None or node.leaf is not None

            return node
        case 1:
            node.left, node.right, node.leaf, node.condition, node.pruning = left_node[
                0], left_node[1], left_node[2], left_node[3], True
            #print("case 1 right: ", node.right)
            #print("node_change case 1: ", node)
            #print("node leaf = ", node.leaf)
            #print("node left leaf = ", node.left.leaf)
            #print("node right leaf = ", node.right.leaf)
            #print('left_node: ', left_node)

            return node
        case 2:
            node.left, node.right, node.leaf, node.condition, node.pruning = right_node[
                0], right_node[1], right_node[2], right_node[3], True
            #print("node_change case 2: ", node)
            #print("node leaf = ", node.leaf)
            #print("node left leaf = ", node.left.leaf)
            #print("node right leaf = ", node.right.leaf)
            #print('right_node: ', right_node)
            return node


def pruning(dataset, node, top_node):
    if node.leaf is not None:
        return node
    if node.left.leaf is not None and node.right.leaf is not None:  # is this a fruit
        node = clip_tree(dataset, node, top_node)
        return node
    if (node.left.leaf is None and node.right.leaf is not None) or (node.left.leaf is not None and node.right.leaf is None) or (node.left.leaf is None and node.right.leaf is None):
        node.left = pruning(dataset, node.left, top_node)
        node.right = pruning(dataset, node.right, top_node)
        #print("left pruning:", node.left.pruning)
        #print("right pruning:", node.right.pruning)
        if node.left.pruning is True or node.right.pruning is True:
            node.left.pruning = False
            node.right.pruning = False
            print("repruning")
            print("RE left pruned val: ", node.left.pruning)
            print("RE right pruned val: ", node.right.pruning)
            node = pruning(dataset, node, top_node)
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


Test()

x = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
seed = 6969
rg = default_rng(seed)
no_prune_accs = 0
pre_prune_accs = 0
post_prune_accs = 0
shuffled_dataset = shuffle_dataset(x, random_generator=rg)
for i in range(10):
    training_data, test_data, validation_data = split_dataset(
        shuffled_dataset, i)
    training_data_no_prune = np.concatenate((training_data, validation_data))

    tree_start_node_no_prune = decision_tree_learning(
        training_data_no_prune, 0)
    tree_start_node = decision_tree_learning(training_data, 0)

    no_prune_accs += evaluate(test_data, tree_start_node_no_prune[0])[0]
    pre_prune_accs += evaluate(test_data, tree_start_node[0])[0]

    pruning_node = pruning(
        validation_data, tree_start_node[0], tree_start_node[0])

    post_prune_eval = evaluate(test_data, pruning_node)
    post_prune_accs += post_prune_eval[0]
    # print(post_prune_eval[1])


print("no prune 10 fold average accuracy ", no_prune_accs/10)
print("pre prune 10 fold average accuracy ", pre_prune_accs/10)
print("post prune 10 fold average accuracy ", post_prune_accs/10)

# TODO: visualise tree or something
