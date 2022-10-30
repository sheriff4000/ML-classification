from wsgiref import validate
from numpy.random import default_rng
import numpy as np
import random


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
    if node.left.leaf == node.right.leaf:
        node.left.pruning = True
        return node.left
    tmp_node = [node.left, node.right, node.leaf, node.condition, node.pruning]
    accuracies = np.ndarray((3, 1))
    accuracies[2] = evaluate(dataset, top_node)[0]
    node.leaf = tmp_node[0].leaf
    node.left = None
    node.right = None
    node.condition = None
    node.pruning = False
    left_node = [node.left, node.right,
                 node.leaf, node.condition, node.pruning]
    accuracies[0] = evaluate(dataset, top_node)[0]

    node.leaf = tmp_node[1].leaf
    node.left = None
    node.right = None
    node.condition = None
    node.pruning
    right_node = [node.left, node.right,
                  node.leaf, node.condition, node.pruning]
    accuracies[1] = evaluate(dataset, top_node)[0]
    # print(accuracies)

    best_acc_arg = accuracies.argmax()
    assert left_node[0] is not None or left_node[2] is not None
    assert right_node[0] is not None or right_node[2] is not None
    assert tmp_node[0] is not None or tmp_node[2] is not None

    match best_acc_arg:
        case 2:

            node.left, node.right, node.leaf, node.condition, node.pruning = tmp_node[
                0], tmp_node[1], tmp_node[2], tmp_node[3], tmp_node[4]
            # print("case 0: ", node.pruning)
            assert node.left is not None or node.leaf is not None
            assert node.pruning is False
            return node
        case 0:
            node.left, node.right, node.leaf, node.condition, node.pruning = left_node[
                0], left_node[1], left_node[2], left_node[3], True
            # print("case 1: ", node.pruning)
            assert node.pruning is True
            return node
        case 1:
            node.left, node.right, node.leaf, node.condition, node.pruning = right_node[
                0], right_node[1], right_node[2], right_node[3], True
            # print("case 2: ", node.pruning)
            assert node.pruning is True
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
        if node.left.pruning is True or node.right.pruning is True:
            node.left.pruning = False
            node.right.pruning = False
            # print("repruning")
            # print("RE left pruned val: ", node.left.pruning)
            # print("RE right pruned val: ", node.right.pruning)
            node = pruning(dataset, node, top_node)
    return node


def print_tree(node, outfile, num_arrows):
    if node.leaf is not None:
        for i in range(num_arrows):
            outfile.write("-")

        outfile.write(str(node.leaf) + "\n")
        num_arrows -= 1
        return node
    else:
        for i in range(num_arrows):
            outfile.write("-")

        num_arrows += 1
        outfile.write("node\n")
        node.left = print_tree(node.left, outfile, num_arrows)
        node.right = print_tree(node.right, outfile, num_arrows)
    return node


# split 70/10/10/10
def split_dataset(dataset, test_idx):
    subsets = np.split(dataset, 10)

    # test_idx = 0, validation = 1, test_idx = 1 validation = 2
    test_data = subsets.pop(test_idx)
    # validation_index = random.randint(0, 8)
    # pruning_index = random.randint(0, 7)
    validation_data = subsets.pop(test_idx-1)
    training_data = np.concatenate(
        (subsets[0], subsets[1], subsets[2], subsets[3], subsets[4], subsets[5], subsets[6], subsets[7]))
    assert validation_data.all() == test_data.all()
    return training_data, test_data, validation_data


def shuffle_dataset(dataset, random_generator=default_rng()):
    shuffled_indicies = random_generator.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indicies]

    return shuffled_dataset

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
        # print("current node: ", current_node.left)
        while current_node.leaf is None:
            # print(current_node.left, current_node.right)
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


def copy_tree(node):
    new_tree = TreeNode(None, None, node.leaf, None, False)
    if node.leaf is None:
        left = copy_tree(node.left)
        right = copy_tree(node.right)

        condition = node.condition
        leaf = node.leaf
        new_tree = TreeNode(left, right, leaf, condition, False)

    return new_tree


def find_prune(node, validation_data, test_data, best_acc):

    prune_tree = copy_tree(node)
    test_prune = pruning(validation_data, prune_tree, prune_tree)

    current_acc, current_mat = evaluate(test_data, test_prune)

    # print(current_mat)

    if current_acc > best_acc:
        return test_prune, current_acc
    else:
        return node, best_acc


Test()

y = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
x = np.loadtxt(
    "intro2ML-coursework1/wifi_db/clean_dataset.txt", delimiter="\t")
seed = 6969
rg = default_rng(seed)
pre_prune_accs = 0
post_prune_accs = 0
best_prune_accs = 0
copy_accs = 0
tmp = 0
shuffled_dataset = shuffle_dataset(x, random_generator=rg)
for i in range(10):
    training_data, test_data, validation_data = split_dataset(
        shuffled_dataset, i)  # splitting dataset every time
    # creating new tree based on new training data
    # print("training data: ", training_data)
    # print("test data: ", test_data)
    # print("validation data: ", validation_data)
    # print("pruning data: ", pruning_data)
    training_data_no_prune = np.concatenate(
        (training_data, validation_data))

    tree_start_node_no_prune = decision_tree_learning(
        training_data_no_prune, 0)

    tree_start_node = decision_tree_learning(training_data, 0)
    if i == 0:
        best_pruned_tree = tree_start_node[0]
        best_acc = evaluate(test_data, tree_start_node[0])[0]
        tmp = 1
    with open("original_tree_vis" + str(i)+".txt", "w") as outfile:
        print_tree(tree_start_node[0], outfile, num_arrows=0)
    with open("pruned_tree_vis" + str(i)+".txt", "w") as outfile:
        print_tree(pruning(
            validation_data, tree_start_node[0], tree_start_node[0]), outfile, num_arrows=0)
    best_pruned_tree, best_acc = find_prune(
        best_pruned_tree, validation_data, test_data, best_acc)

    # pre_prune_accs += prune_acc[0]
    # print(prune_acc[1])
no_prune_acc = evaluate(test_data, tree_start_node_no_prune[0])
# with open('original tree vis.txt', 'w') as f:
#    print_tree(tree_start_node[0], num_arrows=0)
print("no prune acc", no_prune_acc[0])
print("pre prune accs", evaluate(test_data, tree_start_node[0])[0])
print("best pruned acc no test ", best_acc)
print("best pruned acc tested", evaluate(test_data, best_pruned_tree)[0])

print("tested against opposite datasets ",
      evaluate(y, best_pruned_tree)[0])
# with open('pruned tree vis.txt', 'w') as f:
#    print_tree(best_pruned_tree, num_arrows=0)

# TODO: visualise tree or something
