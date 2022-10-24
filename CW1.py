from numpy.random import default_rng
import numpy as np

class Test:
    def __init__(self) -> None:
        data = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
        set = Dataset(data)
        assert(len(set.unique_labels()[0]) == 4)
        assert(set.unique_labels()[0][0] == 1)

class TreeNode:
    def __init__(self, left, right, leaf, condition) -> None: 
       self.left = left 
       self.right = right
       self.leaf = leaf #
       self.condition = condition

class SplitCondition:
    def __init__(self, attribute, less_than) -> None:
        self.attribute = attribute
        self.less_than = less_than

class Dataset:
    def __init__(self, dataset) -> None:
        self.dataset = dataset
        self.LABEL_COL_INDEX = -1
    
    def attributes(self):
        return self.dataset[:,:self.LABEL_COL_INDEX]

    def labels(self):
        return self.dataset[:, self.LABEL_COL_INDEX]
    
    def unique_labels(self):
        return np.unique(self.labels(), return_counts=True)

    def find_entropy(self) -> float:
        attributes = self.attributes()
        labels = self.labels()
    
        #print(attributes)
        #print(labels)
        unique_labels, label_count = self.unique_labels()
        #print(unique_labels)
        #print(label_count)
        #print(self.dataset)

        assert (np.sum(label_count) == len(labels)) #make sure stuffs not broken i guess
        entropy = 0
        for i in range(len(unique_labels)):
            prob = float(label_count[i]) / float(len(labels))
            entropy -= prob * np.log2(prob)
            #print(entropy)

        return entropy

    def __len__(self):
        return len(self.dataset)

    # def __getitem__(self, i):
    #     return self.dataset[i]        


def find_split(training_set: Dataset):
    max_gain = 0
    entropy = training_set.find_entropy()
    medians = np.median(training_set.attributes(), axis=0)
    for i in range(medians.shape[0]):
        # TODO: find better way to split
        median = medians[i]
        
        attribute = training_set.dataset[:,i]
        a = Dataset(training_set.dataset[attribute < median])
        b = Dataset(training_set.dataset[attribute >= median]) 
        

        remainder = ((len(a) / len(training_set)) * a.find_entropy()) + ((len(b) / len(training_set)) * b.find_entropy())

        gain = entropy - remainder

        if gain >= max_gain:
            max_gain = gain
            split_idx = i
            out_a = a
            out_b = b
        
    
    assert max_gain != 0

    return split_idx, medians[split_idx], out_a.dataset, out_b.dataset

def decision_tree_learning(training_dataset, depth):
    data = Dataset(training_dataset)
    assert len(data) != 0
    labels = data.unique_labels()
    if len(labels[0]) == 1: 
        # TODO: bug: what about if 0?
        print("leaf with label " , labels[0][0])
        return TreeNode(None, None, labels[0][0], None), depth
    attribute, split_val, l_dataset, r_dataset = find_split(data)
    
    split_cond = SplitCondition(attribute, split_val)
    
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)

    node = TreeNode(l_branch, r_branch, None, split_cond)

    
    return (node, max(l_depth, r_depth))

def pruning(dataset, node, top_node):
    tmp_node = node
    if node.left.leaf is not None and node.right.leaf is not None: 
        node.left = pruning(dataset, node.left, top_node)
        #do stuff
        node.right = pruning(dataset, node.right, top_node)
    else:
        accuracies = np.zeros(3,)
        accuracies[0] = evaluate(dataset, top_node)
        
        left_node = node.left
        right_node = node.right
    
        node = left_node
        accuracies[1] = evaluate(dataset, top_node)
        node = right_node
        accuracies[2] = evaluate(dataset, top_node)
    
        max_acc = accuracies.argmax()

        match max_acc:
            case 0:
                return tmp_node
            case 1:
                return left_node
            case 2:
                return right_node
        
    return node

def split_dataset(dataset, random_generator=default_rng()):
    shuffled_indecies = random_generator.permutation(len(dataset))
    shuffled_dataset = dataset[shuffled_indecies]
    return np.array(np.split(shuffled_dataset, 10))

def evaluate(test_db, tree_start):
    test_data = Dataset(test_db)
    current_node = tree_start
    y_classified = []
    print(test_db.shape)
    assert(len(test_db) !=  0)
    for row in test_data.attributes():
        while current_node.leaf is None:
            if row[current_node.condition.attribute] < current_node.condition.less_than:
                current_node = current_node.left
            else:
                current_node = current_node.right
        assert current_node.leaf != None
        y_classified.append(current_node.leaf)
        current_node = tree_start
    
    #simple implementation for accuracy of one run
    #TODO: update to 10 cross fold
    
    y_classified_nparray = np.array(y_classified)

    print(y_classified_nparray)
    print(test_data.labels())

    accuracy = np.sum(y_classified_nparray == test_data.labels())/len(y_classified)


    return accuracy

Test()

x = np.loadtxt("intro2ML-coursework1/wifi_db/noisy_dataset.txt", delimiter=" ")
seed = 6969
rg = default_rng(seed)
data_subsets = split_dataset(x, random_generator=rg)
training_set = np.concatenate((data_subsets[0], data_subsets[1], data_subsets[2], data_subsets[3], data_subsets[4], data_subsets[5], data_subsets[6], data_subsets[7], data_subsets[8]))
tree_start_node = decision_tree_learning(training_set, 0)
print(evaluate(data_subsets[9], tree_start_node[0]))


# TODO: find other split methods (mean?) & compare splits
# TODO: pruning
# TODO: make training set and test set compare accuracy with different values of test set proportion
# TODO: test overfitting by using noisy dataset
# TODO: visualise tree or something