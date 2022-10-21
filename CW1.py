import numpy as np

class Test:
    def __init__(self) -> None:
        data = read_file("intro2ML-coursework1/wifi_db/clean_dataset.txt")
        set = TrainingSet(data)
        assert(len(set.unique_labels()[0]) == 4)
        assert(set.unique_labels()[0][0] == 1)

class TreeNode:
    def __init__(self, left, right, value, condition) -> None:
       self.left = left
       self.right = right
       # Used for a leaf node
       self.value = value
       self.condition = condition

class SplitCondition:
    def __init__(self, attribute, less_than) -> None:
        self.attribute = attribute
        self.less_than = less_than

def read_file(file: str):
    data = np.loadtxt(file, delimiter="\t")

    return data

class TrainingSet:
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

def find_split(training_set: TrainingSet):
    max_gain = 0;
    entropy = training_set.find_entropy()
    medians = np.median(training_set.attributes(), axis=1)
    for i in range(training_set.attributes().shape[1]):
        # TODO: find better way to split
        median = medians[i]
        
        attribute = training_set.dataset[:,i]
        a = TrainingSet(training_set.dataset[attribute < median])
        b = TrainingSet(training_set.dataset[attribute >= median])
        
        sub_entropy = (a.find_entropy() + b.find_entropy()) / 2

        gain = entropy-sub_entropy
        print(gain)

        if gain > max_gain :
            max_gain = gain
            split_idx = i
            out_a = a
            out_b = b

    return split_idx, medians[split_idx], out_a, out_b

def decision_tree_learning(training_dataset, depth):
    data = TrainingSet(training_dataset)
    labels = data.unique_labels()
    if len(labels) == 1: 
        # TODO: bug: what about if 0?
        return TreeNode(None, None, labels[0], None), depth
    
    attribute, split_val, l_dataset, r_dataset = find_split(data.dataset)
    
    split_cond = SplitCondition(attribute, split_val)
    node = TreeNode(l_dataset, r_dataset, None, split_cond)
    
    l_branch, l_depth = decision_tree_learning(l_dataset, depth + 1)
    r_branch, r_depth = decision_tree_learning(r_dataset, depth + 1)
    
    return (node, max(l_depth, r_depth))

Test()

x = read_file("intro2ML-coursework1/wifi_db/clean_dataset.txt")
y = TrainingSet(x)
print(find_split(y))


# TODO: find other split methods (mean?) & compare splits
# TODO: pruning
# TODO: make training set and test set compare accuracy with different values of test set proportion
# TODO: test overfitting by using noisy dataset
# TODO: runtime decision tree classify function
# TODO: visualise tree or something