class TreeNode:
    def __init__(self, left, right, leaf, condition, pruned) -> None:
        self.left = left
        self.right = right
        self.leaf = leaf
        self.condition = condition
        
        # Marks whether the node has been pruned
        self.pruned = pruned
        
        # Used for visualisation
        self.x = 0
        self.y = 0
        self.mod = 0
        self.thread = None
        
    def is_leaf(self):
        return self.left is None and self.right is None
    
    def shallow_copy(self):
        new_node = TreeNode(self.left, self.right, self.leaf, self.condition, self.pruned)
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


def max_depth_finder(node: TreeNode, depth):
    """Recursivly finds the max depth of a given tree
    """
    if node.is_leaf():
        return depth
    l_depth = max_depth_finder(node.left, depth + 1)
    r_depth = max_depth_finder(node.right, depth + 1)
    return max(l_depth, r_depth)


def mean_depth_finder(node: TreeNode, depth):
    """Recursivly finds the mean depth of a given tree
    """
    if node.is_leaf():
        return depth, 1
    l_depth, l_num_of_leafs = mean_depth_finder(node.left, depth + 1)
    r_depth, r_num_of_leafs = mean_depth_finder(node.right, depth + 1)
    if depth == 0:
        return (l_depth + r_depth) / (l_num_of_leafs + r_num_of_leafs)
    else:
        return l_depth + r_depth, l_num_of_leafs + r_num_of_leafs
    

def copy_tree(node: TreeNode):
    """Performs a deep copy of the provided tree.
    """
    if node is None:
        return None

    new_tree = node.shallow_copy()
    new_tree.left = copy_tree(node.left)
    new_tree.right = copy_tree(node.right)
    return new_tree
