

class TreeNode:
    def __init__(self, left, right, leaf, condition, pruned) -> None:
        self.left = left
        self.right = right
        self.leaf = leaf
        self.condition = condition
        
        # Marks whether the node has been pruned
        self.pruning = pruned
        
        # Used for visualisation
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
