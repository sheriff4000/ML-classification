import numpy as np
import matplotlib.pyplot as plt

from tree import TreeNode

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
        delta_x = left_x - right_x
        
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
