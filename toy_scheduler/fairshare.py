import os

import pandas as pd
import numpy as np

# TODO define a base class and make this neater, _ infront of add_parent to indicate private


class Root:
    def __init__(self, initial_usage, name="root"):
        self.name = name
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []

        self.is_root = True
        self.is_leaf = False

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self):
        ret = "root\n"
        for child in self.children:
            ret += child.__str__(1)
        return ret


# NOTE children can be projects or accounts
# NOTE disambiguated from Partition in partitions.py
class PartitionNode:
    def __init__(self, name, shares, initial_usage):
        self.name = name
        self.shares = shares
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = np.inf

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# There can be any number of project levels including 0
class Project:
    def __init__(self, name, shares, initial_usage):
        self.name = name
        self.shares = shares
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = np.inf

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# {Account, User} us the unique identifier i think, so a job will point to an account and user
# (which may belong to multiple accounts). This account and user will have there usages updated
# then the usage will be propagated up the tree.
class Account:
    def __init__(self, name, shares, initial_usage):
        self.name = name
        self.shares = shares
        self.usage = initial_usage

        self.new_child_usage = False

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = np.inf

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# Users with the same name may coexist under different accounts
class User:
    def __init__(self, name, shares, initial_usage):
        self.name = name
        self.shares = shares
        self.usage = initial_usage

        self.parent = None

        self.is_root = False
        self.is_leaf = True

        self.levelfs = np.inf
        self.fairshare_factor = 1.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        return ret


class FairTree:
    def __init__(self, assoc_file, calc_period, decay_halflife, init_time):
        self.last_calc_time = init_time
        self.calc_period = calc_period
        # decay constant for 1 second applied for the duration of a calc interval
        self.decay_constant = (
            (1 + np.log(1/2) / decay_halflife.total_seconds()) ** calc_period.total_seconds()
        )

        self.current_period_num = 0

        self.root_node = self._load_tree_slurm(assoc_file)
        self.levels = [[self.root_node]] # NOTE Don't think I actually need this

        current_level = 0
        while(len(self.levels) > current_level):
            all_level_children = []
            for node in self.levels[current_level]:
                if node.is_leaf:
                    continue
                for child_node in node.children:
                    all_level_children.append(child_node)

            if all_level_children:
                self.levels.append(all_level_children)

            current_level += 1

        self.uniq_users = { acc.name : {} for acc in self.levels[-2] }
        for user in self.levels[-1]:
            self.uniq_users[user.parent.name][user.name] = user

        self.tot_num_assocs = len(self.levels[-1])

    def __str__(self):
        ret = (
            "\t".join(
                [ "l{} ({})".format(level, len(nodes)) for level, nodes in enumerate(self.levels) ]
            ) +
            "\n"
        )
        ret += self.levels[0][0].__str__()
        return ret

    def next_calc(self):
        return self.last_calc_time + self.calc_period

    # Collect usages from running jobs and traverse tree to get fairshare rank and so score
    # (not bothering with ties since they should be extremely rare except for account that submit
    # one job in a blue moon in which case who cares)
    # TODO Some kind of flag to avoid recomputing levelfs and re-sorting sequences that have not changed
    def fairshare_calc(self, running_jobs, time):
        # Collect usages from running jobs
        for job in running_jobs:
            node = self.uniq_users[job.account][job.user]
            usage = job.nodes * (time - max(self.last_calc_time, job.start)).total_seconds() # TODO node->cpu seconds
            self._update_usages(node, usage)

        self.current_period_num += 1
        self.last_calc_time = time

        # Compute levelFS and sort (decay past usages as we go)
        self._tree_traversal(self.root_node)

    def _tree_traversal(self, current_node, rank=0):
        if current_node.is_leaf:
            current_node.fairshare_factor = 1.0 - rank / self.tot_num_assocs
            rank += 1
            return rank

        if not current_node.new_child_usage: # Avoid re-sorting when order is unchanged
            for child_node in current_node.children:
                child_node.usage *= self.decay_constant
        else:
            for child_node in current_node.children:
                child_node.usage *= self.decay_constant
                child_node.levelfs = child_node.shares / child_node.usage
            current_node.new_child_usage = False
            # XXX Temporary for reproducibility. Should implement logic that deals with ties (merge
            # children and give tied users the same rank)
            current_node.children.sort(key=lambda node: (node.levelfs, node.name), reverse=True)

        for child_node in current_node.children:
            rank = self._tree_traversal(child_node, rank)

        return rank

    def job_finish_usage_update(self, job):
        node = self.uniq_users[job.account][job.user]
        usage = job.nodes * (job.end - max(self.last_calc_time, job.start)).total_seconds()
        self._update_usages(node, usage)

    def _update_usages(self, node, usage):
        node.usage += usage
        while not node.is_root:
            node = node.parent
            node.usage += usage
            node.new_child_usage = True

    # Specifically for archer2 slurm data root->partition->proj->acc->user
    def _load_tree_slurm(self, assoc_file):
        df = pd.read_csv(assoc_file, delimiter='|', lineterminator='\n', header=0)
        df = df.drop([ col for col in df.columns if "Unnamed" in col ], axis=1)

        root_node = Root(0.0)
        for _, partition_row in df.loc[(df.ParentName == root_node.name)].iterrows():
            partition = PartitionNode(partition_row.Account, 1, 0.0)
            root_node.add_child(partition)

        for partition_node in root_node.children:
            for _, proj_row in df.loc[(df.ParentName == partition_node.name)].iterrows():
                proj = Project(proj_row.Account, 1, 0.0)
                partition_node.add_child(proj)

            for proj_node in partition_node.children:
                for _, acc_row in df.loc[(df.ParentName == proj_node.name)].iterrows():
                    acc = Account(acc_row.Account, 1, 0.0)
                    proj_node.add_child(acc)

                for acc_node in proj_node.children:
                    for _, user_row in df.loc[(df.Account == acc_node.name) & (df.User.notna())].iterrows():
                        user = User(user_row.User, 1, 0.0)
                        acc_node.add_child(user)

        return root_node

