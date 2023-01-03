import os

import pandas as pd
import numpy as np

# TODO define a base class and make this neater, _ infront of add_parent to indicate private
# NOTE if decay multiplcation operations are slow, could put all usages in a 2d array then give each class an index to their row


class Root():
    def __init__(self, initial_usages, name="root"):
        self.name = name
        self.usages = initial_usages

        self.children = []

        self.is_root = True
        self.is_leaf = False

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def update_usage(self, usage):
        self.usages[0] += usage

    def next_time_period(self):
        self.usage.pop(0)
        self.usage.appendleft(0)

    def __str__(self):
        ret = "root\n"
        for child in self.children:
            ret += child.__str__(1)
        return ret


# NOTE children can be projects or accounts
class Partition():
    def __init__(self, name, shares, initial_usages):
        self.name = name
        self.shares = shares
        self.norm_shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = 0.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def update_usage(self, usage):
        self.usages[0] += usage

    def next_time_period(self):
        self.usage.pop(0)
        self.usage.appendleft(0)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# There can be any number of project levels including 0
class Project():
    def __init__(self, name, shares, initial_usages):
        self.name = name
        self.shares = shares
        self.norm_shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = 0.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def update_usage(self, usage):
        self.usages[0] += usage

    def next_time_period(self):
        self.usage.pop(0)
        self.usage.appendleft(0)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# {Account, User} us the unique identifier i think, so a job will point to an account and user
# (which may belong to multiple accounts). This account and user will have there usages updated
# then the usage will be propagated up the tree.
class Account():
    def __init__(self, name, shares, initial_usages):
        self.name = name
        self.shares = shares
        self.norm_shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

        self.levelfs = 0.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        self.children.append(child)
        child.add_parent(self)

    def update_usage(self, usage):
        self.usages[0] += usage

    def next_time_period(self):
        self.usage.pop(0)
        self.usage.appendleft(0)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return ret


# Users with the same name may coexist under different accounts
class User():
    def __init__(self, name, shares, initial_usages):
        self.name = name
        self.shares = shares
        self.norm_shares = shares
        self.usages = initial_usages

        self.parent = None

        self.is_root = False
        self.is_leaf = True

        self.levelfs = 0.0
        self.fairshare_factor = 1.0

    def add_parent(self, parent):
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def update_usage(self, usage):
        self.usages[0] += usage

    def next_time_period(self):
        self.usage.pop(0)
        self.usage.appendleft(0)

    def __str__(self, level=0):
        ret = "\t"*level + self.name + "\n"
        return ret


class FairTree():
    def __init__(self, assoc_file, calc_period, decay_halflife, simulation_length, init_time):
        self.last_calc_time = init_time
        self.calc_period = calc_period
        self.decay_constant = (1 - np.log(1/2) / decay_halflife) ** calc_period.total_seconds() # decay constant for 1 second applied for the duration of a calc interval

        self.current_period_num = 0

        num_usage_periods = int(simulation_length / calc_period) + 1
        self.root_node = self._load_tree_slurm(assoc_file, num_usage_periods)
        self.levels = [[self.root_node]] # NOTE What am I doing with levels?

        current_level = 0
        while(len(self.levels) > current_level):
            all_level_children = []
            for node in self.levels[current_level]:
                if node.is_leaf:
                    continue
                for child_node in node.children:
                    all_level_children.append(child_node)

            if all_level_children:
                tot_level_shares = sum([ child.shares for child in all_level_children ])
                for child_node in all_level_children:
                    child_node.norm_shares /= tot_level_shares
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
    def fairshare_calc(self, queue, time):
        # Collect usages from running jobs
        for job in queue.queue:
            node = self.uniq_users[job.account][job.user]
            usage = job.nodes * (time - max(self.last_calc_time, job.start)).total_seconds() # TODO node->cpu seconds
            self._update_usages(node, usage)

        self.current_time_period += 1
        self.last_calc_time += self.calc_period

        # Compute levelFS and sort (decay past usages as we go)
        # TODO do a clever loop thing
        rank = 0
        for partition_node in self.root_node.children:
            partition_node.usage[:self.current_time_period] *= self.decay_constant
            partition_node.levelfs = partition_node.norm_shares / partition_node.usages.sum() # NOTE realise I can just use raw shares here
        self.root_node.children.sort(key=lambda node: node.levelfs, reverse=True)

        for partition_node in self.root_node.children:
            for proj_node in partition_node.children:
                proj_node.usage[:self.current_time_period] *= self.decay_constant
                proj_node.levelfs = proj_node.norm_shares / proj_node.usages.sum()
            partition_node.children.sort(key=lambda node: node.levelfs, reverse=True)

            for proj_node in partition_node.children:
                for acc_node in proj_node.children:
                    acc_node.usage[:self.current_time_period] *= self.decay_constant
                    acc_node.levelfs = acc_node.norm_shares / acc_node.usages.sum()
                proj_node.children.sort(key=lambda node: node.levelfs, reverse=True)

                for acc_node in proj_node.children:
                    for user_node in acc_node.children:
                        user_node.usage[:self.current_time_period] *= self.decay_constant
                        user_node.levelfs = user_node.norm_shares / user_node.usages.sum()
                    acc_node.children.sort(key=lambda node: node.levelfs, reverse=True)

                    for user_node in acc_node.children:
                        user_node.fairshare_factor = 1.0 - rank / self.tot_num_assocs
                        rank += 1

    # Collect the remaining usage from finished jobs
    def job_finish_usage_update(self, job):
        node = self.uniq_users[job.account][job.user]
        usage = job.nodes * (job.end - self.last_calc_time).total_seconds() # TODO node->cpu seconds
        self._update_usages(node, usage)

    def _update_usages(self, node, usages):
        node.usages[self.current_period_num] += usage
        while not node.is_root:
            node = node.parent
            node.usages[self.current_period_num] += usage

    # Specifically for archer2 slurm data root->partition->proj->acc->user
    def _load_tree_slurm(self, assoc_file, num_usage_periods):
        df = pd.read_csv(assoc_file, delimiter='|', lineterminator='\n', header=0)
        df = df.drop([ col for col in df.columns if "Unnamed" in col ], axis=1)

        root_node = Root()
        for _, partition_row in df.loc[(df.ParentName == root_node.name)].iterrows():
            partition = Partition(partition_row.Account, 1)
            root_node.add_child(partition, initial_usages=np.zeros(num_usage_periods))

        for partition_node in root_node.children:
            for _, proj_row in df.loc[(df.ParentName == partition_node.name)].iterrows():
                proj = Project(proj_row.Account, 1)
                partition_node.add_child(proj, initial_usages=np.zeros(num_usage_periods))

            for proj_node in partition_node.children:
                for _, acc_row in df.loc[(df.ParentName == proj_node.name)].iterrows():
                    acc = Account(acc_row.Account, 1)
                    proj_node.add_child(acc, initial_usages=np.zeros(num_usage_periods))

                for acc_node in proj_node.children:
                    for _, user_row in df.loc[(df.Account == acc_node.name) & (df.User.notna())].iterrows():
                        user = User(user_row.User, 1)
                        acc_node.add_child(user, initial_usages=np.zeros(num_usage_periods))

        return root_node


""" Tests """

"""
root = Root()

proj1, proj2 = Project("proj1", 1), Project("proj2", 1)
for proj in [proj1, proj2]:
    root.add_partition(proj)

acc1, acc2, acc3 = Account("acc1", 1), Account("acc2", 1), Account("acc3", 1)
for acc in [acc1, acc2]:
    proj1.add_child(acc)
proj2.add_child(acc3)

user1, user2, user3, user4, user5 = User("u1", 1), User("u3", 1), User("u3", 1), User("u4", 1), User("u5", 1),
for user in [user1, user2]:
    acc1.add_child(user)
for user in [user3, user5]:
    acc2.add_child(user)
acc3.add_child(user4)

tree = FairTree(root)
print(tree)
"""

# load_tree_slurm("/work/y02/y02/awilkins/sacct_archer2_assocs_030123.csv")

