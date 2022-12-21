from collections import deque

# TODO define a base class and make this neater


class Root():
    def __init__(self, initial_usages=deque([0] * 5)):
        self.usages = initial_usages

        self.children = []

        self.is_root = True
        self.is_leaf = False

    def add_partition(self, partition):
        self.children.append(partition)
        partition.add_parent(self)

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
    def __init__(self, name, shares, initial_usages=deque([0] * 5)):
        self.name = name
        self.shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

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
    def __init__(self, name, shares, initial_usages=deque([0] * 5)):
        self.name = name
        self.shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

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
    def __init__(self, name, shares, initial_usages=deque([0] * 5)):
        self.name = name
        self.shares = shares
        self.usages = initial_usages

        self.children = []
        self.parent = None

        self.is_root = False
        self.is_leaf = False

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
    def __init__(self, name, shares, initial_usages=deque([0] * 5)):
        self.name = name
        self.shares = shares
        self.usages = initial_usages

        self.parent = None

        self.is_root = False
        self.is_leaf = True

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


class AssocTree():
    def __init__(self, root_node):

        self.levels = [[root_node]]

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

    def __str__(self):
        ret = "\t".join([ "level{} ({})".format(level, len(nodes)) for level, nodes in enumerate(self.levels) ])  + "\n"
        ret += self.levels[0][0].__str__()
        return ret


""" Tests """

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

tree = AssocTree(root)
print(tree)


