# MIT License
#
# Copyright (c) 2023-2025 Hewlett Packard Enterprise Development LP 
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import os

import pandas as pd
import numpy as np

# NOTE Assuming that users appear on the at levels where there are only other users (other leaf
# nodes). This is relevant to tie breakers on mixed account user levels

# See https://slurm.schedmd.com/fair_tree.html#fairshare for a description
# of the fairshare algorithm.

class Root:
    """
    The root of the fair tree. This is usually the 'root' user.
    
    All other accounts stem from this tree, at multiple levels.
    This Root is the parent of users and accounts, which are then
    parents of further users and accounts, all the way to the 
    terminating leaves, which are users.
    """
    def __init__(self, initial_usage, name="root"):
        self.name = name
        """
        The name of the Root. Usually 'root'.
        """
        
        self.usage = initial_usage
        """
        Initial usage of the Root association.
        """

        self.new_child_usage = False
        """
        This is a boolean that signifies the usage has changed for 
        the children of this node.
        """

        self.children = []
        """
        Initialize the list of children as an empty list.
        """

        self.is_root = True
        """
        This is the root.
        """
        
        self.is_leaf = False
        """
        This is not a leaf.
        """

    def add_child(self, child):
        """
        Adds a child to the root.

        Arguments:
        - child: either an Account or a User object
        """

        # Add this child to the Root's list of children
        self.children.append(child)

        # Set the Root as the child's parent
        child.add_parent(self)

    def __str__(self):
        """
        Return a string for the Root. This will recursively call __str__
        and traverse the entire tree, concatenating the level of the tree 
        to the name of each child.
        """
        ret = "root\n"
        for child in self.children:
            ret += child.__str__(1)
        return ret



class Account:
    """
    Handles Accounts for the Fair Share algorithm.
    
    {Account, User} is the unique identifier (XXX Need to revisit), so a job will point to an Account 
    and User (each User may belong to multiple accounts). This Account and User will have their usages 
    updated, then the usage will be propagated up the tree.

    """
    def __init__(self, name, shares, initial_usage):
        """
        Initialize an Account.
        
        Arguments:
        - name: the Account name
        - shares: the amount of shares given to this Account
        - initial_usage: the usage already incurred by this Account
        """
        
        self.name = name
        """
        The Account name. These can be found in the sacctmgr_assocs.csv file.
        Account names are anonymized to Acc###.
        """
        
        self.shares = shares
        """
        The shares provided to this Account.
        """
        
        self.usage = initial_usage
        """
        The usage already incurred by this Account.
        """

        self.new_child_usage = False
        """
        This is a boolean that signifies the usage has changed for 
        the children of this node.
        """

        self.children = []
        """
        Initialize the list of children as an empty list.
        """
        
        self.parent = None
        """
        Initialize the Account without a parent.
        """

        self.is_root = False
        """
        This is not the Root.
        """
        
        self.is_leaf = False
        """
        This is an Account, not a leaf.
        """

        self.levelfs = np.inf if shares else 0
        """
        Initialize the Level FS, which is used to rank the Account.
        """

    def add_parent(self, parent):
        """
        Set the parent for this Account.

        Arguments:
        - parent: the parent for this Account
        """
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def add_child(self, child):
        """
        Add a child to the list of children for this Account.

        Arguments:
        - child: the child to be added, either an Account or a User.
        """
        # Add the child to the list of children
        self.children.append(child)

        # Set this Account as the parent of this child
        child.add_parent(self)

    def __str__(self, level=0):
        """
        Return a string for this Account. This will recursively call __str__
        and traverse the entire subtree of which this Account is the root,
        concatenating the level of the subtree to the name of each child.
        """
        ret = "\t"*level + self.name + "\n"
        for child in self.children:
            ret += child.__str__(level + 1)
        return  ret



class User:
    """
    Handles Users for the Fair Share algorithm.

    Users with the same name may coexist under different accounts.
    """
    def __init__(self, name, shares, initial_usage, partition, max_jobs, max_submit):
        """
        Initialize the User.

        Arguments:
        - name: the User name.
        - shares: the amount of shares given to this User
        - initial_usage: the usage already incurred by this User
        - partition: the partition associated with the row of this User
        - max_jobs: the max jobs running listed in the row of this User
        - max_submit: the max jobs submitted listed in the row of this User
        """
        
        self.name = name
        """
        The User name. These can be found in the sacctmgr_assocs.csv file.
        User names are anonymized to User###.
        """
        
        self.partition = partition
        
        self.shares = shares
        """
        The amount of shares given to this User.
        """
        
        self.usage = initial_usage
        """
        The usage already incurred by this User.
        """

        self.max_jobs = max_jobs
        self.max_submit = max_submit

        self.parent = None
        """
        Initialize the User without a parent.
        """

        self.is_root = False
        """
        This is not the root.
        """
        
        self.is_leaf = True
        """
        This is a User, so it is a leaf.
        """

        self.levelfs = np.inf if shares else 0
        """
        Initialize the Level FS, which is used to rank the User.
        """
        
        self.fairshare_factor = 1.0
        """
        Initialize the Fairshare Factor, which is similar to:
        (this user's rank) / (total number of User associations)
        """

    def add_parent(self, parent):
        """
        Set the parent for this User.

        Arguments:
        - parent: the parent for this User
        """
        if self.parent:
            raise ValueError("Already assigned a parent!")
        self.parent = parent

    def __str__(self, level=0):
        """
        Return a string for this User.
        
        Arguments:
        - level: the level of tree/subtree where this User is found.
        """
        ret = "\t"*level + self.name + "\n"
        return ret


class FairTree:
    """
    This is the FairTree used by the FairShare algorithm.
    """
    def __init__(
        self, assoc_source, calc_period, decay_halflife, init_time, active_usrs, excess_usr_assocs,
        partitions
    ):
        """
        Initialize the FairTree.

        Arguments:
        - assoc_source: path to sacctmgr_assocs.csv, or a prepared DataFrame with
                        the same columns (e.g. synthesized from the job trace)
        - calc_period: PriorityCalcPeriod from the Slurm configuration.
        - decay_halflife: PriorityDecayHalfLife from the Slurm configuration.
        - init_time: Minimum job start time
        - active_usrs: the set of all users that submitted jobs
        - excess_usr_assocs: a simulator-specific parameter that approximates the number of excess
                             associations. approx_excess_assocs removes a number of unused in workload 
                             traceassocs from the assoc tree, this is relevant since the fairshare factor 
                             scales with the tot number of user assocs. This happened because to capture 
                             all assocs they need to be dumped "withDeleted" so you end up with some assocs 
                             that never existed at any given time.
        - partitions: the main Partitions object 
                        partitions:         the set of all Partition objects
                        partitions_by_name: a dictionary of all Partition objects indexed by their string name
                        nodes:              the set of all Node objects
                        reservations:       a dictionary of all reservations. Initialized as an empty set here.
                        free_blocks:        a dictionary keeping track of the intervals within which nodes are  
                                            available for specified reservations.

        """
        
        self.last_calc_time = init_time
        """
        The last time the FairTree was calculated.
        Initialize this to the initial simulation time.
        """
        
        self.calc_period = calc_period
        """
        PriorityCalcPeriod from the Slurm configuration.
        """
        
        self.decay_constant = (1 - np.log(2) / decay_halflife.total_seconds())
        """
        decay constant for 1 second applied for the duration of a calc interval
        https://en.wikipedia.org/wiki/Half-life
        decay constant = ln(2)/halflife
        This is how much of the original quantity decays every unit of time. 
        We can apply this decay constant by multiplying the quantity by 1 - (decay constant) 
        at every timestep. So the decay_constant here is really:
        1 - (the decay constant we see in the wikipedia equation).
        """
        
        self.decay_constant_this_traversal = None
        """
        XXX Need to revisit this. Seems like a bookkeeping parameter.
        """

        self.root_node, flat_tree = self._load_tree_slurm(
            assoc_source, active_usrs, excess_usr_assocs, partitions
        )
        """
        Get the root node and a list of all nodes (users and accounts) in the tree.
        Nodes are added one level at a time.
        """
        
        self.levels = [[self.root_node]]
        """
        A 2D list with a list for each level, and all users and accounts at each
        level within each level list. Initialized as one level with only the root node.
        """

        self._tie_eps = 1e-12  
        """
        Stable, conservative tolerance for LevelFS ties
        """

        # Populate self.levels with nodes at each level.
        current_level = 0
        while(len(self.levels) > current_level): # If no children were added at the previous level, then
                                                 # a new level wasn't added but the count was incremented,
                                                 # and we are done.
            all_level_children = []
            for node in self.levels[current_level]:
                if node.is_leaf: # Only User nodes are leaves, they have no children.
                    continue
                for child_node in node.children:
                    all_level_children.append(child_node)

            if all_level_children: # If any children have been added to the list
                self.levels.append(all_level_children) # add this list as a new level

            current_level += 1

        
        self.assocs, self.tot_num_assocs = {}, 0
        """
        Initialize the set of all associations and a total count of associations.
        """

        # Populate (and count) the set of all associations
        # Each node is either a User or an Account object (or it is the root)
        for node in flat_tree:
            if not node.is_leaf: # Only include User associations
                continue

            # If this node is not associated with a specific partition,
            # then it is associated with all partitions.
            if node.partition is None:
                for partition in partitions.partitions:
                    self.assocs[(node.name, partition, node.parent.name)] = node
            else:
                self.assocs[(node.name, node.partition, node.parent.name)] = node

            self.tot_num_assocs += 1

        print("Num unique user assocs = {}".format(self.tot_num_assocs))

    def next_calc(self):
        """
        Get the time for the next FairShare calculation:
        Next Calculation Time = Last Calculation Time + Calculation Period
        """
        return self.last_calc_time + self.calc_period

    def job_finish_usage_update(self, job, time):
        """
        Update the usage
        """
        # This is just how it's implemented in source code as far as I can tell
        # This is the relevant Slurm code:
        # https://github.com/SchedMD/slurm/blob/master/src/plugins/priority/multifactor/priority_multifactor.c
        # Line 1116: run_delta = difftime(end_period, start_period);
        #            if (run_delta < 0)
		#                 run_delta = 0;
        # Line 1177: run_decay = run_delta * pow(decay_factor, run_delta);
        # Line 1205: tres_run_decay[i] = (long double)run_decay * (long double)job_ptr->tres_alloc_cnt[i];

        # Get the new elapsed time for this job.
        # We are updating usage for this job, so we use the last_calc_time (last time FairShare was
        # calculated) if the job started before then (because the difference between the two was
        # already accounted for the last time we calculated the usage); and we use the current time
        # if the job has not yet reached it's end limit, but we use the end limit time if it is in the past,
        # because we don't want to count time beyond the end limit of the job.
        delta_t = max(
            (min(job.endlimit, time) - max(self.last_calc_time, job.start)).total_seconds(), 0
        )

        # The old equation seems incorrect. From what I can tell, it should be:
        # usage = job.nodes * delta_t * (self.decay_constant ** delta_t)
        # This also would match with the usage equation in fairshare_calc below.
        # Changing now, keeping track of old equation: -KM
        # OLD EQUATION: usage = job.nodes * delta_t ** (self.decay_constant ** delta_t)
        usage = job.nodes * delta_t * (self.decay_constant ** delta_t) # NEW EQUATION 

        # self.assocs is a dictionary of user nodes indexed by associations (user, partition, account)
        user_node = self.assocs[job.assoc]

        # Update this user/partition/account association with this usage
        self._update_usages(user_node, usage)

        

    def fairshare_calc(self, running_jobs, time):
        """
        This is the main fair share calculation function.

        Arguments:
        - running_jobs: a list of Job objects for currently running jobs
        - time: the current time
        """
        # Get the remaining proportion based on the time elapsed since the last calculation
        remaining_proportion = self.decay_constant ** (time - self.last_calc_time).total_seconds()

        # Decay all usages by this proportion
        # New usage = old usage * remaining proportion
        self._decay_all(remaining_proportion)

        # Collect usages from running jobs between now and last_calc_time
        for job in running_jobs:
            # Calculate elapsed time for new usage
            # Use job.start if job started after last calc time, otherwise use last calc time
            delta_t = (time - max(job.start, self.last_calc_time)).total_seconds()

            # Calculate the usage for this job
            usage = job.nodes * delta_t * (self.decay_constant ** delta_t)

            # self.assocs is a dictionary of user nodes indexed by associations (user, partition, account)
            user_node = self.assocs[job.assoc]

            # Update this user association with this usage
            self._update_usages(user_node, usage)

        # Compute levelFS and sort (decay past usages as we go)
        self._tree_traversal(self.root_node)

        self.last_calc_time = time

    
    def _decay_all(self, remaining_proportion):
        """
        Decay all nodes (accounts and users) due to elapsed time since
        the last fairshare calculation. The remaining proportion was calculated as:
        remaining_proportion = decay_constant ** (current_time - last_calc_time)
        """
        for level in self.levels:
            for node in level:
                node.usage *= remaining_proportion

    def update_shares(self, current_node, account_shares):
        """
        Update shares for all Account nodes in fairtree.

        Parameters:
        - current_node: starts with Root
        - account_shares: dictionary of new account share indexed by account name for all accounts
        """
        if current_node.is_leaf:
            return
        else:
            current_node.shares = account_shares[current_node.name]
            
        for child_node in current_node.children:
            self.update_shares(child_node, account_shares)
        
    
    def _tree_traversal(self, current_node, rank=-1, last_leaf_levelfs=None, tie_cnt=0):
        """
        Recursively compute levelFS and sort (decay past usages as we go)

        NOTE: Ties between sibling users handled properly but not implemented ties 
              between accounts (merge children and sort)

        Arguments:
        - current_node: starts with the root node
        - rank: the current rank
        - last_leaf_levelfs: the LevelFS of the last leaf that was encountered
        - tie_cnt: the current number of ties at the current LevelFS

        Returns:
        - rank: the current rank
        - last_leaf_levelfs: the LevelFS of the last leaf that was encountered
        - tie_cnt: the current number of ties at the current LevelFS
        """
        # If the current node is a leaf (it is a User node)
        # We need to calculate the fairshare factor for this node.
        if current_node.is_leaf:
            # if current_node.levelfs == last_leaf_levelfs:
            if last_leaf_levelfs is not None and abs(current_node.levelfs - last_leaf_levelfs) <= self._tie_eps:
                # Increment the tie count when there is a tie
                tie_cnt += 1 
            else:
                # Only increment the rank when there is no longer a tie
                rank += 1 + tie_cnt 

                # Reset tie count
                tie_cnt = 0 

                # Set the last_leaf_levelfs to the current node's LevelFS
                last_leaf_levelfs = current_node.levelfs

            # Calculate the FairShare factor for this node.
            # tot_num_assocs was calculated when the tree was initialized.
            current_node.fairshare_factor = 1.0 - rank / self.tot_num_assocs

            # Since we are at a leaf node, go back up the tree.
            return rank, last_leaf_levelfs, tie_cnt

        # Handling Account nodes.
        # Avoid re-sorting when order of this node's children is unchanged
        # It's not possible for the order to change if the usage of the children 
        # hasn't been updated. When the usage of any of the Account node's children
        # has been updated, the new_child_usage boolean will be set to True.
        # See https://slurm.schedmd.com/fair_tree.html
        if current_node.new_child_usage: 
            # Calculate total shares and usage for all children of this node
            total_shares = 0
            total_usage = 0
            for child_node in current_node.children:
                total_shares += child_node.shares
                total_usage += child_node.usage

            # Get Level FS for each child
            for child_node in current_node.children:
                if child_node.usage:
                    # Update the child node's LevelFS if it has any usage.
                    S = child_node.shares / total_shares
                    U = child_node.usage / total_usage
                    child_node.levelfs = S / U
                else:
                    child_node.levelfs = np.inf if child_node.shares else 0
            # We have now updated the LevelFS of all the children
            # So we reset this flag to False
            current_node.new_child_usage = False

            # Sort these children by their LevelFS
            current_node.children.sort(key=lambda node: (node.levelfs, node.name), reverse=True)

        # Now that the children are sorted, we can traverse them in order.
        for child_node in current_node.children:
            rank, last_leaf_levelfs, tie_cnt = self._tree_traversal(
                child_node, rank=rank, last_leaf_levelfs=last_leaf_levelfs, tie_cnt=tie_cnt
            )

        # Once we have traversed all the nodes in the branch of which this node
        # is the root, go back up the tree.
        return rank, last_leaf_levelfs, tie_cnt

    def _update_usages(self, node, usage):
        """
        Update the usage of a Node and all of its ancestors.

        Arguments:
        - node: the node that incurred the new usage
        - usage: the new usage to add to the node
        """
        # Add this usage to this node
        node.usage += usage
        # Add this usage to the node's ancestors up to the Root node
        while not node.is_root:
            node = node.parent
            node.usage += usage
            # The children of this node have updated usage.
            node.new_child_usage = True

    def _load_tree_slurm(self, assoc_source, active_usrs, excess_usr_assocs, partitions):
        """
        Load the tree from the Slurm dump data.
        This function starts at the root node, then adds nodes level by level.

        Arguments:
        - assoc_source: path to sacctmgr_assocs.csv, or a prepared DataFrame with
                        the same columns (e.g. synthesized from the job trace)
        - active_usrs: the set of all users that submitted jobs
        - excess_usr_assocs: a simulator-specific parameter that approximates the number of excess
                             associations. (See above for more thorough explanation)
        - partitions: the main Partitions object (See above for more thorough explanation)
        """

        # Read association data from sacctmgr_assocs.csv
        # Old Columns: User|Account|ParentName|Partition|MaxJobs|MaxSubmit
        # New Columns: Account|User|ParentName|Partition|Shares
        if isinstance(assoc_source, pd.DataFrame):
            assoc_df = assoc_source
        else:
            assoc_df = pd.read_csv(assoc_source, delimiter='|', lineterminator='\n', header=0)
        assoc_df = assoc_df.drop([ col for col in assoc_df.columns if "Unnamed" in col ], axis=1)

        if "Shares" not in assoc_df.columns:
            msg = (
                f"[WARN] 'Shares' column not found in {assoc_source}. "
                "Defaulting Shares=1 for all associations."
            )
            print(msg)

            assoc_df["Shares"] = 1

        # Initialize the Root node with 0 initial usage, with name 'root'
        root_node = Root(0.0)

        # Initialize the list of nodes at the current level with the root node
        level_nodes = [root_node]

        # Initialize the flattened tree with the root node
        flat_tree = [root_node]

        # No user associations have been removed
        # usr_assocs_removed = 0

        while level_nodes: # While there are nodes at the current level
            next_level_nodes = [] # Create an empty list for nodes at the next level

            for node in level_nodes: # For each node at this level
                # Add accounts to the tree
                # Filter the associations dataframe to include only rows where the association Parent is this Node
                for _, child_row in assoc_df.loc[(assoc_df.ParentName == node.name)].iterrows(): 
                    # Create an Account with Account(name, shares, initial_usage)
                    acc = Account(child_row.Account, child_row.Shares, 0.0)

                    # Add this account to the flat tree
                    flat_tree.append(acc)

                    # Add this account as a child of this node
                    node.add_child(acc)

                    # Add this account to the list of nodes at the next level
                    next_level_nodes.append(acc)

                    # Add users to the tree
                    # Filter the associations DataFrame to include rows with the same Account and a value for User
                    for _, usr_row in (
                        assoc_df.loc[(assoc_df.Account == acc.name) & (assoc_df.User.notna())].iterrows()
                    ):
                        # Don't add this user if the user never submitted a job and
                        # we have not yet removed enough associations to meet the approximate
                        # number of excess associations (again, this is a simulator-specific 
                        # parameter that was created as a workaround to inaccuracies in the
                        # associations data).
                        # Removing this for now - we should be handling this in the data_reader
                        # module. After getting df_jobs, we can pass df_jobs into a get_associations
                        # method, and filter out any user-account rows that don't show up in df_jobs. -KM
                        # If a user-account association doesn't show up 
                        # if (
                        #     usr_assocs_removed < excess_usr_assocs and
                        #     usr_row.User not in active_usrs
                        # ):
                        #     usr_assocs_removed += 1
                        #     continue

                        # If there is a value for Partition in this row, and it is one of the
                        # considered partitions listed in the config file, then get the Partition
                        # object with this name.
                        if pd.isna(usr_row.Partition):
                            partition = None
                        elif usr_row.Partition in partitions.partitions_by_name:
                            partition = partitions.partitions_by_name[usr_row.Partition]
                        else:
                            continue # There is a partition value, but it's not in the considered partitions.

                        
                        # Set max_jobs and max_submit if columns exist and data is present in the row
                        max_jobs = (
                            int(usr_row["MaxJobs"])
                            if "MaxJobs" in assoc_df.columns and pd.notna(usr_row["MaxJobs"])
                            else None
                        )
                        max_submit = (
                            int(usr_row["MaxSubmit"])
                            if "MaxSubmit" in assoc_df.columns and pd.notna(usr_row["MaxSubmit"])
                            else None
                        )

                        # Create a User from the user in this row
                        # User(name, shares, initial_usage, partition, max_jobs, max_submit)
                        # TODO: Set initial usage based on historic usage before simulator start time
                        usr = User(usr_row.User, usr_row.Shares, 0.0, partition, max_jobs, max_submit)

                        # Add this user to the flat tree
                        flat_tree.append(usr)

                        # Add this user as a child of this Account
                        acc.add_child(usr)

            # Go to the next level
            level_nodes = next_level_nodes

        return root_node, flat_tree

    def __str__(self):
        ret = (
            "\t".join(
                [ "l{} ({})".format(level, len(nodes)) for level, nodes in enumerate(self.levels) ]
            ) +
            "\n"
        )
        ret += self.levels[0][0].__str__()
        return ret

