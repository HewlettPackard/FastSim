# TODO Implement partitions as objects attached to the nodes rather than having to got through this
# dictionary using the name
class MFPrioritySorter:
    def __init__(
        self, init_time, size_weight, age_weight, fairshare_weight, max_age, partition_weight,
        qos_weight, no_partition_priority_tiers
    ):
        self.size_weight = size_weight / 5860
        self.age_weight = age_weight
        self.fairshare_weight = fairshare_weight
        self.max_age = max_age.total_seconds()
        self.partition_weight = partition_weight
        self.qos_weight = qos_weight
        self.time = init_time

        self.fairtree = None

        # Relevant for ARCHER2
        self.no_partition_priority_tiers = no_partition_priority_tiers
        self.priority_factors = []
        if size_weight:
            self.priority_factors.append(self._size_priority)
        if age_weight:
            self.priority_factors.append(self._age_priority)
        if fairshare_weight:
            self.priority_factors.append(self._fairshare_priority)
        if partition_weight:
            self.priority_factors.append(self._partition_priority)
        if qos_weight:
            self.priority_factors.append(self._qos_priority)

    def sort(self, queue, time):
        self.time = time
        if self.no_partition_priority_tiers:
            queue.sort(
                key=lambda job: (
                    sum(priority_calc(job) for priority_calc in self.priority_factors),
                    self.time - job.submit,
                    job.id
                )
            )
            return

        queue.sort(
            key=lambda job: (
                self._partition_priority_tier(job),
                sum(priority_calc(job) for priority_calc in self.priority_factors),
                self.time - job.submit,
                job.id
            )
        )
        return

    def _partition_priority_tier(self, job):
        return job.partition.priority_tier

    def _age_priority(self, job):
        return (
            min((self.time - job.launch_time).total_seconds() / self.max_age, 1) * self.age_weight
        )

    def _size_priority(self, job):
        return job.nodes * self.size_weight

    def _fairshare_priority(self, job):
        return (
            self.fairtree.assocs[job.assoc].fairshare_factor *
            self.fairshare_weight
        )

    def _partition_priority(self, job):
        return self.partitions[job.partition].priority_weight * self.partition_weight

    def _qos_priority(self, job):
        return job.qos.priority * self.qos_weight

