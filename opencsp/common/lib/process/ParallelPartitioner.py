import math
from typing import TypeVar

import opencsp.common.lib.tool.log_tools as lt

T = TypeVar("T")


class ParallelPartitioner:
    """Helps portion out data to be executed per cpu, per server.

    This class is here as a simple solution for parallel execution of
    trivially parallelizable problems. For anything more complicated,
    consider using a different parallel execution framework, such as
    Dask.
    """

    def __init__(self, nservers: int, server_idx: int, ncpus: int, cpu_idx: int, npartitions_ceil: int = -1):
        """Helps portion out data to be excecuted for a single server+cpu instance.

        Typical usage is to get all partitioners for all cores with the generator::

            partitioners = ParallelPartioner.get_partitioners(nservers, serv_idx, ncpus)

        Args:
            - nservers (int): How many servers this program is running on.
            - server_idx (int): Which server index this is (starting at 0).
            - ncpus (int): How many processor cores per server this program is running on.
            - cpu_idx (int): Which processor index this is (starting at 0).
            - npartitions_ceil (int): A soft limit on the number of partitions to split the data into. -1 for no limit. Defaults to -1.
        """
        self.nservers = nservers
        self.server_idx = server_idx
        self.ncpus = ncpus
        self.cpu_idx = cpu_idx
        self.npartitions_ceil = npartitions_ceil

        if nservers > 0:
            if server_idx < 0 or server_idx >= nservers:
                raise ValueError(f"server_idx {server_idx} is out of range for nservers={nservers}")
        if ncpus > 0:
            if cpu_idx < 0 or cpu_idx >= ncpus:
                raise ValueError(f"cpu_idx {cpu_idx} is out of range for ncpus={ncpus}")

    @classmethod
    def get_partitioners(cls, nservers: int, server_idx: int, ncpus: int, npartitions_ceil: int = -1):
        """Generate partitioners to split data into even chunks for each node and cpu core.

        There are some cases where a task can be parallelized across many server nodes
        but not across cpu cores (such as ffmpeg, which is already parallelized across
        all cores). In these cases use 1 for the ncpus argument.

        Args:
            - nservers (int): The number of server that this code is running on.
            - server_idx (int): The node index of this server (starts at 0).
            - ncpus (int): The number of cpus on this node.
            - npartitions_ceil (int, optional): A soft limit on the number of partitions to split the data into. -1 for no limit. Defaults to -1.

        Returns:
            - list[ParallelPartitioner]: A list of parallel partitioners, one for each core for the given server_idx.
        """
        return [cls(nservers, server_idx, ncpus, i, npartitions_ceil) for i in range(ncpus)]

    def _get_portion_range(
        self, count: int, nworkers: int, worker_idx: int, partitions_per_worker: int, npartitions: int = -1
    ):
        if npartitions < 0:
            npartitions = partitions_per_worker * nworkers

        # get the range of data to operate on
        rstart, rend = 0, count
        if nworkers > -1:
            rel_server_size_f = count / npartitions * partitions_per_worker
            rstart = math.floor(rel_server_size_f * (worker_idx))
            rstart_next = math.floor(rel_server_size_f * (worker_idx + 1))
            is_last_server = worker_idx == nworkers - 1
            if is_last_server:
                rend = count
            else:
                rend = rstart_next

        if rstart >= count:
            return -1, -1
        rend = int(min(rend, count))
        return rstart, rend

    def _get_portion(
        self, data: list[T], nworkers: int, worker_idx: int, partitions_per_worker: int, npartitions: int = -1
    ):
        rstart, rend = self._get_portion_range(len(data), nworkers, worker_idx, partitions_per_worker, npartitions)

        # check range bounds
        if rstart == -1 and rend == -1:
            ret: list[T] = []
            return ret

        # get the portion of data to operate on
        return data[rstart:rend]

    def get_my_range(self, data: list[T], desc: str = None):
        """Like get_my_portion(), but returns the start and end indices instead of a subselection of data.

        If desired, the subselection of data can be retrieved with this method. Example::

            rstart, rend = partioner.get_my_range(data)
            if rstart == -1 and rend == -1:
                data_subselection = []
            else:
                data_subselection = data[rstart:rend]

        This is equivalent to::

            data_subselection = partitioner.get_my_portion(data)

        Returns:
            - int: the start index (inclusive), or -1 if this partitioner has a range of length 0
            - int: the stop index (exclusive), or -1 if this partitioner has a range of length 0
        """
        if desc != None:
            lt.info(f"Subselecting {desc} for parallel execution")
        npartitions_per_server: int = self.ncpus

        # check max partitions (server)
        if self.npartitions_ceil > 0:
            npartitions_per_server = int(max(self.npartitions_ceil / self.nservers, 1.0))

        # get the portion of data that this server should operate on
        rstart_serv, rend_serv = self._get_portion_range(
            len(data), self.nservers, self.server_idx, npartitions_per_server
        )
        if rstart_serv == -1 and rend_serv == -1:
            return -1, -1

        # check max partitions (cpu)
        if self.cpu_idx >= npartitions_per_server:
            return -1, -1

        # Get a subselection of the data for this cpu
        rstart_cpu, rend_cpu = self._get_portion_range(rend_serv - rstart_serv, self.ncpus, self.cpu_idx, 1)
        rstart_cpu += rstart_serv
        rend_cpu += rstart_serv

        return rstart_cpu, rend_cpu

    def get_my_portion(self, data: list[T], desc: str = None):
        """Get the subarray of the data list to operate on for the node + core that this partitioner represents.

        The data list is guaranteed to be split:
            * as evenly as possible
            * such that each element is in exactly one partition amongst all nodes and cpu core

        The data is first split evenly by server node index, and then by cpu core index.
        Because of this, it is recommended to use this class with servers that have
        identical performance characteristics (eg all have the same processor with the
        same number of cores).

        Args:
            data (list): The data list to operate on.
            desc (str, optional): A description of the data. If not none, then an info message will be printed. Defaults to None.

        Returns:
            list[T]: A light-weight subarray of data with the portion for this node + core to operate on.
        """
        rstart, rend = self.get_my_range(data, desc)
        if rstart == -1 and rend == -1:
            ret: list[T] = []
            return ret
        return data[rstart:rend]

    def server_identifier(self):
        """
        Generate a zero-padded string identifier for the server index.

        The length of the identifier is determined by the number of servers.
        For example, if there are 10 servers, the identifier will be a 2-digit string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A zero-padded string identifier for the server index.

        Examples
        --------
        >>> partitioner = ParallelPartitioner(nservers=10, server_idx=5)
        >>> partitioner.server_identifier()
        '05'
        """
        server_idx_len = len(str(self.nservers))
        return f"%0{server_idx_len}d" % self.server_idx

    def cpu_identifier(self):
        """
        Generate a zero-padded string identifier for the CPU index.

        The length of the identifier is determined by the number of CPUs.
        For example, if there are 10 CPUs, the identifier will be a 2-digit string.

        Parameters
        ----------
        None

        Returns
        -------
        str
            A zero-padded string identifier for the CPU index.

        Examples
        --------
        >>> partitioner = ParallelPartitioner(ncpus=10, cpu_idx=5)
        >>> partitioner.cpu_identifier()
        '05'
        """
        cpu_idx_len = len(str(self.ncpus))
        return f"%0{cpu_idx_len}d" % self.cpu_idx

    def identifier(self):
        """Filename-friendly format for identifying this partitioner."""
        return f"{self.server_identifier()}_{self.cpu_identifier()}"
