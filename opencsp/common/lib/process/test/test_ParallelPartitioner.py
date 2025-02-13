import copy
import math
import unittest

import opencsp.common.lib.process.ParallelPartitioner as ppart


class TestParallelPartitioner(unittest.TestCase):
    def test_S1s0C1c0_list1(self):
        partitioner = ppart.ParallelPartitioner(nservers=1, server_idx=0, ncpus=1, cpu_idx=0)
        data = ["a"]
        portion = partitioner.get_my_portion(data)
        self.assertEqual(portion, data)

    def test_S1s0C1c0_list1000(self):
        partitioner = ppart.ParallelPartitioner(nservers=1, server_idx=0, ncpus=1, cpu_idx=0)
        data = ["a"] * 1000
        portion = partitioner.get_my_portion(data)
        self.assertEqual(len(portion), 1000)
        self.assertEqual(portion, data)

    def test_S2s0C1c0_list1(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=0, ncpus=1, cpu_idx=0)
        data = ["a"]
        portion = partitioner.get_my_portion(data)
        self.assertEqual(portion, [])

    def test_S2s1C1c0_list1(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=1, ncpus=1, cpu_idx=0)
        data = ["a"]
        portion = partitioner.get_my_portion(data)
        self.assertEqual(portion, data)

    def test_S2s0C1c0_list2(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=0, ncpus=1, cpu_idx=0)
        data = ["a", "b"]
        portion = partitioner.get_my_portion(data)
        self.assertEqual(portion, ["a"])

    def test_S2s1C1c0_list2(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=1, ncpus=1, cpu_idx=0)
        data = ["a", "b"]
        portion = partitioner.get_my_portion(data)
        self.assertEqual(portion, ["b"])

    def test_S2s0C1c0_list1000(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=0, ncpus=1, cpu_idx=0)
        data = (["a"] * 500) + (["b"] * 500)
        portion = partitioner.get_my_portion(data)
        self.assertEqual(len(portion), 500)
        self.assertEqual(portion, ["a"] * 500)

    def test_S2s1C1c0_list1000(self):
        partitioner = ppart.ParallelPartitioner(nservers=2, server_idx=1, ncpus=1, cpu_idx=0)
        data = (["a"] * 500) + (["b"] * 500)
        portion = partitioner.get_my_portion(data)
        self.assertEqual(len(portion), 500)
        self.assertEqual(portion, ["b"] * 500)

    def test_S50ssC1c0_list25000(self):
        alphabet = []
        for i in range(int(25000 / 50)):
            alphabet.append(chr((i % 25) + 65))
        data = []
        for i in range(50):
            data += copy.deepcopy(alphabet)

        for s in range(50):
            partitioner = ppart.ParallelPartitioner(nservers=50, server_idx=s, ncpus=1, cpu_idx=0)
            portion = partitioner.get_my_portion(data)
            self.assertEqual(len(portion), 25000 / 50)
            self.assertEqual(portion, alphabet)

    def test_S25ssC100cc_list25000(self):
        cpu_size = 25000 / 25 / 100
        alphabet = []
        for i in range(int(cpu_size)):
            alphabet.append(chr((i % 25) + 65))
        data, per_server = [], []
        for c in range(100):
            per_server += copy.deepcopy(alphabet)
        for s in range(25):
            data += copy.deepcopy(per_server)

    def test_S1s0CNcc_list100(self):
        """Verify list of 100 gets split event for any number N of cpus < 100*2."""
        data = list(range(100))

        for N in range(1, 100 * 2):
            data_portioned = []
            for c in range(N):
                partitioner = ppart.ParallelPartitioner(nservers=1, server_idx=0, ncpus=N, cpu_idx=c)
                portion = partitioner.get_my_portion(data)
                data_portioned += portion
                self.assertLessEqual(len(portion), math.ceil(100 / N))
            self.assertListEqual(data, data_portioned)


if __name__ == "__main__":
    unittest.main()
