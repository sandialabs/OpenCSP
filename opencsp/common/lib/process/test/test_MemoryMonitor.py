import time
import unittest

import opencsp.common.lib.process.MemoryMonitor as mm


class TestMemoryMonitor(unittest.TestCase):
    def test_del(self):
        """Test that deleting a memory monitor does not raise exceptions."""
        monitor = mm.MemoryMonitor()
        try:
            monitor.start()
        finally:
            monitor.stop()

    def _wait_done(self, monitor: mm.MemoryMonitor, wait_time: int):
        start_time = time.time()
        while (elapsed := time.time() - start_time) < wait_time:
            if monitor.done():
                return
            time.sleep(0.1)

    def test_max_lifetime_hours(self):
        secs = 2.1
        monitor = mm.MemoryMonitor(max_lifetime_hours=(secs / (60 * 60)))
        monitor.start()
        time.sleep(1)
        self.assertFalse(monitor.done(), "Monitor should not have exited on its own within one second")
        self._wait_done(monitor, 4)
        self.assertTrue(monitor.done(), "Monitor should have exited on its own within 5 seconds")

    def test_zero_lifetime_hours(self):
        monitor = mm.MemoryMonitor(max_lifetime_hours=0)
        monitor.start()
        time.sleep(1)
        self.assertEqual(len(monitor._log), 0, "Monitor should not have had time to record any logs")
        self._wait_done(monitor, 4)
        self.assertTrue(monitor.done(), "Monitor should have exited on its own within 5 seconds")

    def test_stop(self):
        monitor = mm.MemoryMonitor()
        monitor.start()
        self.assertFalse(monitor.done())
        monitor.stop(wait=True)
        self.assertTrue(monitor.done(), "Monitor should have stopped immediately")

    def test_large_memory_usage(self):
        monitor = mm.MemoryMonitor()
        monitor.start()
        import opencsp.common.lib.tool.log_tools as lt
        lt.error(f"{monitor.min_usage()=}, {monitor.max_usage()=}")
        monitor2 = mm.MemoryMonitor(always_print=True, log_func=lt.error)
        monitor2.start()
        time.sleep(1)
        import tracemalloc
        tracemalloc.start()
        a = [1] * 1_000_000_000
        time.sleep(1.1)
        current, peak = tracemalloc.get_traced_memory()
        current, peak = current / 1e9, peak / 1e9
        lt.error(f"{current=}, {peak=}")
        tracemalloc.stop()
        monitor.stop(wait=True)
        monitor2.stop(wait=True)
        lt.error(f"{monitor.min_usage()=}, {monitor.max_usage()=}")
        self.assertGreaterEqual(monitor.max_usage() - monitor.min_usage(), 0.5)


if __name__ == '__main__':
    unittest.main()
