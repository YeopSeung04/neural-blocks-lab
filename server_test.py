#!/usr/bin/env python3
import math
import unittest

import server


class SystemMetricsTest(unittest.TestCase):
    def test_metrics_shape(self):
        metrics = server.collect_metrics()
        self.assertIn("cpu", metrics)
        self.assertIn("memory", metrics)
        self.assertIn("gpu", metrics)
        cpu = metrics["cpu"]["usagePercent"]
        if cpu is not None:
            self.assertTrue(math.isfinite(cpu))
            self.assertGreaterEqual(cpu, 0)
            self.assertLessEqual(cpu, 100)
        memory = metrics["memory"]
        self.assertGreater(memory["totalBytes"], 0)
        if memory["usagePercent"] is not None:
            self.assertGreaterEqual(memory["usagePercent"], 0)
            self.assertLessEqual(memory["usagePercent"], 100)


if __name__ == "__main__":
    unittest.main()
