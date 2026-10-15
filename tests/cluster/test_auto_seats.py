"""Unit tests for the launcher's automatic seat count (scripts/dgpp-cluster).

Run from the repo root: python3 -m unittest tests/cluster/test_auto_seats.py
No node, GPU or network is used: the memory plan and the peers' memory are stubbed.
"""
import importlib.machinery
import importlib.util
import io
import os
import tempfile
import unittest
from contextlib import redirect_stdout

HERE = os.path.dirname(os.path.abspath(__file__))
LAUNCHER = os.path.join(HERE, "..", "..", "scripts", "dgpp-cluster")


def load_launcher():
    import sys
    scripts = os.path.abspath(os.path.join(HERE, "..", "..", "scripts"))
    if scripts not in sys.path:
        sys.path.insert(0, scripts)
    loader = importlib.machinery.SourceFileLoader("dgpp_cluster", LAUNCHER)
    spec = importlib.util.spec_from_loader("dgpp_cluster", loader)
    module = importlib.util.module_from_spec(spec)
    loader.exec_module(module)
    return module


class FakePlanRun:
    """Stands in for subprocess.run of `dgpp-serve --memory-plan`."""

    def __init__(self, totals):
        self.totals = totals  # seats -> plan total in GiB

    def __call__(self, argv, **kwargs):
        config = argv[argv.index("--config") + 1]
        import json
        with open(config) as f:
            seats = json.load(f)["engine"]["max_concurrency"]
        class Result:
            returncode = 0
            stdout = f"INFO memory plan total {self.totals[seats]:.2f} GiB (x)\n"
            stderr = ""
        return Result()


def make_launcher(launcher, nodes, mem_gib, plan_totals, log_dir):
    """A launcher object with just enough state for size_seats()."""
    obj = launcher.Cluster.__new__(launcher.Cluster)
    obj.cfg = {"engine": {}, "ports": {"http": 18300}, "nodes": list(nodes), "http": {"port": 18300}}
    obj.nodes = list(nodes)
    obj.bin = "/fake/dgpp-serve"
    obj.log_dir = log_dir
    obj.available_memory_gib = lambda rank, host: mem_gib[host]
    return obj


class AutoSeatTests(unittest.TestCase):
    def setUp(self):
        self.launcher = load_launcher()
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def run_size(self, obj, plan_totals, fraction=None):
        import unittest.mock as mock
        env = {} if fraction is None else {"DGPP_AUTO_MEMORY_FRACTION": str(fraction)}
        out = io.StringIO()
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(self.launcher.subprocess, "run", FakePlanRun(plan_totals)), \
                redirect_stdout(out):
            ok = obj.size_seats()
        return ok, out.getvalue()

    def test_formula_gives_per_node_minimum(self):
        # fixed 40 GiB, 2 GiB per seat (plan(2)-plan(1)=2, plan(1)=42 => fixed 40).
        plan = {1: 42.0, 2: 44.0}
        obj = make_launcher(self.launcher, ["a", "b"], {"a": 100.0, "b": 90.0}, plan, self.tmp.name)
        ok, out = self.run_size(obj, plan, fraction=0.96)
        self.assertTrue(ok)
        # node a: floor((0.96*100 - 40)/2) = 28 ; node b: floor((0.96*90 - 40)/2) = 23 -> min 23
        self.assertEqual(obj.cfg["engine"]["max_concurrency"], 23)
        self.assertEqual(obj.cfg["engine"]["kv_capacity"], 23 * 262144)
        self.assertEqual(obj.cfg["engine"]["queue_limit"], 46)
        self.assertIn("seats = 23", out)

    def test_seat_cap_applies(self):
        plan = {1: 10.0, 2: 10.5}  # cheap seats: formula would give hundreds
        obj = make_launcher(self.launcher, ["a"], {"a": 120.0}, plan, self.tmp.name)
        ok, out = self.run_size(obj, plan)
        self.assertTrue(ok)
        self.assertEqual(obj.cfg["engine"]["max_concurrency"], 31)
        self.assertIn("cap 31", out)

    def test_96_percent_is_the_default_fraction(self):
        plan = {1: 40.0, 2: 42.0}
        obj = make_launcher(self.launcher, ["a"], {"a": 100.0}, plan, self.tmp.name)
        ok, _ = self.run_size(obj, plan)
        self.assertTrue(ok)
        # fixed 38, slope 2: floor((96 - 38)/2) = 29
        self.assertEqual(obj.cfg["engine"]["max_concurrency"], 29)

    def test_explicit_integer_overrides_auto(self):
        plan = {1: 40.0, 2: 42.0}
        obj = make_launcher(self.launcher, ["a"], {"a": 100.0}, plan, self.tmp.name)
        obj.cfg["engine"]["max_concurrency"] = 12
        ok, out = self.run_size(obj, plan)
        self.assertTrue(ok)
        self.assertEqual(obj.cfg["engine"]["max_concurrency"], 12)
        self.assertEqual(out, "")  # no sizing runs when the count is explicit

    def test_too_little_memory_fails_clearly(self):
        plan = {1: 80.0, 2: 82.0}
        obj = make_launcher(self.launcher, ["a"], {"a": 50.0}, plan, self.tmp.name)
        ok, out = self.run_size(obj, plan)
        self.assertFalse(ok)
        self.assertIn("not enough memory for one seat", out)

    def test_plan_failure_is_reported(self):
        import unittest.mock as mock
        obj = make_launcher(self.launcher, ["a"], {"a": 100.0}, {1: 1.0, 2: 2.0}, self.tmp.name)
        class Bad:
            returncode = 1
            stdout = ""
            stderr = "ERROR refusing"
        out = io.StringIO()
        with mock.patch.object(self.launcher.subprocess, "run", lambda *a, **k: Bad()), redirect_stdout(out):
            ok = obj.size_seats()
        self.assertFalse(ok)
        self.assertIn("memory plan failed", out.getvalue())


class NoHardCodedSixteenTests(unittest.TestCase):
    def test_auto_path_has_no_literal_sixteen(self):
        with open(LAUNCHER) as f:
            src = f.read()
        start = src.index("    def size_seats(self):")
        end = src.index("    def _up(self):")
        body = src[start:end]
        self.assertNotIn("16", body.replace("262144", ""), "a literal 16 appears on the auto path")


if __name__ == "__main__":
    unittest.main()
