"""Device-free integration checks for failure artifacts from native ROS tools.

Set STEREOFORGE_ROS_BUILD to a Jazzy build directory and source its ROS setup.
"""
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

BUILD = os.environ.get("STEREOFORGE_ROS_BUILD")


@unittest.skipUnless(BUILD, "ROS build directory not supplied")
class StartupFailureTests(unittest.TestCase):
    def test_setup_failure_is_bounded_and_never_overwrites(self):
        cases = [
            ("stereoforge_ros2_depth_publisher", "publisher", [
                "--source", "camera", "--manifest", "/missing-manifest", "--clock-map", "/missing-clock-map",
                "--frame-id", "camera", "--run-id", "failure-test"]),
            ("stereoforge_ros2_delivery_probe", "subscriber", [
                "--frame-id", "camera", "--run-id", "failure-test", "--depth-topic", "/invalid//topic"]),
        ]
        for app, role, args in cases:
            with self.subTest(app=app), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "trace.json"
                command = [str(Path(BUILD) / app), *args, "--trace", str(path)]
                result = subprocess.run(command, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertTrue(0 < path.stat().st_size < 2048)
                failure = json.loads(path.read_text())
                self.assertEqual(failure["schema"], "stereoforge.ros_delivery_failure.v1")
                self.assertEqual(failure["kind"], "startup_or_output_failure")
                self.assertEqual(failure["role"], role)
                self.assertTrue(0 < len(failure["message"]) <= 512)
                self.assertTrue(result.stderr)
                path.write_text("existing evidence")
                result = subprocess.run(command, capture_output=True, timeout=30)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(path.read_text(), "existing evidence")


if __name__ == "__main__":
    unittest.main()
