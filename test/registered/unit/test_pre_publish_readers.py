"""The readers that run before ``publish`` must not ask a config bag.

``publish`` is what projects the bags; every accessor fails closed until then
(``config namespace 'observability' not published``). Most readers live deep in
a runtime path and are safely downstream of it, so converting a reader to a bag
is normally free. A handful are not: they run in the main process on the way
*to* publish, or they serve a ``ServerArgs`` that never publishes these bags at
all. Those must keep reading the record they were handed.

This is a class of defect no other test in the tree catches: the converted
reader is exercised everywhere by tests that publish first, so it passes unit
CI and then takes the server down on the first real launch. Each case below
pins one such reader against a context where nothing has been published.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import logging
import unittest

from sglang.srt.runtime_context import get_observability, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import configure_logger
from sglang.test.test_utils import CustomTestCase


class TestPrePublishReaders(CustomTestCase):
    def setUp(self):
        self._levels = {
            name: logging.getLogger(name).level
            for name in (None, "sglang", "httpx", "httpcore")
        }
        reset_context()
        self.addCleanup(self._restore_levels)
        self.addCleanup(reset_context)

    def _restore_levels(self):
        for name, level in self._levels.items():
            logging.getLogger(name).setLevel(level)

    def test_nothing_is_published_here(self):
        """The premise: this fixture really is a pre-publish context."""
        with self.assertRaises(ValueError) as caught:
            get_observability()
        self.assertIn("not published", str(caught.exception))

    def test_configure_logger_runs_before_publish(self):
        """``_launch_subprocesses`` calls it before ``publish``, and
        ``multimodal_gen`` calls it with a ``ServerArgs`` of its own that never
        publishes these bags. It reads the record it was handed."""
        server_args = ServerArgs(model_path="dummy", log_level="warning")
        configure_logger(server_args)
        self.assertEqual(logging.getLogger().level, logging.WARNING)

    def test_the_level_comes_from_the_record_it_was_handed(self):
        """Not from any ambient default: a second record moves the level."""
        configure_logger(ServerArgs(model_path="dummy", log_level="error"))
        self.assertEqual(logging.getLogger().level, logging.ERROR)


if __name__ == "__main__":
    unittest.main()
