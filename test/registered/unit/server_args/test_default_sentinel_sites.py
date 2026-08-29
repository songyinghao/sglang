"""A family provider that asks "did the operator leave this alone?" must be the
first thing to touch the field.

Four sites in the per-model declarations decide "not set" by comparing the
config against the class default:

    if cfg.mamba_radix_cache_strategy == ServerArgs.mamba_radix_cache_strategy:
        overrides["mamba_radix_cache_strategy"] = "extra_buffer"

That reads the *resolving* view, which walks the declaration stash, so it
answers "unset" only while nothing earlier has declared the field. One pass can
declare one of these fields before the family providers run:
``_mamba_radix_cache_resolution``, invoked from
``handle_model_specific_adjustments`` at the slot just above
``collect_model_override_declarations`` -- but only for an architecture with a
linear-attention spec that uses the mamba radix cache.

No architecture using the sentinel has such a spec today, so all four are
correct. The coupling is invisible from either end though: giving Inkling a
linear-attn spec would leave its pin in place, still reading, and silently
never firing again. This is that coupling written down.

The comparison's other weakness -- an operator who passes the default value
looks like one who passed nothing -- is not something a test can settle. It
needs to know which flags were on the command line, which the record does not
carry.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import ast
import pathlib
import unittest

import sglang
from sglang.srt.configs.linear_attn_model_registry import get_linear_attn_spec_by_arch
from sglang.test.test_utils import CustomTestCase

_PACKAGE_ROOT = pathlib.Path(next(iter(sglang.__path__))) / "srt"
_DECLARATIONS = _PACKAGE_ROOT / "arg_groups" / "model_overrides"

# Fields a pass declares before the family providers run, and the pass that
# does it. A field is safe to compare against its default only where that pass
# does not fire.
_DECLARED_EARLY = {"mamba_radix_cache_strategy"}


def _sentinel_sites():
    """(module, field, architectures) for each `cfg.x == ServerArgs.x`."""
    sites = []
    for path in sorted(_DECLARATIONS.glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8-sig"))
        for function in [
            node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
        ]:
            architectures = [
                arg.value
                for decorator in function.decorator_list
                if isinstance(decorator, ast.Call)
                and getattr(decorator.func, "id", "") == "_register_for"
                for arg in decorator.args
                if isinstance(arg, ast.Constant)
            ]
            if not architectures:
                continue
            for node in ast.walk(function):
                if not (
                    isinstance(node, ast.Compare)
                    and len(node.ops) == 1
                    and isinstance(node.ops[0], ast.Eq)
                ):
                    continue
                left, right = node.left, node.comparators[0]
                if (
                    isinstance(left, ast.Attribute)
                    and isinstance(right, ast.Attribute)
                    and getattr(right.value, "id", None) == "ServerArgs"
                    and left.attr == right.attr
                ):
                    sites.append((path.name, left.attr, tuple(architectures)))
    return sites


class TestDefaultSentinelSites(CustomTestCase):
    def test_the_scan_finds_the_sentinels(self):
        sites = _sentinel_sites()
        self.assertTrue(
            sites,
            "no `cfg.x == ServerArgs.x` comparison found in the per-model "
            "declarations; the scan is broken, or the pattern is gone and this "
            "file can go with it",
        )

    def test_no_sentinel_field_is_declared_before_its_provider_runs(self):
        for module, field, architectures in _sentinel_sites():
            if field not in _DECLARED_EARLY:
                continue
            for architecture in architectures:
                with self.subTest(module=module, field=field, arch=architecture):
                    spec = get_linear_attn_spec_by_arch(architecture)
                    self.assertFalse(
                        spec is not None and spec.uses_mamba_radix_cache,
                        f"{module} decides whether the operator set {field!r} by "
                        f"comparing against the class default, but {architecture} "
                        f"now has a linear-attention spec that uses the mamba "
                        f"radix cache -- so _mamba_radix_cache_resolution declares "
                        f"{field!r} at the slot above "
                        f"collect_model_override_declarations, the comparison is "
                        f"false before the provider ever looks, and the pin "
                        f"silently stops firing. Read the raw input instead.",
                    )


if __name__ == "__main__":
    unittest.main()
