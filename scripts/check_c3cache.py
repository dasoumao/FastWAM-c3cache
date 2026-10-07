"""Lightweight C3ache checks; no model imports, inference, or GPU required.

Run with ``python scripts/check_c3cache.py``.
"""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
CACHE = runpy.run_path(str(ROOT / "src/fastwam/models/wan22/c3cache.py"))
C3Cache = CACHE["C3Cache"]
validate_c3cache_range = CACHE["validate_c3cache_range"]
validate_c3cache_residual_space = CACHE["validate_c3cache_residual_space"]


def load_tensor_cores():
    """Extract only the two pure methods, avoiding the heavy model imports."""
    source = ROOT / "src/fastwam/models/wan22/fastwam.py"
    tree = ast.parse(source.read_text())
    fastwam = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastWAM")
    names = {"_denoise_action_c3cache_refresh", "_denoise_action_c3cache_reuse"}
    methods = [node for node in fastwam.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(methods) == 2
    module = ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[]))
    namespace = {"torch": SimpleNamespace(Tensor=float)}
    exec(compile(module, str(source), "exec"), namespace)
    return namespace


CORES = load_tensor_cores()


class FakeActionExpert:
    @staticmethod
    def action_encoder(value):
        return 2 * value + 1

    def prepare(self, action_tokens, timestep, context, context_mask):
        del context, context_mask
        return self.action_encoder(action_tokens), None, timestep, None, None, None

    @staticmethod
    def post(tokens):
        return 5 * tokens + 7


class FakeMoT:
    def __init__(self):
        self.calls = 0

    def forward_action_with_video_cache_tensor(self, **kwargs):
        self.calls += 1
        return kwargs["action_tokens"] + 3 * kwargs["video_cache_k"][0] + 2 * kwargs["action_t_mod"] + 1


class FakeModel:
    _denoise_action_c3cache_refresh = CORES["_denoise_action_c3cache_refresh"]
    _denoise_action_c3cache_reuse = CORES["_denoise_action_c3cache_reuse"]

    def __init__(self):
        self.action_expert = FakeActionExpert()
        self.mot = FakeMoT()


def sample_chunk(model, cache, signature, image_value, interval=4, start=0, end=6, proprio_value=0,
                 residual_space="hidden"):
    """A scalar Euler sampler using the actual cache policy and tensor cores."""
    cache.begin(signature, residual_space=residual_space)
    current = 1.0
    staged = {}
    full = reused = 0
    for step in range(10):
        if cache.should_reuse(step, start, end, interval):
            prediction = model._denoise_action_c3cache_reuse(
                current, cache.residuals[step], residual_space=residual_space
            )
            reused += 1
        else:
            prediction, residual = model._denoise_action_c3cache_refresh(
                current, step, None, None, [image_value + proprio_value], [], None,
                residual_space=residual_space,
            )
            if start <= step <= end:
                staged[step] = residual
            full += 1
        current += -0.01 * prediction
    cache.commit(staged, full, reused)
    return current


def full_sample_chunk(model, image_value):
    current = 1.0
    for step in range(10):
        prediction, _ = model._denoise_action_c3cache_refresh(
            current, step, None, None, [image_value], [], None
        )
        current += -0.01 * prediction
    return current


class C3CacheChecks(unittest.TestCase):
    def test_inclusive_schedule_and_refresh(self):
        model, cache = FakeModel(), C3Cache()
        outputs = [sample_chunk(model, cache, ("same",), image) for image in (10, 20, 30, 40, 50)]
        self.assertEqual(model.mot.calls, 29)
        self.assertEqual(cache.stats()["cached_steps"], tuple(range(7)))
        self.assertEqual(cache.stats()["full_steps"], 29)
        self.assertEqual(cache.stats()["reused_steps"], 21)
        self.assertNotEqual(outputs[0], outputs[1])
        self.assertNotEqual(outputs[3], outputs[4])
        self.assertEqual(cache.residuals[6], 3 * 50 + 2 * 6 + 1)

    def test_tau_zero_and_signature_invalidation(self):
        model, cache = FakeModel(), C3Cache()
        sample_chunk(model, cache, ("a",), 10, interval=0)
        sample_chunk(model, cache, ("a",), 20, interval=0)
        sample_chunk(model, cache, ("a",), 30, interval=0)
        self.assertEqual((cache.full_steps, cache.reused_steps), (16, 14))
        sample_chunk(model, cache, ("different_prompt",), 40, interval=0)
        self.assertEqual((cache.chunk_index, cache.full_steps, cache.reused_steps), (1, 10, 0))
        self.assertEqual(cache.residuals[0], 121)
        cache.reset()
        self.assertEqual(cache.stats()["cached_steps"], ())

    def test_refresh_boundaries_and_nonzero_range(self):
        for interval, chunks, expected_full, expected_reused in (
            (1, 3, 30, 0),
            (8, 9, 41, 49),
        ):
            with self.subTest(interval=interval):
                model, cache = FakeModel(), C3Cache()
                outputs = []
                for chunk in range(chunks):
                    outputs.append(sample_chunk(model, cache, ("task",), image_value=chunk, interval=interval))
                self.assertEqual((cache.full_steps, cache.reused_steps), (expected_full, expected_reused))
                self.assertEqual(model.mot.calls, expected_full)
                if interval == 1:
                    baseline_model = FakeModel()
                    self.assertEqual(
                        outputs,
                        [full_sample_chunk(baseline_model, chunk) for chunk in range(chunks)],
                    )
        model, cache = FakeModel(), C3Cache()
        sample_chunk(model, cache, ("task",), image_value=1, start=2, end=4)
        sample_chunk(model, cache, ("task",), image_value=2, start=2, end=4)
        self.assertEqual(cache.stats()["cached_steps"], (2, 3, 4))
        self.assertEqual((cache.full_steps, cache.reused_steps), (17, 3))

    def test_changing_image_and_proprio_preserves_episode(self):
        model, cache = FakeModel(), C3Cache()
        first = sample_chunk(model, cache, ("prompt", "same task"), image_value=1, proprio_value=2)
        second = sample_chunk(model, cache, ("prompt", "same task"), image_value=5, proprio_value=7)
        self.assertNotEqual(first, second)
        self.assertEqual((cache.chunk_index, cache.full_steps, cache.reused_steps), (2, 13, 7))
        source = (ROOT / "src/fastwam/models/wan22/fastwam.py").read_text().split("    def infer_action(", 1)[1]
        self.assertLess(
            source.index("c3cache_context_identity ="),
            source.index("context, context_mask = self._append_proprio_to_context("),
        )
        self.assertIn("c3cache_context_identity,", source)
        signature_body = source.split("c3cache_signature = (", 1)[1].split("timestep_video =", 1)[0]
        self.assertNotIn("proprio", signature_body)
        self.assertNotIn("_c3cache_tensor_digest(input_image)", signature_body)

    def test_refresh_and_reuse_arithmetic(self):
        model = FakeModel()
        full, residual = model._denoise_action_c3cache_refresh(2.0, 3, None, None, [10], [], None)
        self.assertEqual(residual, 37)
        self.assertEqual(full, 5 * ((2 * 2 + 1) + 37) + 7)
        self.assertEqual(model._denoise_action_c3cache_reuse(4.0, residual), 5 * ((2 * 4 + 1) + 37) + 7)
        self.assertEqual(model.mot.calls, 1)

    def test_velocity_residual_cancels_head_bias_and_uses_current_input(self):
        model = FakeModel()
        full, residual = model._denoise_action_c3cache_refresh(
            2.0, 3, None, None, [10], [], None, residual_space="velocity"
        )
        self.assertEqual(full, 217)
        self.assertEqual(residual, 185)  # 5 * hidden residual 37, NOT head(37)=192.
        reused = model._denoise_action_c3cache_reuse(4.0, residual, residual_space="velocity")
        self.assertEqual(reused, 237)
        self.assertNotEqual(reused, full)  # Not reusing the previous velocity itself.
        self.assertEqual(model.mot.calls, 1)  # Reuse skips the DiT.

    def test_velocity_cache_has_action_dimension_with_rectangular_head(self):
        class Vector(tuple):
            def __add__(self, other):
                assert len(self) == len(other)
                return Vector(a + b for a, b in zip(self, other))

            def __sub__(self, other):
                assert len(self) == len(other)
                return Vector(a - b for a, b in zip(self, other))

        def encode(x):
            return Vector((2 * x[0] + 1, 3 * x[1] - 2, x[0] - x[1]))

        def head(h):
            return Vector((h[0] + 2 * h[1] - h[2] + 7, 3 * h[0] + h[2] - 4))

        model = FakeModel()
        model.action_expert = SimpleNamespace(
            action_encoder=encode, post=head,
            prepare=lambda action_tokens, **kw: (encode(action_tokens), None, None, None, None, None),
        )
        model.mot = SimpleNamespace(
            forward_action_with_video_cache_tensor=lambda **kw: kw["action_tokens"] + Vector((2, -3, 4)),
        )
        inputs = (Vector((1, 2)), 0, None, None, [], [], None)
        full_h, rh = model._denoise_action_c3cache_refresh(*inputs)
        full_v, rv = model._denoise_action_c3cache_refresh(*inputs, residual_space="velocity")
        self.assertEqual(full_h, full_v)
        self.assertEqual((len(rh), len(rv)), (3, 2))
        current = Vector((5, -2))
        self.assertEqual(
            model._denoise_action_c3cache_reuse(current, rh),
            model._denoise_action_c3cache_reuse(current, rv, residual_space="velocity"),
        )

    def test_velocity_matches_hidden_in_affine_scalar_sampler(self):
        for interval in (0, 1, 4, 8):
            for start, end in ((0, 6), (2, 4), (0, 9)):
                with self.subTest(interval=interval, start=start, end=end):
                    hidden_model, velocity_model = FakeModel(), FakeModel()
                    hidden_cache, velocity_cache = C3Cache(), C3Cache()
                    for chunk in range(10):
                        args = dict(signature=("same",), image_value=chunk,
                                    proprio_value=chunk * 2, interval=interval, start=start, end=end)
                        hidden = sample_chunk(hidden_model, hidden_cache, **args)
                        velocity = sample_chunk(velocity_model, velocity_cache,
                                                residual_space="velocity", **args)
                        self.assertAlmostEqual(hidden, velocity)
                    self.assertEqual(hidden_model.mot.calls, velocity_model.mot.calls)
                    self.assertEqual(hidden_cache.reused_steps, velocity_cache.reused_steps)
                    self.assertEqual(velocity_cache.stats()["residual_space"], "velocity")

    def test_residual_space_switch_invalidates_even_with_same_signature(self):
        model, cache = FakeModel(), C3Cache()
        sample_chunk(model, cache, ("same",), 10)
        sample_chunk(model, cache, ("same",), 20)
        self.assertEqual(cache.reused_steps, 7)
        sample_chunk(model, cache, ("same",), 30, residual_space="velocity")
        self.assertEqual((cache.chunk_index, cache.full_steps, cache.reused_steps), (1, 10, 0))
        self.assertEqual(cache.residuals[0], 5 * (3 * 30 + 1))
        sample_chunk(model, cache, ("same",), 40)
        self.assertEqual((cache.chunk_index, cache.full_steps, cache.reused_steps), (1, 10, 0))
        self.assertEqual(cache.stats()["residual_space"], "hidden")
        cache.reset()
        self.assertEqual(cache.residuals, {})

    def test_velocity_tau_one_preserves_full_predictions(self):
        velocity_model, baseline_model, cache = FakeModel(), FakeModel(), C3Cache()
        for image in (10, 20, 30):
            self.assertEqual(
                sample_chunk(velocity_model, cache, ("same",), image,
                             interval=1, residual_space="velocity"),
                full_sample_chunk(baseline_model, image),
            )
        self.assertEqual(cache.reused_steps, 0)

    def test_residual_space_validation_and_wiring(self):
        for mode in ("hidden", "velocity"):
            validate_c3cache_residual_space(mode)
        for mode in ("velocty", "", None, 1, True):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                validate_c3cache_residual_space(mode)
        source = (ROOT / "src/fastwam/models/wan22/fastwam.py").read_text().split("    def infer_action(", 1)[1]
        signature_body = source.split("c3cache_signature = (", 1)[1].split("timestep_video =", 1)[0]
        self.assertIn("c3cache_residual_space,", signature_body)
        tree = ast.parse((ROOT / "src/fastwam/models/wan22/fastwam.py").read_text())
        for call in ast.walk(tree):
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id in (
                "c3cache_refresh", "c3cache_reuse",
            ):
                self.assertIn("residual_space", [kw.arg for kw in call.keywords])

    def test_range_validation(self):
        validate_c3cache_range(10, 0, 9, 0)
        for args in ((10, 0, 10, 4), (10, -1, 6, 4), (10, 7, 6, 4), (10, 0, 6, -1), (10, True, 6, 4)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                validate_c3cache_range(*args)


if __name__ == "__main__":
    unittest.main()
