"""Execute the real FastWAM cache sampler with scalar fake tensors; no torch import."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace
import unittest


ROOT = Path(__file__).resolve().parents[1]
CACHE = runpy.run_path(str(ROOT / "src/fastwam/models/wan22/c3cache.py"))
C3Cache = CACHE["C3Cache"]
validate_c3cache_method = CACHE["validate_c3cache_method"]


def load_sampler_code():
    path = ROOT / "src/fastwam/models/wan22/fastwam.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FastWAM")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "infer_action")
    start = next(i for i, node in enumerate(method.body)
                 if isinstance(node, ast.If) and "self._c3cache.begin(" in ast.unparse(node))
    body = method.body[start:-1]
    return compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec")


SAMPLER = load_sampler_code()


def load_partial_mot():
    path = ROOT / "src/fastwam/models/wan22/mot.py"
    tree = ast.parse(path.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MoT")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
                  and node.name == "forward_action_with_video_cache_tensor")
    module = ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[]))
    namespace = {"torch": SimpleNamespace(Tensor=float, cat=lambda values, dim: values),
                 "flash_attention": lambda **_kwargs: 0}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace[method.name]


PARTIAL_MOT = load_partial_mot()


class Scalar:
    def __init__(self, value):
        self.value = float(value)
        self.dtype = "float32"
        self.device = "cpu"

    def __add__(self, other):
        return Scalar(self.value + val(other))

    __radd__ = __add__

    def __sub__(self, other):
        return Scalar(self.value - val(other))

    def __mul__(self, other):
        return Scalar(self.value * val(other))

    __rmul__ = __mul__

    def unsqueeze(self, _dim):
        return self

    def to(self, **_kwargs):
        return self

    def detach(self):
        return self

    def clone(self):
        return Scalar(self.value)

    def __getitem__(self, _key):
        return self


def val(value):
    return value.value if isinstance(value, Scalar) else value


class FakeScheduler:
    def __init__(self):
        self.steps = 0
        self.fail_at = None
        self.in_place = False

    def step(self, prediction, delta, sample):
        self.steps += 1
        if self.steps == self.fail_at:
            raise RuntimeError("scheduler failure")
        if self.in_place:
            sample.value += val(prediction) * val(delta)
            return sample
        return sample + prediction * delta


class FakeModel:
    def __init__(self):
        self._c3cache = C3Cache()
        self.infer_action_scheduler = FakeScheduler()
        self.device = "cpu"
        self.full_calls = 0
        self.probe_calls = 0
        self.virtual_calls = 0
        self.graph_prediction = Scalar(0)
        self.graph_residual = Scalar(0)

    def reset_c3cache(self):
        self._c3cache.reset()

    def _denoise_action_with_video_cache(self, latents_action, timestep_action,
                                         context, context_mask, video_cache_k,
                                         video_cache_v, action_attention_mask):
        self.full_calls += 1
        self.graph_prediction.value = 5 * (
            2 * val(latents_action) + 1 + 3 * val(video_cache_k[0])
            + 2 * val(timestep_action) + val(context) + 1
        )
        return self.graph_prediction

    def _denoise_action_c3cache_refresh(self, **kwargs):
        prediction = self._denoise_action_with_video_cache(**kwargs)
        self.graph_residual.value = (val(prediction) / 5) - (2 * val(kwargs["latents_action"]) + 1)
        return prediction, self.graph_residual

    def _denoise_action_c3cache_reuse(self, latents_action, residual):
        return Scalar(5 * (2 * val(latents_action) + 1 + val(residual)))

    def _denoise_action_c3cache_virtual(self, latents_action):
        self.virtual_calls += 1
        return Scalar(5 * (2 * val(latents_action) + 1))

    def _denoise_action_c3cache_probe(self, **kwargs):
        self.probe_calls += 1
        return Scalar(5 * (2 * val(kwargs["latents_action"]) + 1)
                      + kwargs["depth_limit"] * (val(kwargs["context"]) + val(kwargs["video_cache_k"][0])))


def sample(model, method, *, signature=("task",), image=1, context=1, seed=10,
           interval=4, end=6, start=None, depth=1, n=10):
    start = (1 if method == "velocity_delta" else 0) if start is None else start
    validate_c3cache_method(method, start, depth, 4)
    scope = {
        "self": model, "torch": SimpleNamespace(float32="float32", Tensor=Scalar),
        "WanContinuousFlowMatchScheduler": FakeScheduler,
        "c3cache_enabled": True, "c3cache_signature": (*signature, method, start, end, interval, depth,
                                                     seed if method in {"velocity", "prefix"} else None),
        "c3cache_method": method, "c3cache_start_step": start, "c3cache_end_step": end,
        "c3cache_refresh_interval": interval, "c3cache_probe_depth": depth, "seed": seed,
        "compile_action_infer": False, "latents_action": Scalar(1),
        "infer_timesteps_action": [Scalar(i) for i in range(n)],
        "infer_deltas_action": [Scalar(-.001) for _ in range(n)],
        "context": Scalar(context), "context_mask": None,
        "video_cache_k": [Scalar(image)], "video_cache_v": [], "action_attention_mask": None,
        "c3cache_refresh": model._denoise_action_c3cache_refresh,
        "c3cache_reuse": model._denoise_action_c3cache_reuse,
        "c3cache_virtual": model._denoise_action_c3cache_virtual,
        "c3cache_probe": model._denoise_action_c3cache_probe,
        "denoise_action_with_video_cache": model._denoise_action_with_video_cache,
    }
    exec(SAMPLER, scope)
    return val(scope["action_out"])


class VelocityCacheChecks(unittest.TestCase):
    def test_modes_and_step_counts(self):
        expected = {
            "hidden": (3, 7, 0, 0),
            "velocity_delta": (4, 6, 0, 0),
            "velocity": (3, 7, 0, 0),
            "prefix": (3, 7, 7, 0),
            "velocity_virtual": (3, 7, 0, 0),
            "velocity_probe": (3, 7, 0, 2),
        }
        for method, counts in expected.items():
            with self.subTest(method=method):
                model = FakeModel()
                sample(model, method, depth=2)
                self.assertEqual(model._c3cache.stats()["last_chunk_full_steps"], 10)
                sample(model, method, image=8, context=6, depth=2)
                stats = model._c3cache.stats()
                self.assertEqual((stats["last_chunk_full_steps"], stats["last_chunk_reused_steps"],
                                  stats["last_chunk_scheduler_skipped_steps"], stats["last_chunk_probe_blocks"]), counts)
                self.assertEqual(stats["refresh_chunks"], 1)
                self.assertEqual(stats["reuse_chunks"], 1)
                self.assertEqual(stats["cached_from_chunk"], 0)
                self.assertEqual(model.infer_action_scheduler.steps, 20 - counts[2])

    def test_fixed_noise_prefix_replay_matches_hidden(self):
        results = {}
        for method in ("hidden", "velocity", "prefix", "velocity_virtual"):
            model = FakeModel()
            sample(model, method, image=1, context=1)
            results[method] = sample(model, method, image=9, context=4)
        for method in ("velocity", "prefix", "velocity_virtual"):
            self.assertAlmostEqual(results[method], results["hidden"], places=10)

    def test_anchor_recurrences_under_changed_condition(self):
        for method, end, depth in (("velocity_delta", 6, 1),
                                   ("velocity_delta", 7, 1),
                                   ("velocity_probe", 6, 2)):
            with self.subTest(method=method, end=end):
                model = FakeModel()
                sample(model, method, image=1, context=1, end=end, depth=depth)
                offsets = {k: val(v) for k, v in model._c3cache.residuals.items()}
                current = 1.0
                image, context = 8, 6
                anchor = None
                if method == "velocity_probe":
                    anchor = 5 * (2 * current + 1) + depth * (image + context)
                for step in range(10):
                    full = 5 * (2 * current + 1 + 3 * image + 2 * step + context + 1)
                    if method == "velocity_delta" and step == 0:
                        anchor = full
                    prediction = (anchor + offsets[step]) if step in offsets else full
                    current -= .001 * prediction
                self.assertAlmostEqual(sample(model, method, image=image, context=context,
                                              end=end, depth=depth), current)
                if method == "velocity_delta" and end == 7:
                    self.assertEqual(model._c3cache.stats()["last_chunk_full_steps"], 3)

    def test_tau_one_matches_full_baseline_for_every_method(self):
        for method in ("hidden", "velocity_delta", "velocity", "prefix",
                       "velocity_virtual", "velocity_probe"):
            with self.subTest(method=method):
                model = FakeModel()
                baseline = FakeModel()
                for chunk in range(3):
                    result = sample(model, method, interval=1, image=chunk + 2, context=chunk + 5)
                    expected = sample(baseline, "hidden", interval=1,
                                      image=chunk + 2, context=chunk + 5)
                    self.assertAlmostEqual(result, expected)
                self.assertEqual((model._c3cache.full_steps, model._c3cache.reused_steps), (30, 0))

    def test_missing_slot_refreshes_whole_trajectory(self):
        for method in ("hidden", "velocity_delta", "velocity", "prefix",
                       "velocity_virtual", "velocity_probe"):
            with self.subTest(method=method):
                model = FakeModel()
                sample(model, method, interval=0)
                if method == "prefix":
                    model._c3cache.endpoint = None
                else:
                    del model._c3cache.residuals[3]
                sample(model, method, interval=0)
                self.assertEqual(model._c3cache.stats()["last_chunk_reason"], "cache_miss")
                self.assertEqual((model._c3cache.full_steps, model._c3cache.reused_steps), (20, 0))
                self.assertEqual(model._c3cache.stats()["cached_from_chunk"], 1)

    def test_prefix_endpoint_survives_in_place_scheduler(self):
        model = FakeModel()
        model.infer_action_scheduler.in_place = True
        sample(model, "prefix", interval=0)
        endpoint = val(model._c3cache.endpoint)
        sample(model, "prefix", interval=0, image=7, context=4)
        self.assertEqual(val(model._c3cache.endpoint), endpoint)

    def test_direct_and_prefix_need_identical_seed(self):
        for method in ("velocity", "prefix"):
            with self.subTest(method=method):
                model = FakeModel()
                sample(model, method, seed=None)
                sample(model, method, seed=None)
                self.assertEqual(model._c3cache.stats()["last_chunk_reason"], "unfixed_noise")
                self.assertEqual(model.full_calls, 20)
                sample(model, method, seed=3)
                sample(model, method, seed=4)
                self.assertEqual(model._c3cache.stats()["last_chunk_reason"], "first_chunk")
                self.assertEqual(model._c3cache.stats()["completed_chunks"], 1)

    def test_refresh_intervals_signatures_and_failure(self):
        for interval in (0, 1, 4):
            model = FakeModel()
            for chunk in range(5):
                sample(model, "velocity_delta", interval=interval, image=chunk)
            expected = {0: (1, 4), 1: (5, 0), 4: (2, 3)}[interval]
            self.assertEqual((model._c3cache.refresh_chunks, model._c3cache.reuse_chunks), expected)
            sample(model, "velocity_delta", interval=interval, signature=("new model",))
            self.assertEqual((model._c3cache.chunk_index, model._c3cache.full_steps), (1, 10))
        model = FakeModel()
        sample(model, "velocity", interval=0)
        model.infer_action_scheduler.fail_at = model.infer_action_scheduler.steps + 1
        with self.assertRaises(RuntimeError):
            sample(model, "velocity", seed=None, interval=0)
        self.assertEqual(model._c3cache.stats()["completed_chunks"], 0)
        model.infer_action_scheduler.fail_at = None
        sample(model, "velocity", interval=0)
        self.assertEqual(model._c3cache.stats()["last_chunk_reason"], "first_chunk")

    def test_real_anchor_offsets_and_compiled_style_ownership(self):
        model = FakeModel()
        sample(model, "velocity_delta", interval=0)
        anchor = val(model._c3cache.anchor)
        offset = val(model._c3cache.residuals[1])
        self.assertNotEqual(offset, 0)
        self.assertAlmostEqual(anchor + offset, 5 * (2 * (1 - .001 * anchor) + 1 + 3 + 2 + 1 + 1))
        saved = [val(item) for item in model._c3cache.residuals.values()]
        model.graph_prediction.value = 99999
        model.graph_residual.value = 99999
        self.assertEqual(saved, [val(item) for item in model._c3cache.residuals.values()])
        for method in ("velocity", "prefix", "velocity_virtual", "velocity_probe"):
            model = FakeModel()
            sample(model, method, interval=0)
            source = model._c3cache.endpoint if method == "prefix" else model._c3cache.residuals[0]
            previous = val(source)
            model.graph_prediction.value = 99999
            self.assertEqual(val(source), previous)

    def test_validation_and_range_boundary(self):
        for method, start in (("velocity_delta", 0), ("velocity_delta", 2), ("prefix", 1),
                              ("velocity", 1), ("velocity_virtual", 1), ("velocity_probe", 1)):
            with self.assertRaises(ValueError):
                validate_c3cache_method(method, start, 1, 4)
        for depth in (0, 5, True):
            with self.assertRaises(ValueError):
                validate_c3cache_method("velocity_probe", 0, depth, 4)
        model = FakeModel()
        sample(model, "prefix", end=9)
        sample(model, "prefix", end=9)
        self.assertEqual((model._c3cache.full_steps, model._c3cache.reused_steps,
                          model._c3cache.scheduler_skipped_steps), (10, 10, 10))

    def test_partial_mot_runs_exact_requested_blocks(self):
        class Mot:
            forward = PARTIAL_MOT

            def __init__(self):
                self.num_layers = 4
                self.num_heads = 1
                self.mixtures = {"action": SimpleNamespace(blocks=list(range(4)))}
                self.visited = []

            def _build_expert_attention_io(self, *, block, x, **_kwargs):
                self.visited.append(block)
                return (Scalar(0), Scalar(0), Scalar(0), x,
                        None, None, None, None, False)

            def _apply_expert_post_block_tensor(self, *, residual_x, **_kwargs):
                return residual_x + 1

        for depth in (1, 2, 4, None):
            mot = Mot()
            result = mot.forward(
                action_tokens=Scalar(0), action_freqs=None, action_t_mod=None,
                action_context=None, action_context_mask=None,
                video_cache_k=[Scalar(0)] * 4, video_cache_v=[Scalar(0)] * 4,
                action_attention_mask=Scalar(0), depth_limit=depth,
            )
            expected = 4 if depth is None else depth
            self.assertEqual(val(result), expected)
            self.assertEqual(mot.visited, list(range(expected)))
        for depth in (0, 5):
            with self.assertRaises(ValueError):
                Mot().forward(Scalar(0), None, None, None, None, [], [], Scalar(0), depth)


if __name__ == "__main__":
    unittest.main()
