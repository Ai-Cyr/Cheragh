from __future__ import annotations

import math
from pathlib import Path
import subprocess
import sys
import unittest
from unittest.mock import patch

from cheragh import Document
from cheragh.training.data import RetrievalTrainingExample, RetrievalTrainingPipeline, TeacherScoreDistiller
from cheragh.training.torch_trainer import TorchRetrievalTrainer

try:
    import torch
except ImportError:
    torch = None


class OptionalTorchImportTests(unittest.TestCase):
    def test_importing_training_does_not_require_or_import_torch(self):
        source = str(Path(__file__).resolve().parents[1] / "src")
        code = (
            "import sys\n"
            f"sys.path.insert(0, {source!r})\n"
            "class RejectTorch:\n"
            "    def find_spec(self, fullname, path=None, target=None):\n"
            "        if fullname == 'torch' or fullname.startswith('torch.'):\n"
            "            raise AssertionError('Unexpected PyTorch import')\n"
            "sys.meta_path.insert(0, RejectTorch())\n"
            "from cheragh.training.torch_trainer import TorchRetrievalTrainer\n"
            "assert 'torch' not in sys.modules\n"
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_missing_dependency_has_an_actionable_install_message(self):
        with patch("cheragh.training.torch_trainer.importlib.import_module", side_effect=ImportError("torch")):
            with self.assertRaisesRegex(ImportError, r"cheragh\[training\]"):
                TorchRetrievalTrainer(lambda texts: None, lambda texts: None, object())


@unittest.skipIf(torch is None, "Optional PyTorch training dependency is not installed")
class TorchRetrievalTrainingTests(unittest.TestCase):
    def _model(self, *, sparse=False):
        class TinyEncoder(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.vocabulary = {"query": 0, "positive": 1, "negative": 2, "second positive": 3}
                self.embedding = torch.nn.Embedding(4, 2, sparse=sparse)
                self.calls = []
                with torch.no_grad():
                    self.embedding.weight.copy_(torch.tensor([[1., 0.], [0., 1.], [1., 0.], [0., -1.]]))

            def forward(self, texts):
                self.calls.append((tuple(texts), self.training))
                return self.embedding(torch.tensor([self.vocabulary[text] for text in texts]))

        return TinyEncoder()

    def _example(self, *, multiple_positives=False):
        positives = [Document("positive", doc_id="p")]
        if multiple_positives:
            positives.append(Document("second positive", doc_id="p2"))
        return RetrievalTrainingExample("query", tuple(positives), (Document("negative", doc_id="n"),))

    def _contrastive_loss(self, model):
        with torch.no_grad():
            query = model(["query"])
            logits = query @ model(["positive", "negative"]).T
            return float(torch.nn.functional.cross_entropy(logits, torch.tensor([0])))

    def test_pipeline_really_updates_weights_and_reduces_contrastive_loss(self):
        model = self._model()
        before = model.embedding.weight.detach().clone()
        initial_loss = self._contrastive_loss(model)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TorchRetrievalTrainer(model, model, optimizer, max_grad_norm=1.0)

        result = RetrievalTrainingPipeline().fit([self._example()], trainer, epochs=25, shuffle=False)

        self.assertFalse(torch.equal(before, model.embedding.weight))
        self.assertLess(self._contrastive_loss(model), initial_loss * 0.3)
        self.assertLess(result["epoch_losses"][-1], result["epoch_losses"][0])
        self.assertEqual(result["steps"], 25)
        self.assertEqual(result["objective"], "contrastive")
        self.assertTrue(all(math.isfinite(value) for value in result["epoch_losses"]))

    def test_multiple_positives_share_numerator_and_candidates_stay_local(self):
        model = self._model()
        trainer = TorchRetrievalTrainer(model, model, torch.optim.SGD(model.parameters(), lr=0.0))
        example = self._example(multiple_positives=True)

        result = trainer.fit([example, example], batch_size=2, shuffle=False)

        self.assertAlmostEqual(result["final_loss"], math.log(2 + math.e) - math.log(2), places=6)
        self.assertEqual(result["steps"], 1)
        self.assertEqual(model.calls[0][0], ("query", "query"))

    def test_kl_distillation_learns_teacher_distribution_with_teacher_temperature(self):
        model = self._model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TorchRetrievalTrainer(model, model, optimizer, temperature=0.01)
        distilled = TeacherScoreDistiller(lambda query, docs: [4.0, 0.0], temperature=2.0).distill(self._example())

        def kl_loss():
            with torch.no_grad():
                logits = (model(["query"]) @ model(["positive", "negative"]).T).squeeze(0) / 2
                return float(torch.nn.functional.kl_div(
                    logits.log_softmax(0), torch.tensor(distilled.document_probabilities), reduction="sum"
                ) * 4)

        initial = kl_loss()
        before = model.embedding.weight.detach().clone()
        result = trainer.fit([distilled], epochs=30, shuffle=False)

        self.assertEqual(result["objective"], "distillation_kl")
        self.assertAlmostEqual(result["epoch_losses"][0], initial, places=6)
        self.assertLess(kl_loss(), initial * 0.1)
        self.assertFalse(torch.equal(before, model.embedding.weight))

    def test_epoch_metrics_weight_partial_batch_by_examples_and_allow_mixed_targets(self):
        model = self._model()
        trainer = TorchRetrievalTrainer(model, model, torch.optim.SGD(model.parameters(), lr=0.0))
        ordinary = self._example()
        teacher = TeacherScoreDistiller(lambda query, docs: [0.0, 1.0]).distill(ordinary)

        result = trainer.fit([ordinary, ordinary, teacher], batch_size=2, shuffle=False)

        self.assertEqual(result["steps"], 2)
        self.assertEqual(result["examples"], 3)
        self.assertEqual(result["objective"], "mixed")
        self.assertAlmostEqual(result["final_loss"], 2 * math.log1p(math.e) / 3, places=6)

    def test_nonfinite_gradients_never_reach_optimizer_and_modes_are_restored(self):
        model = self._model()
        model.eval()
        model.embedding.train()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TorchRetrievalTrainer(model, model, optimizer)
        before = model.embedding.weight.detach().clone()
        handle = model.embedding.weight.register_hook(lambda grad: torch.full_like(grad, float("inf")))
        try:
            with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                with self.assertRaisesRegex(ValueError, "gradients.*finite"):
                    trainer.fit([self._example()])
                step.assert_not_called()
        finally:
            handle.remove()
        self.assertTrue(torch.equal(before, model.embedding.weight))
        self.assertIsNone(model.embedding.weight.grad)
        self.assertFalse(model.training)
        self.assertTrue(model.embedding.training)

    def test_invalid_embeddings_fail_before_update(self):
        cases = [
            (lambda model, texts: model(texts) * float("nan"), ValueError, "finite"),
            (lambda model, texts: model(texts)[0], ValueError, "shape"),
            (lambda model, texts: model(texts).long(), TypeError, "floating"),
        ]
        for transform, exception, message in cases:
            with self.subTest(message=message):
                model = self._model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
                trainer = TorchRetrievalTrainer(lambda texts: transform(model, texts), model, optimizer)
                before = model.embedding.weight.detach().clone()
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(exception, message):
                        trainer.fit([self._example()])
                    step.assert_not_called()
                self.assertTrue(torch.equal(before, model.embedding.weight))

    def test_detached_and_disconnected_training_graphs_are_rejected(self):
        for disconnected in (False, True):
            with self.subTest(disconnected=disconnected):
                model = self._model()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

                def encode(texts):
                    return model(texts).detach().requires_grad_(disconnected)

                trainer = TorchRetrievalTrainer(encode, encode, optimizer)
                with patch.object(optimizer, "step", wraps=optimizer.step) as step:
                    with self.assertRaisesRegex(ValueError, "differentiable|disconnected"):
                        trainer.fit([self._example()])
                    step.assert_not_called()

    def test_optimizer_parameters_must_be_finite_before_forward(self):
        model = self._model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TorchRetrievalTrainer(model, model, optimizer)
        with torch.no_grad():
            model.embedding.weight[0, 0] = float("inf")
        with self.assertRaisesRegex(ValueError, "parameters.*finite"):
            trainer.fit([self._example()])
        self.assertEqual(model.calls, [])

    def test_bound_module_methods_enter_training_mode_and_restore_eval(self):
        model = self._model()
        model.eval()
        trainer = TorchRetrievalTrainer(model.forward, model.forward, torch.optim.SGD(model.parameters(), lr=0.1))

        trainer.fit([self._example()])

        self.assertTrue(all(training for _, training in model.calls))
        self.assertFalse(model.training)
        self.assertFalse(model.embedding.training)

    def test_sparse_gradients_update_without_clipping(self):
        model = self._model(sparse=True)
        initial_loss = self._contrastive_loss(model)
        trainer = TorchRetrievalTrainer(model, model, torch.optim.SGD(model.parameters(), lr=0.1))

        trainer.fit([self._example()], epochs=5)

        self.assertLess(self._contrastive_loss(model), initial_loss)

    def test_normalized_embeddings_and_shuffle_are_reproducible(self):
        results = []
        orders = []
        examples = [self._example(), self._example(multiple_positives=True), self._example()]
        for _ in range(2):
            model = self._model()
            trainer = TorchRetrievalTrainer(
                model, model, torch.optim.SGD(model.parameters(), lr=0.05), normalize_embeddings=True
            )
            results.append(trainer.fit(examples, epochs=2, batch_size=1, shuffle=True, seed=13))
            orders.append(model.calls)
        self.assertEqual(results[0], results[1])
        self.assertEqual(orders[0], orders[1])

    def test_invalid_dataset_is_rejected_before_any_update(self):
        model = self._model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        trainer = TorchRetrievalTrainer(model, model, optimizer)
        no_negatives = RetrievalTrainingExample("query", (Document("positive", doc_id="p"),))
        for examples in ([], [object()], [self._example(), no_negatives]):
            with self.subTest(examples=examples):
                with self.assertRaises((ValueError, TypeError)):
                    trainer.fit(examples, batch_size=1)
        self.assertEqual(model.calls, [])

    def test_invalid_configuration_is_rejected(self):
        model = self._model()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for kwargs in ({"temperature": True}, {"temperature": 0}, {"temperature": float("nan")},
                       {"max_grad_norm": -1}, {"normalize_embeddings": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    TorchRetrievalTrainer(model, model, optimizer, **kwargs)
        with self.assertRaises(TypeError):
            TorchRetrievalTrainer(model, model, object())
        trainer = TorchRetrievalTrainer(model, model, optimizer)
        for kwargs in ({"epochs": True}, {"epochs": 0}, {"batch_size": 0}, {"shuffle": 1},
                       {"seed": True}, {"unknown_option": 1}):
            with self.subTest(kwargs=kwargs):
                with self.assertRaises((TypeError, ValueError)):
                    trainer.fit([self._example()], **kwargs)


if __name__ == "__main__":
    unittest.main()
