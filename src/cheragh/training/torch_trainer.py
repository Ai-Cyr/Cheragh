"""Optional, differentiable training for injected PyTorch retrieval encoders.

Importing this module does not import PyTorch. No weights are downloaded, and
tokenization, model/device placement, and optimizer construction belong to the
caller. Candidate sets are local to each example: another query's positive is
never silently introduced as an in-batch negative.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import importlib
import math
import random
from typing import Any

from .data import DistilledRetrievalExample, RetrievalTrainingExample


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and > 0")
    return float(value)


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an integer")
    if value <= 0:
        raise ValueError(f"{name} must be > 0")
    return value


class TorchRetrievalTrainer:
    """Train query/document encoders with per-example contrastive or KL loss.

    Each encoder accepts a sequence of strings and returns a differentiable
    floating tensor of shape ``(len(texts), embedding_dimension)``. Encoders
    may share parameters; the supplied ``torch.optim.Optimizer`` must own the
    trainable parameters. Outputs must have matching device and dtype.

    Ordinary examples minimize ``logsumexp(all scores) - logsumexp(positive
    scores)`` with dot products divided by ``temperature``. Multiple positives
    therefore form one relevant set. At least one explicit negative is needed.
    Distilled examples instead minimize ``T**2 * KL(teacher || student)`` with
    student dot products divided by the example's teacher temperature ``T``;
    the constructor temperature applies only to the contrastive objective.

    Batches combine losses, not candidate pools. This small training loop does
    not implement distributed training, gradient accumulation, mixed precision,
    automatic checkpointing, tokenization, or generator/RAFT fine-tuning.
    ``seed`` controls example shuffling only; callers seed their own models.
    Directly supplied modules (and bound module methods) enter training mode
    during ``fit`` and have their previous modes restored afterward. Wrappers
    hiding a module must manage that module's mode themselves.
    """

    def __init__(
        self,
        query_encoder: Callable[[Sequence[str]], Any],
        document_encoder: Callable[[Sequence[str]], Any],
        optimizer: Any,
        *,
        temperature: float = 1.0,
        normalize_embeddings: bool = False,
        max_grad_norm: float | None = None,
    ) -> None:
        if not callable(query_encoder) or not callable(document_encoder):
            raise TypeError("query_encoder and document_encoder must be callable")
        self.temperature = _positive_number(temperature, "temperature")
        if not isinstance(normalize_embeddings, bool):
            raise TypeError("normalize_embeddings must be a boolean")
        self.normalize_embeddings = normalize_embeddings
        self.max_grad_norm = (
            None if max_grad_norm is None else _positive_number(max_grad_norm, "max_grad_norm")
        )
        try:
            self._torch = importlib.import_module("torch")
        except ImportError as exc:
            raise ImportError(
                "TorchRetrievalTrainer requires PyTorch; install cheragh[training] "
                "or an appropriate PyTorch build for your hardware."
            ) from exc
        if not isinstance(optimizer, self._torch.optim.Optimizer):
            raise TypeError("optimizer must be a torch.optim.Optimizer")
        self.query_encoder = query_encoder
        self.document_encoder = document_encoder
        self.optimizer = optimizer
        self._parameters()  # Fail early for optimizers without trainable parameters.

    def _parameters(self) -> list[Any]:
        parameters: list[Any] = []
        seen: set[int] = set()
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.requires_grad and id(parameter) not in seen:
                    seen.add(id(parameter))
                    parameters.append(parameter)
        if not parameters:
            raise ValueError("optimizer must own at least one trainable parameter")
        return parameters

    def _validate_tensor(self, tensor: Any, rows: int, name: str) -> None:
        torch = self._torch
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise TypeError(f"{name} must be a floating PyTorch tensor")
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must be a dense tensor")
        if tensor.ndim != 2 or tensor.shape[0] != rows or tensor.shape[1] == 0:
            raise ValueError(f"{name} must have shape ({rows}, nonzero embedding_dimension)")
        if not bool(torch.isfinite(tensor).all()):
            raise ValueError(f"{name} must contain only finite values")

    def _batch_loss(self, examples: Sequence[RetrievalTrainingExample | DistilledRetrievalExample]) -> Any:
        torch = self._torch
        raw = [item.example if isinstance(item, DistilledRetrievalExample) else item for item in examples]
        documents = [doc for item in raw for doc in (*item.positive_documents, *item.negative_documents)]
        queries = self.query_encoder([item.query for item in raw])
        candidates = self.document_encoder([doc.content for doc in documents])
        self._validate_tensor(queries, len(raw), "query embeddings")
        self._validate_tensor(candidates, len(documents), "document embeddings")
        if queries.shape[1] != candidates.shape[1]:
            raise ValueError("query and document embedding dimensions must match")
        if queries.device != candidates.device or queries.dtype != candidates.dtype:
            raise ValueError("query and document embeddings must have matching device and dtype")
        # Accumulate half/bfloat16 scores in float32 to avoid avoidable overflow.
        if queries.dtype in (torch.float16, torch.bfloat16):
            queries = queries.float()
            candidates = candidates.float()
        if self.normalize_embeddings:
            queries = torch.nn.functional.normalize(queries, p=2, dim=-1)
            candidates = torch.nn.functional.normalize(candidates, p=2, dim=-1)
        losses = []
        offset = 0
        for index, item in enumerate(examples):
            source = raw[index]
            count = len(source.positive_documents) + len(source.negative_documents)
            scores = candidates[offset : offset + count] @ queries[index]
            offset += count
            temperature = item.temperature if isinstance(item, DistilledRetrievalExample) else self.temperature
            logits = scores / temperature
            if not bool(torch.isfinite(logits).all()):
                raise ValueError("training logits must contain only finite values")
            if isinstance(item, DistilledRetrievalExample):
                targets = logits.new_tensor(item.document_probabilities)
                loss = torch.nn.functional.kl_div(
                    torch.nn.functional.log_softmax(logits, dim=0), targets, reduction="sum"
                ) * (temperature ** 2)
            else:
                loss = torch.logsumexp(logits, dim=0) - torch.logsumexp(
                    logits[: len(source.positive_documents)], dim=0
                )
            losses.append(loss)
        return torch.stack(losses).mean()

    def fit(
        self,
        examples: Sequence[RetrievalTrainingExample | DistilledRetrievalExample],
        *,
        epochs: int = 1,
        batch_size: int = 8,
        shuffle: bool = True,
        seed: int = 0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Update injected weights and return example-weighted training losses.

        ``epoch_losses`` and ``final_loss`` describe losses measured immediately
        before each update, not a held-out or post-training evaluation. Each
        successful batch makes one optimizer step. Invalid embeddings, losses,
        or gradients raise before that batch's step and clear optimizer-owned
        gradients; earlier successful batches remain applied.
        """
        if kwargs:
            raise TypeError(f"Unsupported training options: {', '.join(sorted(kwargs))}")
        epochs = _positive_integer(epochs, "epochs")
        batch_size = _positive_integer(batch_size, "batch_size")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer")
        prepared: list[RetrievalTrainingExample | DistilledRetrievalExample] = []
        for item in examples:
            if not isinstance(item, (RetrievalTrainingExample, DistilledRetrievalExample)):
                raise TypeError("examples must contain retrieval training or distilled examples")
            source = item.example if isinstance(item, DistilledRetrievalExample) else item
            snapshot = RetrievalTrainingExample(
                source.query, source.positive_documents, source.negative_documents, source.answer, source.metadata
            )
            if isinstance(item, DistilledRetrievalExample):
                if len(snapshot.positive_documents) + len(snapshot.negative_documents) < 2:
                    raise ValueError("distillation requires at least two candidate documents")
                prepared.append(DistilledRetrievalExample(
                    snapshot, item.document_probabilities, item.teacher_scores, item.temperature
                ))
            else:
                if not snapshot.negative_documents:
                    raise ValueError("contrastive training requires at least one explicit negative per query")
                prepared.append(snapshot)
        if not prepared:
            raise ValueError("Retrieval training requires at least one example")

        torch = self._torch
        parameters = self._parameters()
        modules: dict[int, tuple[Any, bool]] = {}
        roots = []
        for encoder in (self.query_encoder, self.document_encoder):
            module = encoder if isinstance(encoder, torch.nn.Module) else getattr(encoder, "__self__", None)
            if isinstance(module, torch.nn.Module):
                roots.append(module)
                for child in module.modules():
                    modules.setdefault(id(child), (child, child.training))

        rng = random.Random(seed)
        epoch_losses = []
        steps = 0
        try:
            for module in roots:
                module.train()
            for _ in range(epochs):
                indices = list(range(len(prepared)))
                if shuffle:
                    rng.shuffle(indices)
                loss_sum = 0.0
                for start in range(0, len(indices), batch_size):
                    batch = [prepared[index] for index in indices[start : start + batch_size]]
                    self.optimizer.zero_grad(set_to_none=True)
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in parameters):
                        raise ValueError("optimizer parameters must contain only finite values")
                    loss = self._batch_loss(batch)
                    if not bool(torch.isfinite(loss)):
                        raise ValueError("training loss must be finite")
                    if not loss.requires_grad:
                        raise ValueError("encoder outputs must preserve a differentiable training graph")
                    loss.backward()
                    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
                    if not gradients:
                        raise ValueError("optimizer parameters are disconnected from the encoder training graph")
                    for gradient in gradients:
                        values = gradient.coalesce().values() if gradient.is_sparse else gradient
                        if not bool(torch.isfinite(values).all()):
                            raise ValueError("training gradients must contain only finite values")
                    if self.max_grad_norm is not None:
                        if any(gradient.is_sparse for gradient in gradients):
                            raise ValueError("max_grad_norm requires dense gradients")
                        torch.nn.utils.clip_grad_norm_(
                            parameters, self.max_grad_norm, error_if_nonfinite=True
                        )
                    value = float(loss.detach())
                    self.optimizer.step()
                    steps += 1
                    loss_sum += value * len(batch)
                epoch_losses.append(loss_sum / len(prepared))
        except BaseException:
            self.optimizer.zero_grad(set_to_none=True)
            raise
        finally:
            for module, training in modules.values():
                module.training = training

        kinds = {isinstance(item, DistilledRetrievalExample) for item in prepared}
        objective = "mixed" if len(kinds) == 2 else ("distillation_kl" if True in kinds else "contrastive")
        return {
            "examples": len(prepared),
            "epochs": epochs,
            "steps": steps,
            "epoch_losses": epoch_losses,
            "final_loss": epoch_losses[-1],
            "objective": objective,
        }


__all__ = ["TorchRetrievalTrainer"]
