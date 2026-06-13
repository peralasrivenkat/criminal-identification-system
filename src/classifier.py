from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Union
import warnings

import joblib
import numpy as np
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import LeaveOneOut, StratifiedKFold
from sklearn.neural_network import MLPClassifier

from config import MLP_HIDDEN_LAYER_SIZES, MLP_MAX_ITERATIONS, MLP_MODEL, RANDOM_STATE


@dataclass
class SingleClassClassifier:
    label: int
    training_config_: dict
    cv_balanced_accuracy_: float = 1.0

    def __post_init__(self) -> None:
        self.classes_ = np.array([int(self.label)], dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full(X.shape[0], int(self.label), dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((X.shape[0], 1), dtype="float32")


ClassifierModel = Union[MLPClassifier, SingleClassClassifier]


def train_classifier(X: np.ndarray, y: np.ndarray) -> ClassifierModel:
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        model = SingleClassClassifier(
            label=int(unique_labels[0]),
            training_config_={
                "mode": "single_class_proxy",
                "reason": "Only one criminal class is available for training.",
            },
        )
        joblib.dump(model, MLP_MODEL)
        return model

    config, cv_score = select_classifier_config(X, y)
    model = _build_classifier(config)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", ConvergenceWarning)
        model.fit(X, y)
    model.training_config_ = asdict(config)
    model.cv_balanced_accuracy_ = float(cv_score)
    joblib.dump(model, MLP_MODEL)
    return model


def predict(model: ClassifierModel, X: np.ndarray) -> np.ndarray:
    return model.predict(X)


def predict_proba(model: ClassifierModel, X: np.ndarray) -> np.ndarray:
    if not hasattr(model, "predict_proba"):
        raise AttributeError("The configured classifier does not expose predict_proba.")
    return model.predict_proba(X)


def load_model(path=MLP_MODEL):
    return joblib.load(path)


@dataclass(frozen=True)
class ClassifierConfig:
    hidden_layer_sizes: tuple[int, ...]
    alpha: float
    solver: str
    activation: str
    max_iter: int
    learning_rate_init: float
    random_state: int
    early_stopping: bool


def _resolve_hidden_layers(sample_count: int, class_count: int) -> tuple[int, ...]:
    if sample_count <= 48:
        return (max(16, class_count * 6),)
    if sample_count <= 120:
        return (max(24, class_count * 8), max(12, class_count * 4))
    return MLP_HIDDEN_LAYER_SIZES


def select_classifier_config(X: np.ndarray, y: np.ndarray) -> tuple[ClassifierConfig, float]:
    candidate_configs = _candidate_classifier_configs(X.shape[0], len(np.unique(y)))
    if X.shape[0] < 8 or len(np.unique(y)) < 2:
        return candidate_configs[0], 0.0

    splitter = _build_splitter(y)
    best_config = candidate_configs[0]
    best_score = float("-inf")
    for config in candidate_configs:
        scores: list[float] = []
        for train_index, valid_index in splitter.split(X, y):
            model = _build_classifier(config)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                model.fit(X[train_index], y[train_index])
            predictions = model.predict(X[valid_index])
            scores.append(float(balanced_accuracy_score(y[valid_index], predictions)))
        score = float(np.mean(scores)) if scores else float("-inf")
        if scores:
            score -= float(np.std(scores)) * 0.05
        if score > best_score:
            best_score = score
            best_config = config
    return best_config, best_score


def _candidate_classifier_configs(sample_count: int, class_count: int) -> list[ClassifierConfig]:
    base_hidden = _resolve_hidden_layers(sample_count, class_count)
    first_layer = base_hidden[0]
    use_early_stopping = sample_count >= 60
    seeds = [RANDOM_STATE, RANDOM_STATE + 11]
    candidates: list[ClassifierConfig] = []
    base_configs = [
        dict(
            hidden_layer_sizes=(max(16, first_layer // 2),),
            alpha=0.02,
            solver="lbfgs",
            activation="tanh",
            max_iter=max(MLP_MAX_ITERATIONS, 1500),
            learning_rate_init=0.001,
            early_stopping=False,
        ),
        dict(
            hidden_layer_sizes=base_hidden,
            alpha=0.05,
            solver="lbfgs",
            activation="tanh",
            max_iter=max(MLP_MAX_ITERATIONS, 1800),
            learning_rate_init=0.001,
            early_stopping=False,
        ),
        dict(
            hidden_layer_sizes=base_hidden,
            alpha=0.2,
            solver="lbfgs",
            activation="relu",
            max_iter=max(MLP_MAX_ITERATIONS, 1800),
            learning_rate_init=0.001,
            early_stopping=False,
        ),
    ]
    if use_early_stopping:
        base_configs.append(
            dict(
                hidden_layer_sizes=base_hidden,
                alpha=1e-3,
                solver="adam",
                activation="relu",
                max_iter=max(MLP_MAX_ITERATIONS, 1200),
                learning_rate_init=0.0008,
                early_stopping=True,
            )
        )

    for seed in seeds:
        for params in base_configs:
            candidates.append(
                ClassifierConfig(
                    hidden_layer_sizes=params["hidden_layer_sizes"],
                    alpha=float(params["alpha"]),
                    solver=str(params["solver"]),
                    activation=str(params["activation"]),
                    max_iter=int(params["max_iter"]),
                    learning_rate_init=float(params["learning_rate_init"]),
                    random_state=int(seed),
                    early_stopping=bool(params["early_stopping"]),
                )
            )
    return candidates


def _build_classifier(config: ClassifierConfig) -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=config.hidden_layer_sizes,
        activation=config.activation,
        solver=config.solver,
        alpha=config.alpha,
        max_iter=config.max_iter,
        random_state=config.random_state,
        learning_rate_init=config.learning_rate_init,
        early_stopping=config.early_stopping,
        validation_fraction=0.2 if config.early_stopping else 0.1,
        n_iter_no_change=20,
    )


def _build_splitter(y: np.ndarray):
    _, class_counts = np.unique(y, return_counts=True)
    minimum_class_count = int(np.min(class_counts))
    if minimum_class_count >= 3:
        n_splits = min(5, minimum_class_count)
        return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    return LeaveOneOut()
