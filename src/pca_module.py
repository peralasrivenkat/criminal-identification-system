from __future__ import annotations

from dataclasses import dataclass

import joblib
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from config import PCA_MODEL, RANDOM_STATE, SCALER_MODEL

PCA_VARIANCE_FLOOR = 0.96
PCA_VARIANCE_TARGET = 0.985


@dataclass(frozen=True)
class PCAArtifacts:
    pca: PCA
    original_dim: int
    retained_variance: float
    variance_target: float = PCA_VARIANCE_TARGET

    @property
    def n_components_(self) -> int:
        return int(self.pca.n_components_)

    @property
    def explained_variance_ratio_(self) -> np.ndarray:
        return self.pca.explained_variance_ratio_

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.pca.transform(X)


def train_feature_scaler(
    X: np.ndarray,
    persist: bool = False,
) -> tuple[np.ndarray, StandardScaler]:
    if X.ndim != 2:
        raise ValueError("Input features must be a 2D matrix.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    if persist:
        joblib.dump(scaler, SCALER_MODEL)
    return X_scaled, scaler


def apply_feature_scaler(scaler: StandardScaler, X: np.ndarray) -> np.ndarray:
    return scaler.transform(X)


def train_pca(
    X_scaled: np.ndarray,
    n_components: int,
    y: np.ndarray | None = None,
    persist: bool = False,
) -> tuple[np.ndarray, PCAArtifacts]:
    if X_scaled.ndim != 2:
        raise ValueError("Input features must be a 2D matrix.")

    max_components = max(1, min(X_scaled.shape[0], X_scaled.shape[1], n_components))
    selected_components = _resolve_component_count(X_scaled, y, max_components)

    pca = PCA(
        n_components=selected_components,
        whiten=True,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    X_pca = pca.fit_transform(X_scaled)
    artifacts = PCAArtifacts(
        pca=pca,
        original_dim=int(X_scaled.shape[1]),
        retained_variance=float(np.sum(pca.explained_variance_ratio_)),
    )
    if persist:
        joblib.dump(artifacts, PCA_MODEL)
    return X_pca, artifacts


def apply_pca(pca: PCAArtifacts | PCA, X: np.ndarray) -> np.ndarray:
    if isinstance(pca, PCAArtifacts):
        return pca.transform(X)
    return pca.transform(X)


def _resolve_component_count(
    X_scaled: np.ndarray,
    y: np.ndarray | None,
    max_components: int,
) -> int:
    probe = PCA(
        n_components=max_components,
        svd_solver="full",
        random_state=RANDOM_STATE,
    )
    projected = probe.fit_transform(X_scaled)
    cumulative = np.cumsum(probe.explained_variance_ratio_)
    if cumulative.size == 0:
        return 1

    floor_components = _components_for_variance(cumulative, PCA_VARIANCE_FLOOR)
    target_components = _components_for_variance(cumulative, PCA_VARIANCE_TARGET)
    lower_bound = max(1, min(max_components, floor_components))
    upper_bound = max(lower_bound, min(max_components, target_components))

    if y is None or len(np.unique(y)) < 2 or lower_bound == upper_bound:
        return upper_bound

    candidate_counts = _candidate_component_counts(lower_bound, upper_bound)
    scores: dict[int, float] = {}
    for count in candidate_counts:
        projection = projected[:, :count]
        retained_variance = float(cumulative[count - 1])
        component_fraction = float(count) / float(max_components)
        scores[count] = _component_quality_score(
            projection,
            y,
            retained_variance=retained_variance,
            component_fraction=component_fraction,
        )

    best_score = max(scores.values())
    tolerance = 0.01 * max(1.0, abs(best_score))
    valid_counts = [count for count, score in scores.items() if score >= best_score - tolerance]
    return min(valid_counts)


def _components_for_variance(cumulative: np.ndarray, variance_target: float) -> int:
    target_index = int(np.searchsorted(cumulative, variance_target, side="left"))
    return max(1, min(len(cumulative), target_index + 1))


def _candidate_component_counts(lower_bound: int, upper_bound: int) -> list[int]:
    if lower_bound >= upper_bound:
        return [lower_bound]

    if upper_bound - lower_bound <= 8:
        return list(range(lower_bound, upper_bound + 1))

    step = max(1, int(round((upper_bound - lower_bound) / 6.0)))
    counts = {lower_bound, upper_bound}
    for count in range(lower_bound, upper_bound + 1, step):
        counts.add(count)
    return sorted(counts)


def _component_quality_score(
    projection: np.ndarray,
    y: np.ndarray,
    *,
    retained_variance: float,
    component_fraction: float,
) -> float:
    unique_labels = np.unique(y)
    if len(unique_labels) < 2:
        return retained_variance

    centroids = []
    within_class_spread = 0.0
    class_count = 0
    for label in unique_labels:
        class_samples = projection[y == label]
        if class_samples.size == 0:
            continue
        centroid = class_samples.mean(axis=0)
        centroids.append(centroid)
        within_class_spread += float(np.linalg.norm(class_samples - centroid, axis=1).mean())
        class_count += 1

    if len(centroids) < 2 or class_count == 0:
        return retained_variance

    between_class_distance = 0.0
    comparisons = 0
    for index, centroid_a in enumerate(centroids):
        for centroid_b in centroids[index + 1 :]:
            between_class_distance += float(np.linalg.norm(centroid_a - centroid_b))
            comparisons += 1

    separation = between_class_distance / max(comparisons, 1)
    compactness = within_class_spread / float(class_count)
    return (
        (separation / max(compactness, 1e-6))
        + (retained_variance * 0.2)
        - (component_fraction * 0.05)
    )
