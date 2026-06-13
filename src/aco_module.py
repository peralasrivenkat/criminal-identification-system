from __future__ import annotations

import joblib
import numpy as np
from sklearn.feature_selection import f_classif, mutual_info_classif

from config import ACO_MODEL, RANDOM_STATE

ACO_ALPHA = 1.0
ACO_BETA = 1.6
ACO_EVAPORATION = 0.22


def initialize_pheromones(
    n_features: int,
    feature_scores: np.ndarray | None = None,
) -> np.ndarray:
    if feature_scores is None:
        return np.ones(n_features, dtype="float64")
    normalized = np.asarray(feature_scores, dtype="float64").reshape(-1)
    normalized = np.clip(normalized, 1e-6, None)
    normalized /= normalized.mean()
    return normalized


def select_features(
    pheromones: np.ndarray,
    n_select: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    if n_select > len(pheromones):
        raise ValueError("n_select cannot be larger than the number of features.")
    generator = rng or np.random.default_rng(RANDOM_STATE)
    probability = pheromones / pheromones.sum()
    return np.sort(generator.choice(len(pheromones), size=n_select, replace=False, p=probability))


def fitness_function(
    X: np.ndarray,
    y: np.ndarray | None = None,
    feature_scores: np.ndarray | None = None,
    redundancy_penalty: float = 0.0,
) -> float:
    variance_score = float(np.mean(np.var(X, axis=0)))
    relevance_score = _feature_score_bonus(feature_scores)
    if y is None or len(np.unique(y)) < 2:
        return variance_score + relevance_score - redundancy_penalty

    class_means = [X[y == label].mean(axis=0) for label in np.unique(y)]
    between_class_distance = 0.0
    comparisons = 0
    for index, mean_a in enumerate(class_means):
        for mean_b in class_means[index + 1 :]:
            between_class_distance += float(np.linalg.norm(mean_a - mean_b))
            comparisons += 1
    separation_score = between_class_distance / max(comparisons, 1)
    compactness_penalty = float(
        np.mean(
            [
                np.linalg.norm(X[y == label] - X[y == label].mean(axis=0), axis=1).mean()
                for label in np.unique(y)
            ]
        )
    )
    return (
        (relevance_score * 0.45)
        + (separation_score * 0.4)
        + (variance_score * 0.15)
        - compactness_penalty
        - redundancy_penalty
    )


def update_pheromones(
    pheromones: np.ndarray,
    ant_results: list[tuple[np.ndarray, float]],
    feature_scores: np.ndarray,
    evaporation: float = ACO_EVAPORATION,
) -> np.ndarray:
    updated = (1.0 - evaporation) * pheromones
    if not ant_results:
        return np.clip(updated, 1e-6, None)

    ordered_results = sorted(ant_results, key=lambda item: item[1], reverse=True)
    elite_count = max(1, len(ordered_results) // 3)
    score_scale = max(float(ordered_results[0][1]), 1e-6)
    normalized_scores = np.clip(feature_scores.astype("float64"), 1e-6, None)
    normalized_scores /= normalized_scores.mean()
    for rank, (selected, fitness) in enumerate(ordered_results[:elite_count], start=1):
        deposit = max(fitness, 1e-6) / (score_scale * rank)
        updated[selected] += deposit * normalized_scores[selected]
    return np.clip(updated, 1e-6, None)


def run_aco(
    X: np.ndarray,
    y: np.ndarray | None = None,
    n_ants: int = 8,
    n_iter: int = 15,
    n_select: int = 32,
    random_state: int = RANDOM_STATE,
    persist: bool = False,
) -> np.ndarray:
    n_features = X.shape[1]
    selection_size = max(1, min(n_select, n_features))
    feature_scores = compute_feature_scores(X, y)
    redundancy = compute_redundancy_matrix(X)
    pheromones = initialize_pheromones(n_features, feature_scores=feature_scores)
    rng = np.random.default_rng(random_state)

    ranked_indices = np.argsort(feature_scores)[::-1]
    best_features = np.sort(ranked_indices[:selection_size])
    best_score = _score_selected_features(X, y, feature_scores, redundancy, best_features)

    for _ in range(n_iter):
        ant_results: list[tuple[np.ndarray, float]] = []
        for _ in range(n_ants):
            selected = _construct_solution(
                pheromones,
                feature_scores,
                redundancy,
                selection_size,
                rng,
            )
            score = _score_selected_features(X, y, feature_scores, redundancy, selected)
            ant_results.append((selected, score))
            if score > best_score:
                best_score = score
                best_features = selected

        ant_results.append((best_features, best_score))
        pheromones = update_pheromones(pheromones, ant_results, feature_scores)

    if persist:
        joblib.dump(best_features, ACO_MODEL)
    return best_features


def compute_feature_scores(X: np.ndarray, y: np.ndarray | None = None) -> np.ndarray:
    variance = np.var(X, axis=0).astype("float64")
    variance = variance / max(float(np.mean(variance)), 1e-6)
    if y is None or len(np.unique(y)) < 2:
        return variance + 1e-6

    f_scores, _ = f_classif(X, y)
    f_scores = np.nan_to_num(f_scores, nan=0.0, posinf=0.0, neginf=0.0)
    mutual_information = mutual_info_classif(X, y, discrete_features=False, random_state=RANDOM_STATE)
    mutual_information = np.nan_to_num(mutual_information, nan=0.0, posinf=0.0, neginf=0.0)

    overall_mean = X.mean(axis=0)
    between = np.zeros(X.shape[1], dtype="float64")
    within = np.zeros(X.shape[1], dtype="float64")
    for label in np.unique(y):
        class_samples = X[y == label]
        class_mean = class_samples.mean(axis=0)
        between += class_samples.shape[0] * np.square(class_mean - overall_mean)
        within += np.square(class_samples - class_mean).sum(axis=0)
    fisher_like = (between + 1e-6) / (within + 1e-6)
    combined = variance + f_scores + mutual_information + fisher_like
    return np.nan_to_num(combined, nan=1e-6, posinf=1e-6, neginf=1e-6) + 1e-6


def compute_redundancy_matrix(X: np.ndarray) -> np.ndarray:
    if X.shape[1] <= 1:
        return np.zeros((X.shape[1], X.shape[1]), dtype="float64")
    correlation = np.corrcoef(X, rowvar=False)
    correlation = np.nan_to_num(correlation, nan=0.0, posinf=0.0, neginf=0.0)
    np.fill_diagonal(correlation, 0.0)
    return np.abs(correlation).astype("float64")


def _feature_score_bonus(feature_scores: np.ndarray | None) -> float:
    if feature_scores is None or len(feature_scores) == 0:
        return 0.0
    return float(np.mean(feature_scores))


def _construct_solution(
    pheromones: np.ndarray,
    feature_scores: np.ndarray,
    redundancy: np.ndarray,
    selection_size: int,
    rng: np.random.Generator,
) -> np.ndarray:
    available = np.arange(len(pheromones))
    selected: list[int] = []
    while len(selected) < selection_size and len(available) > 0:
        heuristic = np.clip(feature_scores[available], 1e-6, None)
        if selected:
            redundancy_penalty = 1.0 + redundancy[np.ix_(available, np.asarray(selected, dtype=int))].mean(axis=1)
            heuristic = heuristic / redundancy_penalty
        desirability = np.power(pheromones[available], ACO_ALPHA) * np.power(heuristic, ACO_BETA)
        desirability_sum = float(desirability.sum())
        if desirability_sum <= 0.0:
            probabilities = np.full(len(available), 1.0 / len(available), dtype="float64")
        else:
            probabilities = desirability / desirability_sum
        chosen = int(rng.choice(available, p=probabilities))
        selected.append(chosen)
        available = available[available != chosen]
    return np.sort(np.asarray(selected, dtype=int))


def _score_selected_features(
    X: np.ndarray,
    y: np.ndarray | None,
    feature_scores: np.ndarray,
    redundancy: np.ndarray,
    selected: np.ndarray,
) -> float:
    selected_scores = feature_scores[selected]
    redundancy_penalty = _redundancy_penalty(redundancy, selected)
    return fitness_function(
        X[:, selected],
        y,
        feature_scores=selected_scores,
        redundancy_penalty=redundancy_penalty,
    )


def _redundancy_penalty(redundancy: np.ndarray, selected: np.ndarray) -> float:
    if len(selected) <= 1:
        return 0.0
    matrix = redundancy[np.ix_(selected, selected)]
    upper_indices = np.triu_indices_from(matrix, k=1)
    if upper_indices[0].size == 0:
        return 0.0
    return float(matrix[upper_indices].mean() * 0.2)
