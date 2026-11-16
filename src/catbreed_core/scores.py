"""Notebook-compatible rejection scores."""

import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

METHODS = ["MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN"]
CLASS_INDICES = list(range(12))


def logit_scores(logits, temperature=1.0):
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    return {
        "MSP": softmax(scaled, axis=1).max(axis=1),
        "MaxLogit": logits.max(axis=1),
        "Energy": temperature * logsumexp(scaled, axis=1),
    }


def unit_norm(x):
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


class DistanceScores:
    def fit(self, features, labels, k):
        x, y = np.asarray(features, dtype=np.float64), np.asarray(labels)
        if set(y.tolist()) != set(CLASS_INDICES):
            raise ValueError("Train bank phải có đủ 12 lớp.")
        if len(x) < k:
            raise ValueError("Train bank nhỏ hơn k; không tự đổi k.")
        self.means = np.stack([x[y == c].mean(axis=0) for c in CLASS_INDICES])
        covariance = LedoitWolf().fit(x - self.means[y])
        self.precision, self.shrinkage = covariance.precision_, float(covariance.shrinkage_)
        return self.index(unit_norm(x), k)

    def index(self, bank, k):
        self.bank, self.k = bank, k
        self.neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", algorithm="brute").fit(bank)
        return self

    def score(self, features):
        x = np.asarray(features, dtype=np.float64)
        xp, mp = x @ self.precision, self.means @ self.precision
        squared = (
            (xp * x).sum(axis=1)[:, None]
            - 2 * xp @ self.means.T
            + (mp * self.means).sum(axis=1)[None, :]
        )
        distances = self.neighbors.kneighbors(unit_norm(x), return_distance=True)[0]
        return {
            "Mahalanobis": -np.maximum(squared, 0).min(axis=1),
            "KNN": -distances[:, self.k - 1],
        }

    def save(self, path):
        np.savez_compressed(path, means=self.means, precision=self.precision,
                            shrinkage=self.shrinkage, bank=self.bank, k=self.k)

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            distance = cls()
            distance.means = stored["means"]
            distance.precision = stored["precision"]
            distance.shrinkage = float(stored["shrinkage"]) if "shrinkage" in stored else None
            return distance.index(stored["bank"], int(stored["k"]))
