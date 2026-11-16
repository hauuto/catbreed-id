"""Điểm từ chối, chép nguyên từ notebooks/kaggle_runner.ipynb (mục 11.1–11.2)."""
import numpy as np
from scipy.special import logsumexp, softmax
from sklearn.covariance import LedoitWolf
from sklearn.neighbors import NearestNeighbors

CLASS_INDICES = list(range(12))  # len(CAT_BREEDS) trong notebook
METHODS = ["MSP", "MaxLogit", "Energy", "Mahalanobis", "KNN"]
LOGIT_METHODS, DISTANCE_METHODS = METHODS[:3], METHODS[3:]


# ─── 11.1 MSP, MaxLogit và Energy từ logits ──────────────────────────────────────────────────────────
def logit_scores(logits, temperature=1.0):
    # Phụ lục B: softmax và logsumexp của SciPy trừ logit lớn nhất trước khi lấy mũ
    logits = np.asarray(logits, dtype=np.float64)
    scaled = logits / temperature
    return {
        "MSP": softmax(scaled, axis=1).max(axis=1),
        "MaxLogit": logits.max(axis=1),
        "Energy": temperature * logsumexp(scaled, axis=1),
    }


# ─── 11.2 Mahalanobis và KNN từ vector đặc trưng ─────────────────────────────────────────────────────
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
        covariance = LedoitWolf().fit(x - self.means[y])  # §2.6.3: hiệp phương sai chung của phần dư
        self.precision, self.shrinkage = covariance.precision_, float(covariance.shrinkage_)
        return self.index(unit_norm(x), k)

    def index(self, bank, k):
        self.bank, self.k = bank, k  # §2.6.4: ngân hàng train đã chuẩn hóa L2
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
        np.savez_compressed(
            path,
            means=self.means,
            precision=self.precision,
            shrinkage=self.shrinkage,
            bank=self.bank,
            k=self.k,
        )

    @classmethod
    def load(cls, path):
        with np.load(path, allow_pickle=False) as stored:
            distance = cls()
            distance.means, distance.precision = stored["means"], stored["precision"]
            distance.shrinkage = float(stored["shrinkage"])
            return distance.index(stored["bank"], int(stored["k"]))


def rejection_scores(arrays, distance):
    return {**logit_scores(arrays["logits"]), **distance.score(arrays["features"])}
