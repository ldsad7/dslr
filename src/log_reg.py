from typing import Generator, Tuple, List, Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import shuffle

np.random.seed(42)


class LogReg:
    def __init__(self, verbose: bool = False, learning_rate: float = 0.5, epsilon: float = 10e-2,
                 batch_size: float = 1.0, validation_size: float = 0.1, max_num_of_losses: int = 20):
        self.coefficients: np.ndarray = np.empty(0)
        self._best_coefficients: np.ndarray = np.empty(0)
        self._features: np.ndarray = np.empty(0)
        self._target: np.ndarray = np.empty(0)
        self._validation_features: np.ndarray = np.empty(0)
        self._validation_target: np.ndarray = np.empty(0)

        self._lr: float = learning_rate
        self._epsilon: float = epsilon
        self._batch_size: float = batch_size
        self._validation_size: float = validation_size
        self._batch_borderline: float = 0.99
        self._min_loss: float = 0
        self._num_of_losses: float = 0

        self._max_num_of_losses: int = max_num_of_losses

        self._verbose: bool = verbose

        self._batch_generator: Optional[Generator[Tuple[np.ndarray, np.ndarray], None, None]] = None

        self._losses: List[List[float]] = []

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.e ** -z)

    def _h(self, features: np.ndarray):
        return self._sigmoid(features.dot(self.coefficients))

    def count_log_reg_loss(self, features: np.ndarray, target: np.ndarray) -> float:
        h = self._h(features)
        loss = -(self._lr / target.size) * (np.log(h) * target + np.log(1 - h) * (1 - target)).sum()
        return loss

    def _gen_batch(self) -> Generator[Tuple[np.ndarray, np.ndarray], None, None]:
        size = self._target.size
        while True:
            X, y = shuffle(self._features, self._target)
            step_size = int(size * self._batch_size) or 1
            for i in range(0, size, step_size):
                yield X[i:i + step_size], y[i:i + step_size]

    def fit(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._num_of_losses = 0
        features, target = shuffle(features, target)
        size: int = int(self._validation_size * target.size)

        self._validation_features: np.ndarray = features[:size]
        self._validation_target: np.ndarray = target[:size]

        self._features: np.ndarray = features[size:]
        self._target: np.ndarray = target[size:]

        self._batch_generator: Generator[Tuple[np.ndarray, np.ndarray], None, None] = self._gen_batch()

        self.coefficients: np.ndarray = np.zeros((self._features.shape[1], 1))

        self._losses.append([])

        self._min_loss = self.count_log_reg_loss(self._validation_features, self._validation_target)
        self._losses[-1].append(self._min_loss)
        self._best_coefficients = self.coefficients

        try:
            while self._make_step():
                pass
        finally:
            return self._best_coefficients

    def _make_step(self) -> bool:
        features, target = next(self._batch_generator)

        gradient = (1 / target.size) * (self._h(features) - target).T.dot(features)

        self.coefficients -= self._lr * gradient.T

        return self.check_stop_conditions(gradient)

    def check_stop_conditions(self, gradient: np.ndarray) -> bool:
        curr_loss: float = self.count_log_reg_loss(self._validation_features, self._validation_target)
        self._losses[-1].append(curr_loss)

        if curr_loss >= self._min_loss:
            self._num_of_losses += 1
        else:
            self._min_loss = curr_loss
            self._best_coefficients = self.coefficients
            self._num_of_losses = 0
        if self._verbose:
            print(f"current LogReg loss on validation dataset: {curr_loss:.10f}, "
                  f"current num of losses: {self._num_of_losses:3}", end='')
        if self._num_of_losses > self._max_num_of_losses:
            if self._verbose:
                print(f"\ncurrent num of losses exceeds the max num of losses so we stop here...")
            return False

        if self._batch_size >= self._batch_borderline:
            step_size: float = np.sqrt(self._lr * gradient.dot(gradient.T))[0, 0]
            if self._verbose:
                print(f", current step size: {step_size:.15f}")
            if step_size < self._epsilon:
                if self._verbose:
                    print(f"current step size is lower than given epsilon ({self._epsilon} so we stop here...")
                return False
        elif self._verbose:
            print()

        return True

    def predict(self, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        self._features: np.ndarray = features
        self.coefficients: np.ndarray = coefficients
        return self._h(self._features)

    def draw_graph(self, labels: List[str], save_to_image: str = 'pictures/log_reg_loss_plot.png'):
        assert len(labels) == len(self._losses)

        fig, ax = plt.subplots(figsize=(14, 10), dpi=90)
        for losses, label in zip(self._losses, labels):
            ax.plot(np.arange(len(losses)), losses, label=label)

        plt.xlabel("Steps")
        plt.ylabel("Validation Loss")

        plt.title(f"LogReg validation loss plot ("
                  f"batch_size={self._batch_size}, lr={self._lr}, validation_size={self._validation_size}"
                  f")", fontsize="x-large")

        plt.legend()

        if save_to_image is not None:
            if not save_to_image.strip():
                save_to_image = f'pictures/log_reg_loss_plot_{self._batch_size}.png'
            plt.savefig(save_to_image, dpi=300)
        plt.show()
