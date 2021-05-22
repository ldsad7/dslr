import numpy as np


class LogReg:
    def __init__(self, verbose: bool = False, learning_rate: float = 0.5, epsilon: float = 10e-2):
        self.verbose = verbose
        self._features: np.ndarray = np.empty(0)
        self._target: np.ndarray = np.empty(0)
        self.coefficients: np.ndarray = np.empty(0)
        self._lr = learning_rate
        self._epsilon = epsilon

    @staticmethod
    def _sigmoid(z):
        return 1 / (1 + np.e ** -z)

    def _h(self):
        return self._sigmoid(self._features.dot(self.coefficients))

    def count_loss(self) -> float:
        """
        counts LogReg loss
        """
        h = self._h()
        loss = -(self._lr / self._features.shape[0]) * (
                np.log(h) * self._target + np.log(1 - h) * (1 - self._target)
        ).sum()
        return loss

    def fit(self, features: np.ndarray, target: np.ndarray) -> np.ndarray:
        self._features: np.ndarray = features
        self._target: np.ndarray = target
        self.coefficients: np.ndarray = np.zeros((self._features.shape[1], 1))

        while True:
            if self.verbose:
                print(f"current LogReg loss: {self.count_loss()}, making one more step...")
            if self._make_step():
                break

        return self.coefficients

    def _make_step(self) -> bool:
        need_to_stop = False

        h = self._h()
        gradient = (1 / self._features.shape[0]) * (h - self._target).T.dot(self._features)

        step_size: float = np.sqrt(gradient.dot(gradient.T))[0, 0]
        if self.verbose:
            print(f"current step size: {step_size}")
        if step_size >= self._epsilon:
            self.coefficients -= self._lr * gradient.T
        else:
            need_to_stop = True

        return need_to_stop

    def predict(self, features: np.ndarray, coefficients: np.ndarray) -> np.ndarray:
        self._features: np.ndarray = features
        self.coefficients: np.ndarray = coefficients
        return self._h()
