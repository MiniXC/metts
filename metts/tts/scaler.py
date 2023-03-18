import numpy as np
import torch

class GaussianMinMaxScaler():
    """
    A min-max scaler that does the following, given the number of samples in the dataset N:
    1. Apply as square root to the data
    2. Apply a min-max scaler to the data ranging from the expected minimum and maximum of a Gaussian distribution with samples N
    """

    def __init__(self, width, for_tensors=True, floor=1e-6):
        self.expected_max = width / 2
        self.max = None
        self.min = None
        self._n = 0
        self.for_tensors = for_tensors
        self.floor = floor
        self._scale = None

    def partial_fit(self, X):
        scale_change = False
        X = X.flatten()
        if self.max is None:
            self.max = X.max()
            if self.for_tensors:
                self.max = self.max.detach()
            scale_change = True
        else:
            max_candidate = X.max()
            if max_candidate > self.max:
                if self.for_tensors:
                    self.max = max_candidate.detach()
                else:
                    self.max = max_candidate
                scale_change = True
        if self.min is None:
            self.min = X.min()
            if self.for_tensors:
                self.min = self.min.detach()
            scale_change = True
        else:
            min_candidate = X.min()
            if min_candidate < self.min:
                if self.for_tensors:
                    self.min = min_candidate.detach()
                else:
                    self.min = min_candidate
                scale_change = True
        if scale_change:
            if self.for_tensors:
                self._scale = torch.sqrt(self.max - self.min + self.floor)
            else:
                self._scale = np.sqrt(self.max - self.min + self.floor)
        # add numpy array size to n, regardless of shape
        self._n += len(X.flatten())

    def transform(self, X):
        X = X - self.min + self.floor
        if self.for_tensors:
            X = X.clip(min=self.floor)
            X = torch.sqrt(X)
            # print if any nan values
            if torch.isnan(X).any():
                print("NAN values in GaussianMinMaxScaler!")
        else:
            X = np.sqrt(X)
        X = X / self._scale
        X = X - 0.5
        X = X * self.expected_max
        return X

    def inverse_transform(self, X):
        X = X.detach()
        X = X / self.expected_max
        X = X + 0.5
        X = X * self._scale
        X = X ** 2
        X = X + self.min
        return X