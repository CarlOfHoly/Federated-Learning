import math
import random
import time
import flwr as fl
import numpy as np

CONFIDENCE_INTERVAL = [1.804, 2, 2.196]
NUM_ROUNDS = 60
TIMEOUT_WINDOW = CONFIDENCE_INTERVAL[0] * 2
NORMALIZE_DISTRIBUTION = True
INCLUDE_OUTLIER = False
CLIENT_VERBOSE = 3
MU, SIGMA = 2, 1


class MnistClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]
        self.time_delay = abs(np.random.normal(MU, SIGMA/10))
        self.should_outlie = random.random() < 0.01

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        if NORMALIZE_DISTRIBUTION:
            time.sleep(self.time_delay)
            print(self.time_delay)
        if INCLUDE_OUTLIER and self.should_outlie:
            print("I will sleep for 10")
            time.sleep(300)

        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=2, verbose=CLIENT_VERBOSE)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):

        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=CLIENT_VERBOSE)
        return loss, len(self.x_val), {"accuracy": acc}
