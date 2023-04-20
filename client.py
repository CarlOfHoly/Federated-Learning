import math
import os
import random
import time
import flwr as fl
import tensorflow as tf


NUM_ROUNDS = 10
TIMEOUT_WINDOW = 100
TIMEOUT_CHANCE = 0.6
SHOULD_TIMEOUT = False
CLIENT_VERBOSE = 3
CLIENT_TIMEOUT = 10

class FlwrClient(fl.client.NumPyClient):
    def __init__(self, model, x_train, y_train) -> None:
        super().__init__()
        self.model = model
        split_idx = math.floor(len(x_train) * 0.9)  # Use 10% of x_train for validation
        self.x_train, self.y_train = x_train[:split_idx], y_train[:split_idx]
        self.x_val, self.y_val = x_train[split_idx:], y_train[split_idx:]
        self.should_timeout = (random.random() < TIMEOUT_CHANCE) and SHOULD_TIMEOUT

    def get_parameters(self, config):
        return self.model.get_weights()

    def fit(self, parameters, config):
        if self.should_timeout:
            time.sleep(CLIENT_TIMEOUT)

        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=2, verbose=CLIENT_VERBOSE)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=CLIENT_VERBOSE)
        return loss, len(self.x_val), {"accuracy": acc}
