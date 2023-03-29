import os
import time
import math
import random
from typing import Tuple, List
import matplotlib.pyplot as plt

from flwr.common import Metrics

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

NUM_CLIENTS = 100
TIMEOUT_WINDOW = 10
TIMEOUT_CHANCE = 0.05
SHOULD_TIMEOUT = False



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
            time.sleep(TIMEOUT_WINDOW+1)

        self.model.set_weights(parameters)
        self.model.fit(self.x_train, self.y_train, epochs=2, verbose=2)
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, acc = self.model.evaluate(self.x_val, self.y_val, verbose=2)
        return loss, len(self.x_val), {"accuracy": acc}


def client_fn(cid: str) -> fl.client.Client:
    # Load model
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])

    # Load data partition (divide MNIST into NUM_CLIENTS distinct partitions)
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    partition_size = math.floor(len(x_train) / NUM_CLIENTS)
    idx_from, idx_to = int(cid) * partition_size, (int(cid) + 1) * partition_size
    x_train_cid = x_train[idx_from:idx_to] / 255.0
    y_train_cid = y_train[idx_from:idx_to]

    # Create and return client
    return FlwrClient(model, x_train_cid, y_train_cid)

def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    return {"accuracy": sum(accuracies) / sum(examples)}

def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Perform two rounds of training with one local epoch, increase to two local
    epochs afterwards.
    """
    config = {
        "server_round": server_round,  # The current round of federated learning
        "local_epochs": 1
    }
    return config

def main() -> None:
    # Start Flower simulation
    result = fl.simulation.start_simulation(
        ray_init_args={"include_dashboard": False},
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=10, round_timeout=TIMEOUT_WINDOW),
        strategy=fl.server.strategy.FedAvg(
            fraction_fit=0.1,
            fraction_evaluate=0.1,
            min_fit_clients=3,
            min_evaluate_clients=2,
            min_available_clients=10,
            evaluate_metrics_aggregation_fn=weighted_average,
            on_fit_config_fn=fit_config,
        ),
    )

    plt.plot(*zip(*result.metrics_distributed['accuracy']))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()
    """ 

    plt.plot(*zip(*result.losses_distributed))
    plt.title('model losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Loss'], loc='upper left')
    plt.show()
"""

if __name__ == "__main__":
    main()
