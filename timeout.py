import os
import json
import timeit
import time
import math
import random
from abc import ABC
from logging import INFO, DEBUG
from typing import Tuple, List, Optional, Dict

import flwr.server
import matplotlib.pyplot as plt
from flwr.common.logger import log

from flwr.common import Metrics, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion
from flwr.server.history import History
from flwr.server.server import EvaluateResultsAndFailures, evaluate_clients

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import flwr as fl
import tensorflow as tf

NUM_CLIENTS = 100
NUM_ROUNDS = 3
TIMEOUT_WINDOW = 5
CLIENT_TIMEOUT = 10
TIMEOUT_CHANCE = 0.6
SHOULD_TIMEOUT = False
DYNAMIC_TIMEOUT = False

CLIENT_VERBOSE = 3


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


class CustomClientManager(fl.server.client_manager.SimpleClientManager):
    def __init__(self) -> None:
        super().__init__()

    def sample(
            self,
            num_clients: int,
            min_num_clients: Optional[int] = None,
            criterion: Optional[Criterion] = None,
    ) -> List[ClientProxy]:

        """Sample a number of Flower ClientProxy instances."""
        # Block until at least num_clients are connected.
        if min_num_clients is None:
            min_num_clients = num_clients
        self.wait_for(min_num_clients)
        # Sample clients which meet the criterion
        available_cids = list(self.clients)
        if criterion is not None:
            available_cids = [
                cid for cid in available_cids if criterion.select(self.clients[cid])
            ]

        if num_clients > len(available_cids):
            log(
                INFO,
                "Sampling failed: number of available clients"
                " (%s) is less than number of requested clients (%s).",
                len(available_cids),
                num_clients,
            )
            return []

        sampled_cids = random.sample(available_cids, num_clients)
        return [self.clients[cid] for cid in sampled_cids]


class ClientManager(fl.server.client_manager.ClientManager, ABC):
    def __init__(self):
        super().__init__()


class CustomServer(fl.server.Server):
    def __init__(self):
        super().__init__(client_manager=CustomClientManager(),
                         strategy=fl.server.strategy.FedAvg(
                             fraction_fit=0.1,
                             fraction_evaluate=0.1,
                             min_fit_clients=3,
                             min_evaluate_clients=2,
                             min_available_clients=10,
                             evaluate_metrics_aggregation_fn=weighted_average,
                             on_fit_config_fn=fit_config,
                         )),

    # pylint: disable=too-many-locals
    def fit(self, num_rounds: int, timeout: Optional[float]) -> History:
        """Run federated averaging for a number of rounds."""
        history = History()

        # Initialize parameters
        log(INFO, "Initializing global parameters CUSTOM SERVER")
        self.parameters = self._get_initial_parameters(timeout=timeout)
        log(INFO, "Evaluating initial parameters CUSTOM SERVER")
        res = self.strategy.evaluate(0, parameters=self.parameters)
        if res is not None:
            log(
                INFO,
                "initial parameters (loss, other metrics): %s, %s",
                res[0],
                res[1],
            )
            history.add_loss_centralized(server_round=0, loss=res[0])
            history.add_metrics_centralized(server_round=0, metrics=res[1])

        # Run federated learning for num_rounds
        log(INFO, "FL starting")
        start_time = timeit.default_timer()

        for current_round in range(1, num_rounds + 1):
            # Train model and replace previous global model
            res_fit = self.fit_round(server_round=current_round, timeout=timeout)
            print("round_timeout: ", timeout)
            if res_fit:
                parameters_prime, _, _ = res_fit  # fit_metrics_aggregated
                if parameters_prime:
                    self.parameters = parameters_prime

            # Evaluate model using strategy implementation
            res_cen = self.strategy.evaluate(current_round, parameters=self.parameters)
            if res_cen is not None:
                loss_cen, metrics_cen = res_cen
                log(
                    INFO,
                    "fit progress: (%s, %s, %s, %s)",
                    current_round,
                    loss_cen,
                    metrics_cen,
                    timeit.default_timer() - start_time,
                )
                history.add_loss_centralized(server_round=current_round, loss=loss_cen)
                history.add_metrics_centralized(
                    server_round=current_round, metrics=metrics_cen
                )

            # Evaluate model on a sample of available clients
            res_fed = self.evaluate_round(server_round=current_round, timeout=timeout)
            if res_fed:
                loss_fed, evaluate_metrics_fed, results_failures = res_fed
                results = len(results_failures[0])
                failures = len(results_failures[1])
                print(f"success: ${results}, failures: ${failures}")

                if DYNAMIC_TIMEOUT:
                    if results == 0:
                        timeout *= 2
                    elif failures // results > 2:
                        timeout *= 1.5

                if loss_fed:
                    history.add_loss_distributed(
                        server_round=current_round, loss=loss_fed
                    )
                    history.add_metrics_distributed(
                        server_round=current_round, metrics=evaluate_metrics_fed
                    )

        # Bookkeeping
        end_time = timeit.default_timer()
        elapsed = end_time - start_time
        log(INFO, "FL finished in %s", elapsed)
        return history

    def evaluate_round(
            self,
            server_round: int,
            timeout: Optional[float],
    ) -> Optional[
        Tuple[Optional[float], Dict[str, Scalar], EvaluateResultsAndFailures]
    ]:
        """Validate current global model on a number of clients."""

        # Get clients and their respective instructions from strategy
        client_instructions = self.strategy.configure_evaluate(
            server_round=server_round,
            parameters=self.parameters,
            client_manager=self._client_manager,
        )
        if not client_instructions:
            log(INFO, "evaluate_round %s: no clients selected, cancel", server_round)
            return None
        log(
            DEBUG,
            "evaluate_round %s: strategy sampled %s clients (out of %s)",
            server_round,
            len(client_instructions),
            self._client_manager.num_available(),
        )

        # Collect `evaluate` results from all clients participating in this round
        results, failures = evaluate_clients(
            client_instructions,
            max_workers=self.max_workers,
            timeout=timeout,
        )
        log(
            DEBUG,
            "evaluate_round %s received %s results and %s failures",
            server_round,
            len(results),
            len(failures),
        )

        # Aggregate the evaluation results
        aggregated_result: Tuple[
            Optional[float],
            Dict[str, Scalar],
        ] = self.strategy.aggregate_evaluate(server_round, results, failures)

        loss_aggregated, metrics_aggregated = aggregated_result
        return loss_aggregated, metrics_aggregated, (results, failures)


def save_result(strategy: str, accuracy: dict[str, list[tuple[int, bool | bytes | float | int | str]]], losses: dict[str, list[tuple[int, bool | bytes | float | int | str]]]) -> None:
    results = {"strategy": strategy, "num_rounds": NUM_ROUNDS, "num_clients": NUM_CLIENTS,
               "should_timeout": SHOULD_TIMEOUT, "dynamic_timeout": DYNAMIC_TIMEOUT, "timeout_chance": TIMEOUT_CHANCE,
               "timeout_window": TIMEOUT_WINDOW, "accuracy": accuracy["accuracy"], "losses": losses}

    f = open("results.txt", "a")

    json_object = json.dumps(results, indent=4)
    f.write(json_object)
    f.write(",\n")
    f.close()

def display_results() -> None:
    return

def main() -> None:
    # Start Flower simulation
    result = fl.simulation.start_simulation(
        ray_init_args={"include_dashboard": False},
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=TIMEOUT_WINDOW),
        client_manager=CustomClientManager(),
        server=CustomServer(),
    )

    save_result("FedAvgM", result.metrics_distributed, result.losses_distributed)

    """ 
    plt.plot(*zip(*result.metrics_distributed['accuracy']))
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['Train'], loc='upper left')
    plt.show()

    plt.plot(*zip(*result.losses_distributed))
    plt.title('model losses')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['Loss'], loc='upper left')
    plt.show()
"""


if __name__ == "__main__":
    main()
