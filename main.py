from __future__ import annotations

import json
import os

import flwr as fl

from client import NUM_ROUNDS, NORMALIZE_DISTRIBUTION, TIMEOUT_WINDOW, MU, SIGMA
from server import CustomClientManager, CustomServer, client_fn_mnist, NUM_CLIENTS, DYNAMIC_TIMEOUT, STRATEGY

# Make TensorFlow logs less verbose
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


DATASET = "mnist"


def save_result(file: str, strategy: str, accuracy: dict[str, list[tuple[int, bool | bytes | float | int | str]]],
                losses: dict[str, list[tuple[int, bool | bytes | float | int | str]]], dataset: str) -> None:
    results = {"strategy": strategy, "num_rounds": NUM_ROUNDS, "num_clients": NUM_CLIENTS,
               "should_timeout": NORMALIZE_DISTRIBUTION, "dynamic_timeout": DYNAMIC_TIMEOUT, "sample_mean": MU,
               "standard_deviation": SIGMA,
               "timeout_window": TIMEOUT_WINDOW, "accuracy": accuracy["accuracy"], "losses": losses, "dataset": dataset}

    f = open(file, "a")

    json_object = json.dumps(results, indent=4)
    f.write(json_object)
    f.write(",\n")
    f.close()


def main() -> None:
    print("-----------------------")
    print(STRATEGY.value)
    print("-----------------------")
    result = fl.simulation.start_simulation(
        ray_init_args={"include_dashboard": False},
        client_fn=client_fn_mnist,
        num_clients=NUM_CLIENTS,
        client_resources={"num_cpus": 4},
        config=fl.server.ServerConfig(num_rounds=NUM_ROUNDS, round_timeout=TIMEOUT_WINDOW),
        client_manager=CustomClientManager(),
        server=CustomServer(),
    )


    save_result(f"results/{DATASET}_clients_normalized_lower.json", STRATEGY.value, result.metrics_distributed,
               result.losses_distributed, DATASET)


if __name__ == "__main__":
    main()
