import json
from matplotlib import pyplot as plt


def plot_graph(strategy: str, accuracy: list[tuple]) -> None:
    plt.plot(*zip(*accuracy), label=strategy)


def plot_results(file: str, title: str) -> None:
    f = open(file, "r")
    data = json.loads(f.read())

    for i in range(len(data)):
    #for i in range(len(data)//2, len(data)):
        result = data[i]
        plot_graph(result["strategy"], result["accuracy"])

    plt.title(title)
    plt.ylabel('accuracy')
    plt.xlabel('communication rounds')
    plt.legend()
    plt.show()

    f.close()


def plot_losses(file: str, title: str) -> None:
    f = open(file, "r")
    data = json.loads(f.read())

    for i in range(len(data)):
        result = data[i]
        plot_graph(result["strategy"], result["losses"])

    plt.title(title)
    plt.ylabel('losses')
    plt.xlabel('communication rounds')
    plt.legend()
    plt.show()

    f.close()


def main() -> None:
    #plot_results("results/mnist_no_timeout.json")
    #plot_results("results/mnist_clients_normalized_lower.json", "Accuracy - Normalized Lower")
    plot_results("results/mnist_clients_normalized_mean.json", "Accuracy - Normalized Mean")
    plot_results("results/mnist_clients_normalized_upper.json", "Accuracy - Normalized Upper")
    #plot_results("results/mnist_dynamic_timeout.json", "Accuracy - Dynamic Timeout")
    #plot_losses("results/mnist_dynamic_timeout.json", "Losses - Dynamic Timeout")


if __name__ == '__main__':
    main()
