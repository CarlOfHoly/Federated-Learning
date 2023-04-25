import json
from matplotlib import pyplot as plt


def plot_graph(strategy: str, accuracy: list[tuple]) -> None:
    plt.plot(*zip(*accuracy), label=strategy)


def plot_results(file: str) -> None:
    f = open(file, "r")
    data = json.loads(f.read())

    for i in range(len(data)):
        result = data[i]
        plot_graph(result["strategy"], result["accuracy"])

    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('communication rounds')
    plt.legend()
    plt.show()

    f.close()


def plot_losses(file: str) -> None:
    f = open(file, "r")
    data = json.loads(f.read())

    for i in range(len(data)):
        result = data[i]
        plot_graph(result["strategy"], result["losses"])

    plt.title('model losses')
    plt.ylabel('losses')
    plt.xlabel('communication rounds')
    plt.legend()
    plt.show()

    f.close()


def main() -> None:
    plot_results("results/mnist_no_timeout.json")


if __name__ == '__main__':
    main()
