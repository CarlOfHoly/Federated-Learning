import json

from matplotlib import pyplot as plt


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


def main() -> None:
    plot_results("results.json")


if __name__ == '__main__':
    main()