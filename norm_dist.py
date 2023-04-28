import numpy as np
import matplotlib.pyplot as plt

def main() -> None:
    mu, sigma = 2, 0.1  # mean and standard deviation
    s = np.random.normal(mu, sigma, 1000)

    count, bins, ignored = plt.hist(s, 30, density=True)

    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    #plt.show()

    tot = 0
    for i in s:
        if 1.836<= i <= 2.164:
            tot += 1
    print(s)
    print(tot)
if __name__ == "__main__":
    main()
