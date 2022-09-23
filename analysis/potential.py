import numpy as np
import matplotlib.pyplot as plt

def main():
    lines = []
    skip = True
    with open('../dataset/test/train/pair_potential.txt', 'r') as f:
        for line in f.readlines():
            if line[0] == "1":
                skip = False
            if skip:
                continue
            else:
                if line[0] == "#":
                    break
                line = line.replace("\n", "")
                line = line.split()
                line = [float(i) for i in line]
                lines.append(line)

    data = np.array(lines, dtype=np.float32)
    plt.plot(data[:, 1], data[:, 2], label='energy')
    plt.plot(data[:, 1], data[:, 3], label='force')
    plt.legend()
    plt.ylim(-1, 2)
    plt.savefig('pair_potential.png')


if __name__ == '__main__':
    main()

