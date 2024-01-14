import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def main():
    
    if (len(sys.argv) < 2):
        print("ERR: Usage\npython3 loss_graph.py [file_loss.npy]")
        return 1

    y = np.load(sys.argv[1])
    x = len(y)

    print(y)
    print(x)


    return 0

if __name__ == "__main__":
    main()
