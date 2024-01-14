import matplotlib.pyplot as plt
import numpy as np
import sys
import os


def main():
    
    if (len(sys.argv) < 2):
        print("ERR: Usage\npython3 loss_graph.py [file_loss.npy]")
        return 1

    nome_rete = os.path.splitext(sys.argv[1])[0]

    y = np.load(sys.argv[1])
    x = len(y)
    
    print(y)

    # Plotting the data
    plt.plot(range(x), y)

    # Adding labels and title
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.title('ciao')

    # Displaying the plot
    plt.show()


    return 0

if __name__ == "__main__":
    main()
