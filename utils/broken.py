import sys
import os

def broken(dir1, dir2):
    for i, j in zip(os.listdir(dir1), os.listdir(dir2)):
        if(i != j):
            print(i, j)
            return True
    return False

def main():

    if(len(sys.argv) < 3):
        print("ERR: Usage\npython3 [dir1] [dir2]")
        return 1
    
    print(broken(sys.argv[1], sys.argv[2]))

    return 0

if __name__ == "__main__":
    main()
