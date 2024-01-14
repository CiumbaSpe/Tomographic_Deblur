import os
import sys
import shutil

def main():
    if(len(sys.argv) < 2):
        print("ERR: Usage\npython3 [dir_to_sync]")
        return 1

    files = sorted(os.listdir(sys.argv[1]))

    os.mkdir("ordered_dir")

    for i in files:
        print(i)
        shutil.copy(sys.argv[1] + "/" + i, "ordered_dir/" + i)
    
    #os.rmdir(sys.argv[1])
    #os.rename("new_dir", sys.argv[1])

    return 0

if __name__ == "__main__":
    main()
