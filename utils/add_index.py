import sys
from os.path import exists


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage is python add_index.py [FILE] [NEW_FILE]")
    
    filename = sys.argv[1]
    f = open(filename,'r').read().splitlines()
    paths = []
    for l in f:
        splitted = l.split(' ')
        paths.append(splitted[0])

    out_file = sys.argv[2]
    with open(out_file, "w") as dst:
        for i, p in enumerate(paths):
          if not exists(p):
            raise ValueError(f"Path {p} does not exist")
          print(f"Writing {p}")
          dst.write(f"{p} {i}\n")


