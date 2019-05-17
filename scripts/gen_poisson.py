import sys
import random
import os

# Writes a configuration of equilibrium fully overlapping spheres
# Usage gen_poisson [n_dim] [n_spheres] [radius] [l_box] [outfile] [seedfile (optional)]
# Don't include extension on outfile, do include extension on optional seedfile
if __name__=="__main__":
    n_dim = int(sys.argv[1])
    n_spheres = int(sys.argv[2])
    radius = float(sys.argv[3])
    l_box = float(sys.argv[4])
    outfile = open(sys.argv[5] + ".dat","w")
    # Idea for reading and writing file as binary comes from
    # Multiple sources convincing that it is a good solution, plus
    # https://stackoverflow.com/questions/6787233/python-how-to-read-bytes-from-file-and-save-it
    # for help with how to do it
    if len(sys.argv) == 7:
        seedfile = open(sys.argv[6],"rb")
        seed = seedfile.read()
        seedfile.close()
    else:
        seed = os.urandom(32)
        print(seed)
        seedfile = open(sys.argv[5] + "_seed.binary", "wb")
        seedfile.write(seed)
        seedfile.close()
    random.seed(seed)
    
    outfile.write(str(n_dim) + " HS poly\n")
    outfile.write(str(n_spheres) + " 1\n")
    outfile.write(str(n_spheres) + "\n")


    for i in range(n_dim):
        for j in range(n_dim):
            if i == j:
                outfile.write(str(l_box))
            else:
                outfile.write(str(0.0))
            if j < n_dim - 1:
                outfile.write(" ")
        outfile.write("\n")
    
    for i in range(n_dim):
        outfile.write("T")
        if i < n_dim - 1:
            outfile.write(" ")
    outfile.write("\n")

    for i in range(n_spheres):
        for j in range(n_dim):
            coord = random.uniform(0.0,l_box);
            outfile.write(str(coord) + " ")
        outfile.write(str(radius)+"\n")
        
    outfile.close()
