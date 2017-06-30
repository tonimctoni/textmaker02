from matplotlib import pyplot as plt
from numpy import *

def my_plot(filename, s):
    times=list()
    # c=0
    with open(filename) as f:
        for line in f:
            # if line.startswith("learning_rate_divisor"):
            #     plt.axvline(x=c)
            if not line.startswith("Error: "): continue
            line=line.split()[1]
            times.append(float(line))
            # c+=1
    X=array(range(1,len(times)+1))
    Y=array(times)
    print map(lambda x: round(x,2),Y)
    plt.plot(X,Y,s)


my_plot("stdout01.txt", "ro")
my_plot("stdout02.txt", "go")
my_plot("stdout03.txt", "bo")
my_plot("stdout04.txt", "mo")
my_plot("stdout05.txt", "co")

# my_plot("stdout06.txt", "r^")
# my_plot("stdout07.txt", "g^")
# my_plot("stdout08.txt", "b^")
# my_plot("stdout09.txt", "m^")
# my_plot("stdout10.txt", "c^")
# plt.axis([0,16.5,0,14])
plt.show()