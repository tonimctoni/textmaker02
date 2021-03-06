from matplotlib import pyplot as plt
from numpy import *

def my_plot(filename, s):
    times=list()
    with open(filename) as f:
        for line in f:
            if not line.startswith("Error: "): continue
            line=line.split()[1]
            times.append(float(line))
    X=array(range(1,len(times)+1))
    Y=array(times)
    print map(lambda x: round(x,2),Y)
    plt.plot(X,Y,s)


my_plot("stdout01.txt", "ro") # one lstm layer
my_plot("stdout02.txt", "bo") # two lstm layers, less synapses
my_plot("stdout03.txt", "go") # two lstm layers, only slightly less synapses
# plt.axis([0,16.5,0,14])
plt.show()