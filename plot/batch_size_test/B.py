from numpy import *

def my_plot(filename):
    times=list()
    # c=0
    with open(filename) as f:
        for line in f:
            if not line.startswith("Seconds: "): continue
            line=line.split()[1]
            times.append(float(line))
            print mean(times[1:])


my_plot("stdout02.txt")