from matplotlib import pyplot as plt
from numpy import *



# def my_plot(num, s):
#     times=list()
#     with open("script.sh.e"+num) as f:
#         for line in f:
#             if not line.startswith("real"): continue
#             line=line.split()[1]
#             assert line.endswith("s")
#             line=line[:-1]
#             [mins,secs]=line.split("m")
#             time=float(secs)+60*float(mins)
#             # line=line.replace("0m", "")
#             # line=line.replace("s", "")
#             times.append(time)

#     X=array(range(len(times)))
#     Y=array(times)
#     plt.plot(X,Y,s)


# my_plot("01", "ro")
# my_plot("02", "go")
# my_plot("03", "bo")
# my_plot("04", "co")
# my_plot("05", "mo")


def my_plot(filename, s):
    times=list()
    with open(filename) as f:
        for line in f:
            if not line.startswith("real"): continue
            line=line.split()[1]
            assert line.endswith("s")
            line=line[:-1]
            [mins,secs]=line.split("m")
            time=float(secs)+60*float(mins)
            # line=line.replace("0m", "")
            # line=line.replace("s", "")
            times.append(time)
    X=array(range(1,len(times)+1))
    # Y=times[0]/array(times)
    Y=17.585/array(times)
    print map(lambda x: round(x,2),Y)
    plt.plot(X,Y,s)


my_plot("out0601.txt", "ro")
my_plot("out0602.txt", "ro")
my_plot("out0401.txt", "go")
my_plot("out0402.txt", "go")
# my_plot("12", "bo")
# plt.axis([0,16.5,0,14])
plt.show()