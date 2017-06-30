

start="""#!/bin/bash
#PBS -l nodes=1
#PBS -l walltime=00:00:59
"""

directories=["01","02","03"]
quantity=3
letters="abcdefghijklmnopqrstuvwxyz"


for d in directories:
    for l in letters[:quantity]:
        with open("b"+d+l+".sh", "w") as f:
            f.write(start)
            f.write("cd ~/textmaker02/for/"+d+"\n")
            f.write("./a.out > ~/stdout"+d+l+".txt"+"\n")
            # f.write("make")
            # "~/textmaker02/for/01/a.out > ~/stdout01a.txt"

with open("script.sh", "w") as f:
    f.write("#!/bin/bash\n")
    for d in directories:
        f.write("cd ~/textmaker02/for/"+d+"\n")
        f.write("make\n")

    for d in directories:
        for l in letters[:quantity]:
            f.write("qsub "+"b"+d+l+".sh"+"\n")
