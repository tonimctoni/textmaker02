all:
	g++ main.cpp -Wall -Wextra -pedantic -std=c++0x -O3 -funroll-loops -ftree-loop-distribution -march=native -fopenmp
run: all
	./a.out