#!/usr/bin/env python3
import csv
import numpy as np


def main():
    ans_julia_fp = open("./ans_julia.csv", "r")
    ans_cpp_fp = open("./ans_cpp.csv", "r")
    ans_julia_csv = csv.reader(ans_julia_fp, delimiter=",")
    ans_cpp_csv = csv.reader(ans_cpp_fp, delimiter=",")

    julia = []

    for i, row in enumerate(ans_julia_csv):
        julia.append([])
        for item in row:
            julia[i].append(float(item))

    julia = np.asarray(julia)
    cpp = []

    for i, row in enumerate(ans_cpp_csv):
        cpp.append([])
        for item in row:
            cpp[i].append(float(item))

    cpp = np.asarray(cpp)
    # print(julia - cpp)
    diff = julia - cpp
    print(np.all(diff < 0.0001))


if __name__ == '__main__':
    main()
