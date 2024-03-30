# coding:UTF-8
# @Time: 2023/8/16 11:24
# @Author: Lulu Cao
# @File: trace_convergence.py
# @Software: PyCharm
import numpy as np
import matplotlib.pyplot as plt

def plot_convergence():
    fno = np.load("advection_f_3d_no.npz")["fitness_best"]
    fit = np.load("advection_f_3d.npz")["fitness_best"]
    gen = np.array(range(len(fno)))


    plt.plot(gen, fno, "b-", label="Minimum Fitness,PR-GPSR")
    plt.plot(gen, fit, "r-", label="Minimum Fitness,HD-TLGP")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("3D Advection2 Convergence")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("3d_advection.png")
    plt.show()


def plot_ind_increase():
    ind1 = np.load("heat_mean_ind_3d_with.npz")["ind_mean"]
    ind2 = np.load("heat_mean_ind_3d.npz")["ind_mean"]
    gen = np.array(range(len(ind1)))


    plt.plot(gen, ind1, "b-", label="HD-TLGP-with-pruning-operator")
    plt.plot(gen, ind2, "r-", label="HD-TLGP-without-pruning-operator")
    plt.xlabel("Generation")
    plt.ylabel("The AVG of Individuals")
    plt.title("The AVG of Individuals on 3D Heat ")
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig("3d_heat_ind.png")
    plt.show()
if __name__ == '__main__':
    plot_ind_increase()