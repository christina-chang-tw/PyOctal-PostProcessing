import numpy as np
import matplotlib.pyplot as plt

from postprocessing.utils.formatter import Publication

def calculate_n_1550(n, p):
    return -(5.4E-22*n**1.011 + 1.53E-18*p**0.838)

def calculate_alpha_1550(n, p):
    return 8.88E-21*n**1.167 + 5.84E-20*p**1.109

def calculate_n_2000(n, p):
    return -(1.91E-21*n**0.992 + 2.28E-18*p**0.841)

def calculate_alpha_2000(n, p):
    return 3.22E-20*n**1.149 + 6.21E-20*p**1.119

def main():
    Publication.set_basics()

    N = np.logspace(16, 19, 1000) # in m^-3
    P = np.logspace(16, 19, 1000) # in m^-3
    zeros = np.zeros(len(N))
    alpha_n = calculate_alpha_1550(N, zeros)
    alpha_p = calculate_alpha_1550(zeros, P)

    n_n = calculate_n_1550(N, zeros)
    n_p = calculate_n_1550(zeros, P)

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(N, alpha_n*1E-2, label=r'$\Delta \alpha = 8.88\times 10^{-21}\Delta N_e^{1.167}$')
    ax[0].plot(P, alpha_p*1E-2, label=r'$\Delta \alpha = 5.84\times 10^{-20}\Delta N_h^{1.109}$')
    ax[0].set_xscale('log')
    ax[0].set_yscale('log')
    ax[0].set_xlabel(r'$\Delta N_e/\Delta N_h$ [cm$^{-3}$]')
    ax[0].set_ylabel(r'$\Delta \alpha$ $[cm^{-1}]$ ')
    ax[0].legend()

    ax[1].plot(N, np.absolute(n_n), label=r'$-\Delta n = 5.4\times 10^{-22}\Delta N_e^{1.011}$')
    ax[1].plot(P, np.absolute(n_p), label=r'$-\Delta n = 1.53\times 10^{-18}\Delta N_h^{0.838}$')
    ax[1].set_xscale('log')
    ax[1].set_yscale('log')
    ax[1].set_xlabel(r'$\Delta N_e/\Delta N_h$ [cm$^{-3}$]')
    ax[1].set_ylabel(r'-$\Delta$n')
    ax[1].legend()

    ax[0].set_title("(a)")
    ax[1].set_title("(b)")
    fig.tight_layout()
    fig.savefig("alpha_n.pdf")
    plt.show()

if __name__ == '__main__':
    main()



