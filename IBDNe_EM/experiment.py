#approximating infinite sums

from scipy.integrate import quad
import numpy as np

maxGen = 100
Ne = 2e4

N = np.full(maxGen, Ne)
chromLen = np.full(30, 100) #suppose 30 chromosomes of length 100Mb each
n = 1000
n_p = (2*n*(2*n-2))/2
N_past = N[-1]
CONSTANT = np.exp(np.log(n_p) - np.log(2*N_past) + np.sum(np.log(1-1/(2*N))))
C = 2
def log_partB(g, N_g, maxGen, C, chromLen):
    part3 = np.sum((C*g/50 + 1)*chromLen) - len(chromLen)*(C**2)*g/50
    part2 = -C*g/50
    part1 = (g-maxGen-1)*np.log(1-1/(2*N_g))
    return np.exp(part1 + part2 + np.log(part3))


integral, err = quad(log_partB, maxGen+1, np.inf, args=(N_past, maxGen, C, chromLen))
print(integral)