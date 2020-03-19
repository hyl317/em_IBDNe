#some miscellaneous functions
def newton(f,Df,x0,epsilon,max_iter, X, Y, prev, interval):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn, X, Y, prev, interval)
        if abs(fxn) < epsilon:
            #print('Found solution after',n,'iterations.')
            return xn
        Dfxn = Df(xn, X, Y, prev, interval)
        if Dfxn == 0:
            #print('Zero derivative. No solution found.')
            return None
        xn = xn - fxn/Dfxn
    #print('Exceeded maximum iterations. No solution found.')
    return None


#FOR TESTING PURPOSE: return a numpy array of the reference FIN effective population size over the past 100 generations
def refFinNe():
    growth_rate1 = 0.0247
    growth_rate2 = 0.182
    N_0 = 1000
    N_curr = N_0
    N = [N_0]
    for g in np.arange(99, 0, -1):
        if g >= 13:
            N_curr = N_curr*(np.exp(growth_rate1))
        else:
            N_curr = N_curr*(np.exp(growth_rate2))
        N.insert(0, N_curr)
    return np.array(N)

#FOR TESTING PURPOSE: return a numpy array of the reference EUR effective population size over the past 100 generations

def refEurNe():
    eur_N_0 = 7.13e+04
    eur_growth_rate = 0.0195
    eur_N_curr = eur_N_0
    N = [N_0]
    for g in np.arange(99, 0, -1):
        eur_N_curr = eur_N_curr*(np.exp(eur_growth_rate))
        N.insert(0, eur_N_curr)
    return np.array(N)


def initializeN_Uniform(maxGen, Ne):
    return np.full(maxGen, Ne)

def initializeN_autoreg(maxGen):
    #initialize N, the population size trajectory
    phi = 0.98
    ar = np.array([1, -phi])
    MU, sigma = math.log(10000), math.log(10000)/10
    #specify a AR(1) model, by default the mean 0 since it's a stationary time series
    N = sm.tsa.arma_generate_sample(ar, np.array([1]), maxGen, scale=math.sqrt(1-phi**2)*sigma)
    return np.exp(N + MU) #now N has mean 10,000

def initializeT_Uniform(numBins, maxGen):
    T = np.full((numBins, maxGen+1), np.log(1/(maxGen+1)))
    return T

def initializeT_Random(numBins, maxGen):
    T = np.random.rand(numBins, maxGen+1)
    return T/T.sum(axis=1)[:, np.newaxis]