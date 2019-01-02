import numpy as np
import cmath
import math
import sympy


# range for loops, from start, to finish with increment as the stepsize
def myRange(start, finish, increment):
    myZero = 1e-17
    while (start <= finish+myZero):
        yield start
        start += increment
        
        

# Periodic Linear Extension: If I want to make sure during optimization that parameters stay between c & b
def paramMapping(x, c, d):

    if ((x>=c) & (x<=d)):
        
        y = x

    else:
        
        range = d-c
        n = math.floor((x-c)/range)
        if (n%2 == 0):
            y = x - n*range;
        else:
            y = d + n*range - (x-c)
            
    return y

#def eValue(params, marketPrices, maturities, strikes, r, q, S0, alpha, eta, n, model):
def eValue(params, *args):
    
    marketPrices = args[0]
    maturities = args[1]
    strikes = args[2]
    r = args[3]
    q = args[4]
    S0 = args[5]
    alpha = args[6]
    eta = args[7]
    n = args[8]
    model = args[9]

    lenT = len(maturities)
    lenK = len(strikes)
    
    modelPrices = np.zeros((lenT, lenK))
    #print(marketPrices.shape)

    count = 0
    mae = 0
    for i in range(lenT):       
        for j in range(lenK):
            count  = count+1
            T = maturities[i]
            K = strikes[j]
            [km, cT_km] = genericFFT(params, S0, K, r, q, T, alpha, eta, n, model)  
            modelPrices[i,j] = cT_km[0]
            tmp = marketPrices[i,j]-modelPrices[i,j]
            mae += tmp**2
    
    rmse = 1/count*math.sqrt(mae)
    return rmse

def coth(x): 
    return 1/(np.sinh(x) / np.cosh(x))

def generic_CF(u, params, S0, r, q, T, model):
    
    if (model == 'GBM'):
        
        sig = params[0]
        mu = np.log(S0) + (r-q-sig**2/2)*T
        a = sig*np.sqrt(T)
        phi = np.exp(1j*mu*u-(a*u)**2/2)
        
    elif(model == 'Heston'):
        
        kappa  = params[0]
        theta  = params[1]
        sigma  = params[2]                           # Vol of Vol
        rho    = params[3]
        v0     = params[4]
        
        kappa = paramMapping(kappa,0.1, 20)          # Why this transformation? for optimization
        theta = paramMapping(theta,0.001, 0.4)       # want to make sure during optimization that parameters stay between c & b
        sigma = paramMapping(sigma,0.01, 0.6)
        rho   = paramMapping(rho  ,-1.0, 1.0)
        v0    = paramMapping(v0   ,0.005, 0.25)
        
        tmp = (kappa-1j*rho*sigma*u)
        g = np.sqrt((sigma**2)*(u**2+1j*u)+tmp**2)
        
        pow1 = 2*kappa*theta/(sigma**2)
        
        numer1 = (kappa*theta*T*tmp)/(sigma**2) + 1j*u*T*r + 1j*u*math.log(S0)
        log_denum1 = pow1 * np.log(np.cosh(g*T/2)+(tmp/g)*np.sinh(g*T/2))
        tmp2 = ((u*u+1j*u)*v0)/(g/np.tanh(g*T/2)+tmp)
        log_phi = numer1 - log_denum1 - tmp2
        phi = np.exp(log_phi)

    elif (model == 'VG'):
        
        sigma  = params[0];
        nu     = params[1];
        theta  = params[2];

        if (nu == 0):
            mu = math.log(S0) + (r-q - theta -0.5*sigma**2)*T
            phi  = math.exp(1j*u*mu) * math.exp((1j*theta*u-0.5*sigma**2*u**2)*T)
        else:
            mu  = math.log(S0) + (r-q + math.log(1-theta*nu-0.5*sigma**2*nu)/nu)*T
            phi = cmath.exp(1j*u*mu)*((1-1j*nu*theta*u+0.5*nu*sigma**2*u**2)**(-T/nu)) # Equivalent to CF VG in the Book
            
    elif (model == 'VGSA'):
        sigma  = params[0]
        nu     = params[1]
        theta  = params[2]
        kappa  = params[3]
        eta    = params[4]
        lda    = params[5]
        
        def log_phi(u1, y0):
            gamma1 = np.sqrt(kappa**2 - 2*lda**2*1j*u1)
            logA = kappa**2*eta*T/lda**2 - (2*kappa*eta/lda**2) * np.log(np.cosh(gamma1*T/2)+(kappa/gamma1)*np.sinh(gamma1*T/2))
            B = 2*1j*u1/(kappa+gamma1*coth(gamma1*T/2))
            return logA + B*y0

        def psi(u2):
            #print(u2)
            """log CF"""
            return -(1./nu)*np.log(1-1j*u2*theta*nu+sigma**2*nu*u2**2/2)

        log_asset = 1j*u*(np.log(S0)+(r-q)*T) + log_phi(-1j*psi(u), 1/nu) - (1j*u) * log_phi(-1j*psi(np.repeat(-1j, len(u))), 1/nu)
        phi = np.exp(log_asset) 

    elif (model == 'VGSSD'):
        sigma  = params[0]
        nu     = params[1]
        theta  = params[2]
        kappa  = params[3]
        eta    = params[4]
        lda    = params[5]
        gamma  = params[6]

        def log_phi_VG(u1):
            #log CF VG
            return -(1./nu)*np.log(1-1j*u1*theta*nu+sigma**2*nu*u1**2/2)

        log_asset = 1j*u*(np.log(S0)+(r-q)*T) + log_phi_VG(u*T**gamma) - log_phi_VG(-1j*T**gamma)
        phi = np.exp(log_asset)
        
    return phi

def genericFFT(params, S0, K, r, q, T, alpha, eta, n, model):
    
    N = 2**n
    
    # step-size in log strike space
    lda = (2*np.pi/N)/eta
    
    #Choice of beta
    #beta = np.log(S0)-N*lda/2
    beta = np.log(K)
    
    # forming vector x and strikes km for m=1,...,N
    km = np.zeros((N))
    xX = np.zeros((N))
    
    # discount factor
    df = math.exp(-r*T)
    
    nuJ = np.arange(N)*eta  # This is an array -> CF needs to be able to take in an array & output one
    psi_nuJ = generic_CF(nuJ-(alpha+1)*1j, params, S0, r, q, T, model)/((alpha + 1j*nuJ)*(alpha+1+1j*nuJ))
    
    for j in range(N):  
        km[j] = beta+j*lda
        if j == 0:
            wJ = (eta/2)
        else:
            wJ = eta
        xX[j] = cmath.exp(-1j*beta*nuJ[j])*df*psi_nuJ[j]*wJ
     
    yY = np.fft.fft(xX)
    cT_km = np.zeros((N))  
    for i in range(N):
        multiplier = math.exp(-alpha*km[i])/math.pi
        cT_km[i] = multiplier*np.real(yY[i])
    
    return km, cT_km
                        
                        
    