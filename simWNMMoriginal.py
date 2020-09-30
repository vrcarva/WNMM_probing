# -*- coding: utf-8 -*-
"""
Simulates original Wendling's CA1 neural mass model
"""
import numpy as np
import matplotlib.pyplot as plt

def OriginalWendlingNMM(y, h, P ):
    #Original Wendling's CA1 neural mass model
    noise =  np.random.normal(P["meanP"], P["sigmaP"])#
    #noise = P["coefMultP"]*np.random.normal(0, P["sigmaP"])*np.sqrt(h) + P["coefMultP"]*P["meanP"] # np.sqrt(dt) para compensar diferentes steps de integração
    dydx = np.zeros(10)
    #derivs
    dydx[0] = y[5]
    dydx[5] = P["A"] * P["a"] * sigm(y[1]-y[2]-y[3],P) - 2. * P["a"] * y[5] - P["a"] * P["a"] * y[0]
    dydx[1] = y[6]
    dydx[6] = P["A"] * P["a"] * (noise + P["C2"] * sigm(P["C1"]* y[0] + P["SG"],P) ) - 2. * P["a"] * y[6] - P["a"]*P["a"] * y[1]
    dydx[2] = y[7];
    dydx[7] = P["B"] * P["b"] * ( P["C4"] * sigm(P["C3"] * y[0],P) ) - 2. * P["b"] * y[7] - P["b"]*P["b"] * y[2]
    dydx[3] = y[8];
    dydx[8] = P["G"] * P["g"] * ( P["C7"] * sigm((P["C5"] * y[0] - P["C6"] * y[4]),P) ) - 2. * P["g"] * y[8] - P["g"]*P["g"] * y[3]
    dydx[4] = y[9]
    dydx[9] = P["B"] * P["b"] * ( sigm(P["C3"] * y[0],P) ) - 2. * P["b"] * y[9] - P["b"]*P["b"] * y[4]  

    yout = np.empty((0))
    for i in range(10):
        yout = np.append(yout,y[i]  + h *dydx[i])
    return yout

def sigm(v,P):#sigmoid
    return 2.*P["e0"]/(1.+np.exp(P["r"]*(P["v0"]-v))) 
# %%

#simulation parameters 
finalTime=15#simulation time
Fs = 512 #sampling frequency (Hz)
A = 4
B = 40
G = 20

#model parameters
P = {"A":A,"B":B,"G":G,"a":100.,"b":50.,"g":350.,"v0":6.,"e0":2.5,"r":0.56,
     "C1":1.,"C2":0.8,"C3":0.25,"C4":0.25,"C5":0.3,"C6":0.1,"C7":0.8,"C":135.}    
P["C1"] *= P["C"]; P["C2"] *= P["C"]; P["C3"] *= P["C"]; P["C4"] *= P["C"]
P["C5"] *= P["C"]; P["C6"] *= P["C"]; P["C7"] *= P["C"];
P["SG"] = 1. #stimulus (synaptic) gain 
P["meanP"] = 90. # Input noise parameters
P["sigmaP"] = 30.
dt = 1./Fs # period
nb_fonc = 10 # Number of ODEs

nbSamples = int(finalTime / dt) # number of samples

simulatedLFP = np.zeros(nbSamples)
yold = np.zeros(nb_fonc)
xstates = np.zeros([nb_fonc,nbSamples])
tvec = np.zeros(nbSamples)
t = 0.
for tt in range(nbSamples):
    ynew = OriginalWendlingNMM(yold,dt,P)
    yold = ynew
    tvec[tt] = t
    t += dt
    xstates[:,tt] = ynew
    simulatedLFP[tt] = ynew[1]-ynew[2]-ynew[3]   

plt.figure()
plt.plot(simulatedLFP)
plt.xlabel('s')
plt.ylabel('a.u.')
plt.title('Simulated LFP')