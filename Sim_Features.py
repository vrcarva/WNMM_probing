# -*- coding: utf-8 -*-
"""

Simulates coupled Wendling neural mass models with and without periodic stimuli to main EXC cells,
changing parameters (EXC/A, SDI/B or coupling gain/K) to bring population activity towards the ictal state. 
Features are extracted throughout each simulation - trends in the resulting feature seried could
result in seizure predictors. 

Block 1 simulates several instances of a specific model configuration (I-A,I-B,I-K, II-A,II-B,II-K)
with increasing stimulus amplitude and extracts features from each one. 

Block 2 calculates correlation measures between the resulting feature series (variance, skewness, etc)
and the shifted parameter. If correlation is high, feature increase/decrease may signal that the 
model is approaching a seizure.  Output variables are saved - features as .npy and correlation measures as .pkl

Block 3 simulates an instance of the model to generate Figure 2 from the paper - with and withou stimulation
of neuronal population 1, while linearly increasing A1 to elicit ictal activity in the end.

@author: Vinícius Rezende Carvalho
vrcarva@ufmg.br

"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import kurtosis, skew, spearmanr
from scipy import signal
import pandas as pd
from joblib import Parallel, delayed
import time
from scipy.ndimage.filters import uniform_filter1d
from sklearn.feature_selection import mutual_info_regression
from itertools import groupby
#from statsmodels.stats.multicomp import pairwise_tukeyhsd
#from statsmodels.stats.multicomp import MultiComparison
#from scipy.ndimage import median_filter
#from mpl_toolkits.axes_grid1 import make_axes_locatable
# Model simulation
class CA1simulation():
    """ Model simulation.
    Pops: CA1popSet neuronal population instance
    Fs: Sampling frequency (Hz). Default = 512 Hz
    finalTime: Simulation time (s). Default = 100s
    stimPeriod: Stimulation period (s). Default = 2
    stimDuration: Stimulus duration (s). Default 0.01
    stimAmp: Stimulation amplitude. Default = 4
    stimPos: Stimulation input position ("DG" for input at main EXC cells or "all" to apply at membrane potential of all sub-populations)
    biphasic: 1/true for biphasic stimuli, 0/false for monophasic
    """ 
    def __init__(self,Pops,Fs = 512, finalTime=100.,stimPeriod = 2,stimDuration = 0.01,stimAmp = 4,stimPos = "DG",biphasic = 0 ):

        #Initialtisation of the Vector of parameters
        self.finalTime = finalTime #simulation time, in seconds
        #Simulation parameters
        self.Fs = Fs #user-defined (typical: 512 Hz)
        self.stimDuration = stimDuration #stimulus duration (seconds)
        self.stimPeriod = stimPeriod # stimulus Period  (in s)
        self.stimAmp = stimAmp #stimulus amplitude 
        self.Pops = Pops
        self.nbSamples = int(self.finalTime*Fs) # number of samples
        
        #For changes in parameters across the simulation - edit these fields after defining the class
        self.As = np.array([Pops.P[fi]["A"]*np.ones(self.nbSamples) for fi in range(len(Pops.P))])
        self.Bs = np.array([Pops.P[fi]["B"]*np.ones(self.nbSamples) for fi in range(len(Pops.P))])
        self.Gs = np.array([Pops.P[fi]["G"]*np.ones(self.nbSamples) for fi in range(len(Pops.P))])
        self.Ks = np.array([Pops.P[fi]["K"]*np.ones(self.nbSamples) for fi in range(len(Pops.P))])

        #Input stimulus
        stimIndxs = [np.arange(int(Fs*stimPeriod)+i,self.nbSamples-i,int(Fs*stimPeriod)) for i in range(int(Fs*stimDuration))]
        stimIndxsNegative = [np.arange(int(Fs*stimPeriod)+int(stimDuration*Fs)+i,self.nbSamples-i,int(Fs*stimPeriod)) for i in range(int(Fs*stimDuration))]
        #[one for each model set] [time samples, 0 for intra-hip stim/ 1 for DG stim]
        uinput = [np.zeros([2,self.nbSamples]) for ui in range(Pops.Nsets)]
        
        if stimPos == "DG":#stimulates model input - APs to main excitatory cells
            uIdx = 1
        else: #stimulates all cells
            uIdx = 0
        for ui in range(Pops.Nsets):
            uinput[ui][uIdx,stimIndxs] = stimAmp
            if biphasic:
                uinput[ui][uIdx,stimIndxsNegative] = -stimAmp#biphasic pulse
        self.stimInput = uinput
           
    def simulateLFP(self,highpass = 1):
        #Simulate model
        nb_fonc = 10 # Number of ODEs
        dt = 1./self.Fs # step size
        Nmodels = self.Pops.Nsets#number of (coupled) model sets
        
        #pyr state output (for lagged inter coupling)
        kTemp = np.zeros([Nmodels, max(1,int(0.001*self.Pops.P[0]["cLag"]*self.Fs))]) 
        
        #new simulation - initialize variables
        LFPout = np.zeros([Nmodels,self.nbSamples])
        tvec = np.zeros(self.nbSamples)#time vector
        t = 0.
        yold = np.zeros([Nmodels,nb_fonc])               
        
        for tt in range(self.nbSamples):
            for ch in range(Nmodels):#for each population set
                #resets coupling values
                couplInputIntra = 0.
                couplInputInter = 0.
                #parameters for current time sample
                self.Pops.P[ch]["A"] = self.As[ch,tt]
                self.Pops.P[ch]["B"] = self.Bs[ch,tt]
                self.Pops.P[ch]["G"] = self.Gs[ch,tt]
                self.Pops.P[ch]["K"] = self.Ks[ch,tt]
    
                if self.Pops.coupling == "inter": #inter-hemisphere coupling - bidirectional, between pyramidal (EXC) cells
                    if (ch-1 >=0):
                        couplInputInter = couplInputInter + self.Pops.P[ch]["K"]*kTemp[(ch-1),-1]#lagged input from another pop set
                    if (ch+1 < Nmodels): #coupling is bidirectional
                        couplInputInter =  couplInputInter + self.Pops.P[ch]["K"]*kTemp[(ch+1),-1]                  
                elif self.Pops.coupling == "intra":#intra-hemisphere coupling - bidirectional, between basket (SDI) cells
                    if (ch-1 >=0):
                        couplInputIntra =  couplInputIntra + self.Pops.P[ch]["K"]*yold[(ch-1),3]  
                    if (ch+1 < Nmodels):#bidirectional coupling
                        couplInputIntra =  couplInputIntra + self.Pops.P[ch]["K"]*yold[(ch+1),3]
    
                ynew = wendlingModel_2(yold[ch,:],self.Pops.P[ch],dt, self.stimInput[ch][:,tt], couplInputIntra, couplInputInter )
                yold[ch,:] = ynew
                LFPout[ch,tt] = ynew[1]-ynew[2]-ynew[3]#model output
                kTemp[ch,0] = ynew[1]
            kTemp = np.roll(kTemp, 1, axis = 1)#shifts y[1]
            tvec[tt] = t
            t += dt 
            
        #detrend    
        LFPout = signal.detrend(LFPout)
          
        if highpass == 1:#highpass filter
            b, a = signal.butter(3, 0.2*2/self.Fs, 'high')   
            for chi in range(Nmodels):
                print(chi)
                LFPout[chi,:] = signal.filtfilt(b, a, LFPout[chi,:])
        return LFPout
    
class CA1popSet():
    """Wendling neural mass model population sets
    Nsets: defines the number of populations or model subsets. Default = 2
    coupling: defines how these are interconected 
        "inter" for coupling through main exc cells (inter-hemisphere)
        "intra" for coupling through basket cells (intra-hippocampal)
    A: main cells excitatory gain
    B: Slow inhibitory gain
    G: Fast inhibitory gain
    K: Coupling gain
    pSTD: input noise standard deviation
    pMean: input noise mean
    clag: Lag (in ms) between coupled neuronal populations
    """
    from itertools import combinations

    def __init__(self,Nsets = 2,coupling = "inter",A = 4., B = 40.,G = 20.,K = 0,pSTD = 1.3, pMean = 90., clag = 10):
        P = {"A":A,"B":B,"G":G,"K":K,"a":100.,"b":50.,"g":350.,"v0":6.,"e0":2.5,"r":0.56,"sigmaP":pSTD,"meanP":pMean,
                          "C1":1.,"C2":0.8,"C3":0.25,"C4":0.25,"C5":0.1,"C6":0.1,"C7":0.8,"C":135.,"cLag":clag}
        P["C1"] *= P["C"]; P["C2"] *= P["C"]; P["C3"] *= P["C"]; P["C4"] *= P["C"]
        P["C5"] *= P["C"]; P["C6"] *= P["C"]; P["C7"] *= P["C"];    
        self.P = [None]*Nsets
        for pi in range(Nsets):
            self.P[pi] = P
            self.coupling = coupling    
            self.Nsets = Nsets

def wendlingModel_2(y, P, h = 1./512, u = np.zeros(1), couplIntra = 0, couplInter = 0 ):
    """ Modified Wendling's neural mass model ODEs 
    y: last state
    P: model parameters
    h: step (1/Fs)
    u: stimulation input - 2 elements u[0] voltage deflection to all cells. u[1] sums the stimulus to the model input p(t) to main excitatory cells
    couplIntra: coupling between basket cells (intra-hipp coupling)
    couplInter: coupling between pyramidal cells (inter-hemisphere coupling)
    """
   
    noise = np.random.normal(0, P["sigmaP"])#using Euler-maruyama, so we multiply this by sqrt(h) below (see Hebbink,2014)
    yNew = np.zeros(10)
    yNew[0] = y[0] + y[5] * h
    yNew[5] = y[5] + (P["A"] * P["a"] * sigm(P["K"]*couplInter+u[0]+y[1]-y[2]-y[3],P) - 2. * P["a"] * y[5] - P["a"]*P["a"] * y[0]) * h
    yNew[1] = y[1] + y[6] * h
    yNew[6] = y[6] + (np.sqrt(h)*noise*P["A"]*P["a"]) + (P["A"] * P["a"] * (u[1]+P["meanP"] + P["C2"] * sigm(P["C1"]* y[0] + u[0],P)) - 2. * P["a"] * y[6] - P["a"]*P["a"] * y[1]) * h
    yNew[2] = y[2] + y[7] * h
    yNew[7] = y[7] + (P["B"] * P["b"] * (P["C4"] * sigm(P["C3"] * y[0] + u[0],P) ) - 2. * P["b"] * y[7] - P["b"]*P["b"] * y[2]) * h
    yNew[3] = y[3] + y[8] * h
    yNew[8] = y[8] + (P["G"] * P["g"] * (P["C7"] * sigm((u[0] + P["C5"] * y[0] - P["C6"] * y[4] - P["K"]*couplIntra),P) ) - 2. * P["g"] * y[8] - P["g"]*P["g"] * y[3]) * h
    yNew[4] = y[4] + y[9] * h
    yNew[9] = y[9] + (P["B"] * P["b"]* ( sigm(P["C3"] * y[0] + u[0],P) ) - 2 * P["b"] * y[9] - P["b"]*P["b"] * y[4]) * h
    
    return yNew
    
def sigm(v,P):#sigmoid
    return 2.*P["e0"]/(1.+np.exp(P["r"]*(P["v0"]-v))) 

def fun_extractERPfeatsUni(erp,preERP,Fs):
    """ fun_extractERPfeatsUni(erp,preERP,Fs)
    Extracts univariate features from epoch
    Inputs:
        erp is the epoch (single channel) or segment to extract features from
        preERP is the baseline signal
        Fs is sampling frequency, in Hz
    Output:
        featsOut with fields: "normEnergy","Energy","Var","Skew","Kurt","Hmob","Kcomp","LLen","LLenNorm","SpectCent" and "lag1AC"
    """

    featsOut = dict()
    #FEATURE EXTRACTION
    #Normalized energy: postStim/preStim
    featsOut["normEnergy"] = np.mean(erp**2) \
    /np.mean(preERP**2)
    featsOut["Energy"] = np.mean(erp**2) 
    # Statistical Moments
    featsOut["Var"] =  np.var(erp)
    featsOut["Skew"] =  skew(erp)
    featsOut["Kurt"] =  kurtosis(erp)
    dyTrecho = np.diff(erp)
    dyBasal = np.diff(preERP)
    #hmobBasal = np.sqrt(np.var(dyBasal)/np.var(simulatedLFP[mi,stimTS[si]-int(tprePEARP*Fs):stimTS[si]]))
    #hcompBasal =  np.sqrt(np.var(np.diff(dyBasal))/np.var(dyBasal))/hmobBasal
    featsOut["Hmob"] = np.sqrt(np.var(dyTrecho)/featsOut["Var"])
    featsOut["Hcomp"] =  np.sqrt(np.var(np.diff(dyTrecho))/np.var(dyTrecho))/featsOut["Hmob"]
    featsOut["LLen"] = np.sum(np.abs(dyTrecho))#line length
    featsOut["LLenNorm"] = featsOut["LLen"]/np.sum(np.abs(dyBasal))#normalized line length
    
    #spectral centroid
    yFFT,wFreq = fftex(erp-np.mean(erp),fs = 512,plotFig = 0)
    yFFT = np.abs(yFFT)/np.sum(np.abs(yFFT))
    featsOut["SpectCent"] = np.sum(wFreq*yFFT)
    #featsOut["Hmob"] = featsOut["Hmob"]/hmobBasal
    #featsOut["Hcomp"] = featsOut["Hcomp"]/hcompBasal
    
    #lag 1 autocorrelation
    featsOut["lag1AC"]= np.corrcoef(erp[:-1], erp[1:])[0,1]
    
    return featsOut

def fun_extractERPfeatsMultivar(erpSynch,erpSynchFilt,Fs):
    """fun_extractERPfeatsMultivar(erpSynch,erpSynchFilt,Fs)
    extract multivariate features from L channels
    #erpSynch is a LxN matrix --> L channels with N samples each, from which synchrony measures are taken (PLV and correlation)
    #detrend and normalize erpSynch (fucks up PLV values in some cases if it's not detrended)
    #erpSynchFilt is the filtered version of erpSynch, for calculating the PLV
    """
    from itertools import combinations
    from sklearn.feature_selection import mutual_info_regression
    erpSynch = (erpSynch.T - np.mean(erpSynch,axis = 1)).T
    erpSynch = (erpSynch.T/np.std(erpSynch,axis = 1)).T
    erpSynchFilt = (erpSynchFilt.T - np.mean(erpSynchFilt,axis = 1)).T
    featsOut = dict()
    #for all channel pair combinations
    featsOut["combinations"] = list(combinations(range(erpSynch.shape[0]), 2))
    #featsOut["Corr"] = np.zeros(len(featsOut["combinations"]),erpSynch.shape[1])
    #featsOut["PLV"] = np.zeros(len(featsOut["combinations"]),erpSynch.shape[1])
    #*** Coupling Measures ***
    #correlation  (max value)
    featsOut["Corr"] = [np.max(signal.correlate(erpSynch[ki[0],:],
             erpSynch[ki[1],:]))/erpSynch.shape[1] for ki in featsOut["combinations"]]
    
    featsOut["CorrCoefs"] = np.corrcoef(erpSynch)
    
    featsOut["MI"] = [mutual_info_regression(erpSynch[ki[0],:].reshape(-1, 1),erpSynch[ki[1],:]) for ki in featsOut["combinations"]]

    #PLV 
    phases = np.array([np.angle(signal.hilbert(erpSynchFilt[ki,:])) for ki in range(erpSynchFilt.shape[0])])
    featsOut["PLV"] = [np.abs(np.sum(np.exp(1j*(phases[ki[0]]-phases[ki[1]])))/
            phases.shape[1]) for ki in featsOut["combinations"]]
    featsOut["PLVphase"] =  [np.mean(np.unwrap(phases[ki[0]]-phases[ki[1]],axis = 0)) for ki in featsOut["combinations"]]
    
    #Coherence
    featsOut["Coh"] = np.zeros([len(featsOut["combinations"])])
    iind = 0
    for ki in featsOut["combinations"]:
        Wxy, Cxy = signal.coherence(erpSynch[ki[0],:], erpSynch[ki[1],:], Fs, nperseg = 128)
        featsOut["Coh"][iind] = np.mean(Cxy[0:11])
        iind+=1
    
    return featsOut

#One-sided fast Fourier transform
def fftex(f, fs = 1, plotFig = 1):
    """
    YFFT,Wfreq = fftex(f, fs = 1, plotFig = 1):
    # f: input signal
    # fs: sampling frequency
    # plotFig: 1 to plot FFT in new figure
    """
    yf = np.fft.fft(f)
    Nf = yf.size
    freq = np.fft.fftfreq(Nf, d=1/fs)
    if plotFig:
        plt.figure()
        plt.plot(freq[:Nf//2], 2.0/Nf*np.abs(yf[:Nf//2]))
    return yf[:Nf//2],freq[:Nf//2]


def SimFeats(simLFP,Nmodels,promedia,stimTS,Fs,simLFPmean = 0):
    """(Feats, FeatsMulti) = SimFeats(simLFP,simLFPmean,Nmodels,promedia,stimTS,Fs)
    Extracts features from each simulation
    """
    #design filters
    bLP, aLP = signal.butter(3, 20*2/Fs)        #lowpass   
    simulatedLFPFilt = np.zeros(simLFP.shape)
    #std of initial segment - used as a reference for identifying ictal periods
    baselineSTDs = np.std(simLFP[:,:int(50*Fs)], axis = 1) 
    #from each simulation, extracts feature sets
    #feature initializing
    if Nmodels >1:   #only for simulations with coupled models - coupling measures
        NSynch = Nmodels*(Nmodels-1)//2
        FeatsMulti = {"PLV":np.ndarray(shape = [len(stimTS),NSynch]),
                      "Corr":np.ndarray(shape = [len(stimTS),NSynch]),
                      "PLVphase":np.ndarray(shape = [len(stimTS),NSynch]),
                      "Coh":np.ndarray(shape = [len(stimTS),NSynch]),
                      "MI":np.ndarray(shape = [len(stimTS),NSynch])}  
        diInit = range(Nmodels+1)

    else:
        diInit = range(Nmodels)
        FeatsMulti = 0
    #initialize features
    Feats = [{"normEnergy":np.zeros([len(stimTS)]),"Var":np.zeros([len(stimTS)]),
             "Skew":np.zeros([len(stimTS)]),"Kurt":np.zeros([len(stimTS)]),
             "Hmob":np.zeros([len(stimTS)]),"Hcomp":np.zeros([len(stimTS)]),
             "PkAmp":np.zeros([len(stimTS)]),"Energy":np.zeros([len(stimTS)]),
             "ValeAmp":np.zeros([len(stimTS)]),"PkLag":np.zeros([len(stimTS)]),
             "ValeLag":np.zeros([len(stimTS)]),"lag1AC":np.zeros([len(stimTS)]),
             "LLen":np.zeros([len(stimTS)]),"LLenNorm":np.zeros([len(stimTS)]),
             "SpectCent":np.zeros([len(stimTS)]), 
             "Ictal":np.zeros([len(stimTS)]),"IctalOnset":0} for di in diInit]
    #stores all Evoked response potentials - allPEARPS and allPEARPS_2
    allPEARPS = np.ndarray(shape=(len(stimTS),int((tprePEARP)*Fs)+int((tposPEARP)*Fs),Nmodels))#(Stims,timesamples,channel)
    if promedia == "erps": #
        allPEARPS_2 = np.zeros([len(stimTS),int((tprePEARP)*Fs)+int((tposPEARP)*Fs)])#  for mean LFP (only for Nmodels >2 and promedia == "erps")
        allPEARPS_2_filt = np.ndarray(shape=(len(stimTS),int((tprePEARP)*Fs)+int((tposPEARP)*Fs),Nmodels))#(Stims,timesamples,channel) same, but lowpass filtered 
    #tpearp = np.linspace(0,allPEARPS.shape[1]/Fs,allPEARPS.shape[1])

    for chi in range(Nmodels):
        simulatedLFPFilt[chi,:] = signal.filtfilt(bLP, aLP, simLFP[chi,:])
        #seizure onset time - find when discharges begin to repeat with intervals <1.5s
        indsRemove = []#remove peaks near stimuli indexes
        indsRemove = np.append(indsRemove, [np.arange(stemp,stemp+int(0.2*Fs)) for stemp in stimTS])#exclude peaks due to stimulation?
        pk = signal.find_peaks(simLFP[chi,:],height=5)
        pk = pk[0]
        pk = np.delete(pk,np.where(np.isin(pk,indsRemove))[0])
       
        if pk.size > 10:  
            pkDif = np.diff(pk)
            gbPks = [(list(v)) for g,v in groupby(pkDif<=4*Fs)]
            gbLen = np.array([len(ii) for ii in gbPks])
            gbWhere = np.where (np.array([len(ii)>=4 for ii in gbPks]) & np.array([gg[0] for gg in gbPks]))[0][0]
            idxOnset = np.sum(gbLen[:gbWhere])

            if idxOnset.size>0:
                Feats[chi]["IctalOnset"] = pk[idxOnset]
            else:
                Feats[chi]["IctalOnset"]= np.NaN
        else:
            Feats[chi]["IctalOnset"]= np.NaN

            
    #ERP features
    for si in range(len(stimTS)):
        indsERP = np.arange(stimTS[si],stimTS[si]+int(tposPEARP*Fs))#ERP indexes (only for "feature" averaging)
        indsERP2 = range(max(0,si-avrgWinSize+1),si+1) #indexes - which ERPs to average (relative to allPEARPS) - mean ERP of last "avrgWinSize" stimuli
        ERP = np.zeros([len(indsERP),Nmodels])
        for chi in range(Nmodels):#for each channel/model
            
            allPEARPS[si,:,chi] = simLFP[chi,stimTS[si]-int(tprePEARP*Fs):stimTS[si]+int(tposPEARP*Fs)]
            if promedia == "features": #smooths through "features"
                ERP[:,chi] = simLFP[chi,indsERP]
                preerp = simLFP[chi,stimTS[si]-int(tprePEARP*Fs):stimTS[si]]
            else: #ERP smoothing
                ERP[:,chi] = np.mean(allPEARPS[indsERP2,int(tprePEARP*Fs):,chi],axis = 0)
                preerp = np.mean(allPEARPS[indsERP2,0:int(tprePEARP*Fs)-1,chi],axis = 0)
                allPEARPS_2_filt[si,:,chi] = simulatedLFPFilt[chi,stimTS[si]-int(tprePEARP*Fs):stimTS[si]+int(tposPEARP*Fs)]
            #FEATURE EXTRACTION
            #Normalized energy: postStim/preStim
            featsTempUNI = fun_extractERPfeatsUni(ERP[:,chi],preerp,Fs)
            #organize feature variable
            for fKey in featsTempUNI.keys():
                Feats[chi][fKey][si] = featsTempUNI[fKey]  
            #seizure detection
            Feats[chi]["Ictal"][si] = ERP[abs(ERP)>=3*baselineSTDs[chi]].size/ERP.size
        if Nmodels >1:
            #UNIVARIATE features from mean(LFP) (LAST element from Feats!)
            if promedia == "features": #smooths through "features"
                ERPmean = simLFPmean[indsERP]
                preMeanerp = simLFPmean[stimTS[si]-int(tprePEARP*Fs):stimTS[si]]
                ERPsFilt = simulatedLFPFilt[:,indsERP] #with filtered signal
            else: #ERP smoothing
                allPEARPS_2[si,:] = simLFPmean[stimTS[si]-int(tprePEARP*Fs):stimTS[si]+int(tposPEARP*Fs)]
                ERPmean = np.mean(allPEARPS_2[indsERP2,int(tprePEARP*Fs):],axis = 0)
                preMeanerp = np.mean(allPEARPS_2[indsERP2,0:int(tprePEARP*Fs)-1],axis = 0)
                ERPsFilt = np.mean(allPEARPS_2_filt[indsERP2,int(tprePEARP*Fs):,:],axis = 0)

            featsTempUNI2 = fun_extractERPfeatsUni(ERPmean,preMeanerp,Fs)
            for fKey in featsTempUNI2.keys():#mean LFP features
                Feats[-1][fKey][si] = featsTempUNI2[fKey]

            #COUPLING FEATURES
            featsTempMULTI = fun_extractERPfeatsMultivar(ERP.T,ERPsFilt,Fs)  
            FeatsMulti["PLV"][si,:] = featsTempMULTI["PLV"]
            FeatsMulti["Corr"][si,:] = [featsTempMULTI["CorrCoefs"][ix] for ix in  featsTempMULTI["combinations"] ]
            FeatsMulti["PLVphase"][si,:] = featsTempMULTI["PLVphase"]
            FeatsMulti["Coh"][si,:] = featsTempMULTI["Coh"]
            FeatsMulti["MI"][si,:] = featsTempMULTI["MI"]
    if Nmodels >1:
        FeatsMulti["SynchPairs"] = featsTempMULTI["combinations"]     
    return (Feats, FeatsMulti)
    
def CorrMeasures(xplot,Feats,plotFeatures,plotFeaturesSynch,chSet = 0,Nmodels = 1, FeatsMulti = 0):
    #correlation measures between features (Feats and FeatsMulti) and shifted parameter (xplot)
    MeasMI = [None]*len(stimAmp) #mutual information [StimAmps][Realizations,features]
    MeasCorr = [None]*len(stimAmp) 
    MeasSpCorr = [None]*len(stimAmp) 
    
    for ui in range(len(stimAmp)): #for each stimulus type 
        MeasMI[ui],MeasSpCorr[ui],MeasCorr[ui] = {},{},{}
        for fkey in plotFeatures:#for each univariate feature
            ysmooth = uniform_filter1d(Feats[ui][chSet][fkey], size=avrgWinSize,axis = 0)#smooth feature series
            MeasMI[ui][fkey] =        mutual_info_regression(ysmooth, xplot[stimTS]) #mutual information
            MeasCorr[ui][fkey] =      np.corrcoef(ysmooth.T,xplot[stimTS])[-1,:-1]#correlation coefficient
            spTemp = spearmanr(ysmooth,xplot[stimTS])[0]#spearman correlation coefficient
            if Nsims>1:
                spTemp = spTemp[-1,:-1]
            MeasSpCorr[ui][fkey] =  spTemp
        if Nmodels>1:#synchronization features
            for fkey in plotFeaturesSynch:
                ysmooth = uniform_filter1d(FeatsMulti[ui][fkey][:,:,0], size=avrgWinSize,axis = 0)
                MeasMI[ui][fkey] =     mutual_info_regression(ysmooth, xplot[stimTS-1])
                MeasCorr[ui][fkey] =   np.corrcoef(ysmooth.T,xplot[stimTS])[-1,:-1]
                spTemp = spearmanr(ysmooth,xplot[stimTS])[0]
                if Nsims>1:
                    spTemp = spTemp[-1,:-1]
                MeasSpCorr[ui][fkey] =  spTemp                
                
        MeasMI[ui]["StimAmp"] = stimAmp[ui]*np.ones(Nsims)
        MeasCorr[ui]["StimAmp"] = stimAmp[ui]*np.ones(Nsims)
        MeasSpCorr[ui]["StimAmp"] = stimAmp[ui]*np.ones(Nsims)
        
        MeasMI[ui]["channel"] = chSet
        MeasCorr[ui]["channel"] = chSet
        MeasSpCorr[ui]["channel"] = chSet
        
        #seizure onset time 
        MeasMI[ui] = pd.DataFrame(MeasMI[ui])
        MeasCorr[ui] = pd.DataFrame(MeasCorr[ui])
        MeasSpCorr[ui] = pd.DataFrame(MeasSpCorr[ui])
    
    MeasMI_df = pd.concat([ii for ii in MeasMI], ignore_index=True)
    MeasSpCorr_df = pd.concat([ii for ii in MeasSpCorr], ignore_index=True)  
    MeasCorr_df = pd.concat([ii for ii in MeasCorr], ignore_index=True) 
    
    return MeasMI_df, MeasSpCorr_df, MeasCorr_df
      

#%% create and simulate models

#sim parameters (default otherwise)
stimAmp = np.linspace(0,200,11) #stimulus amplitudes - One value for each simulation
#stimAmp = np.array([0])
Nmodels = 2 #number of population subsets

#Feature parameters
tprePEARP = 0.4 # pre-stimulus period (baseline for feature extraction)
tposPEARP = 0.4 # post-stimulus period for feature extraction?
promedia = "features" #smoothing features or ERPs: "features" or "erps" 
avrgWinSize = 20 #moving average window size

Nsims = 10 #number of realizations/simulations for each model setting
xvar = 'A' #which parameter is varied (for plotting) ('A','B' or 'K')
configProb = "I" #'I' to stimulate population 2 and "II" to stimulate both population sets

Fs = 512.#sampling rate
#create model
Pops = CA1popSet(coupling = "inter", #"inter" for inter-hemisphere (between PYR cells) or "intra" for intra-hemisphere coupling (between basket cells)
                 Nsets = Nmodels, #number of coupled model/population subsets 
                 A = 4.,# EXC
                 B = 40,#SDI
                 G = 20,#FSI
                 K = 0.3)#coupling factor

PopsModel = [None]*len(stimAmp)
Feats = [None]*len(stimAmp) #[stimsAmp][ModelSubsets][Features][stim/time,realization]
FeatsMulti = [None]*len(stimAmp) #[stimsAmp][Features][stim/time,realization,Channel combination]

for si,stimTemp in enumerate(stimAmp):# for each stimulus parameter
    PopsModel[si] = CA1simulation(Pops, #populations to simulate
                                 Fs = Fs, #user-defined sampling frequency (typical: 512 Hz)
                                 finalTime=2000.,#total simulation time, in seconds
                                 stimAmp = stimTemp,#stimulus amplitude
                                 stimPeriod = 2.,#stimulus period
                                 stimDuration = 0.01,#stimulus duration (seconds)
                                 stimPos = "DG",#"DG" (sum stimulus to noise input to main cells - simulates stimulus to the DG) or "all" (sum stimulus to PSPs of all cells)
                                 biphasic = 0)# #biphasic (1) or monophasic (0) stimulation pulse 
    
    #shift parameters towards ictal activity of a specific population set
    if xvar == "A":
        PopsModel[si].As[0,:] = np.linspace(2.5,4.6, PopsModel[si].nbSamples)
    if xvar == "B":
        PopsModel[si].Bs[0,:] = np.linspace(45., 30., PopsModel[si].nbSamples)
    if xvar == "K":
        PopsModel[si].Ks[:,:] = np.linspace(0., 0.5, PopsModel[si].nbSamples)
    
    if configProb == "I":#configuration I stimulates only population set 2
        PopsModel[si].stimInput[0][:,:] = 0#No stimulus ipsilateral to ictal onset
    
    #stimuli timestamps       
    stimTS = np.arange(int(PopsModel[si].Fs*PopsModel[si].stimPeriod),PopsModel[si].nbSamples,int(PopsModel[si].Fs*PopsModel[si].stimPeriod))    
    tic = time.time()
    #Simulate Nsims instances
    simulatedLFP = Parallel(n_jobs=12,max_nbytes=None)(
            delayed(PopsModel[si].simulateLFP)(highpass = 1)
            for mi in range(Nsims)) 
    print (time.time()-tic)
    simulatedLFP = np.array(simulatedLFP)
    tvec = np.arange(0,PopsModel[si].finalTime,1/PopsModel[si].Fs)
    #plot simulated LFP
    #plt.figure()
    #plt.plot(tvec,simulatedLFP[0,:,:].T)
    
    if Nmodels >1: #if coupled model subsets are used
        meanLFP = np.zeros([Nsims,PopsModel[si].nbSamples])#mean LFP of all channels/models
        meanLFP = np.mean(simulatedLFP,axis = 1).T 
    else: 
        meanLFP = np.zeros([1,Nsims])
    #Feats[si], FeatsMulti[si] = featFromSim(simulatedLFP, Nmodels, Nsims,promedia,stimTS,PopsModel[si].Fs) # extract features - no parfor
    #extract features
    tic = time.time()
    FeatTuple = Parallel(n_jobs=10,max_nbytes=None)(
            delayed(SimFeats)(simulatedLFP[mi,:,:],Nmodels,promedia,stimTS,PopsModel[si].Fs,meanLFP[:,mi]) for mi in range(Nsims))
    #first element - univariate features. Second element = multivariate/synchrony features
    print (time.time()-tic)
    
    #organize univariate features [stimsAmp][ModelSubsets][Features][stim/time,realization]
    Feats[si] = [None]*len(FeatTuple[0][0])
    for chi in range(len(FeatTuple[0][0])):
        featemp = FeatTuple[0][0][0]
        Feats[si][chi] = dict.fromkeys(featemp.keys(), None)
        for fkey in featemp.keys():
            if fkey == 'IctalOnset':
                Feats[si][chi][fkey] = np.array([FeatTuple[mi][0][chi][fkey] for mi in range(Nsims)])
            else:       
                Feats[si][chi][fkey] = np.array([FeatTuple[mi][0][chi][fkey] for mi in range(Nsims)]).T
    #organize synchrony features 
    #[realization][] to  #[stimsAmp][Features][stim/time,realization,Channel combination] 
    if Nmodels >1:
        featemp = FeatTuple[0][1]
        FeatsMulti[si] = dict.fromkeys(featemp.keys(), None)
        for fkey in featemp.keys():
            if fkey == 'SynchPairs':
                FeatsMulti[si][fkey] = np.array([FeatTuple[0][1][fkey]])
            else:    
                FeatsMulti[si][fkey] = np.ndarray(shape = [len(stimTS),Nsims,len(FeatTuple[0][1]["SynchPairs"])])
                for mi in range(Nsims):
                    FeatsMulti[si][fkey][:,mi,:] = FeatTuple[mi][1][fkey]

#%%  Calculates correlation measures - Spearman and Pearson (also mutual information) between shifted parameter (A,B or K) and each feature
#select which features to plot
plotFeatures = ["Var","Skew","Kurt","lag1AC","Hcomp","Hmob","ValeAmp","ValeLag","normEnergy", "SpectCent"]#which features
plotFeaturesSynch = ["MI","Corr","PLV" ]#which synchrony features
xplot = eval("PopsModel[si].%ss[%d,:]"%(xvar,0))
# correlation measures
MeasMI_df, MeasSpCorr_df, MeasCorr_df = CorrMeasures(xplot,Feats,
                                                     plotFeatures,
                                                     plotFeaturesSynch,
                                                     chSet = 0,#select which population/model set
                                                     Nmodels = Nmodels, 
                                                     FeatsMulti = FeatsMulti)
if Nmodels >1:
    MeasMI_df2, MeasSpCorr_df2, MeasCorr_df2 = CorrMeasures(xplot,Feats,
                                                         plotFeatures,
                                                         plotFeaturesSynch,
                                                         chSet = 1,#select which population/model set
                                                         Nmodels = Nmodels, 
                                                         FeatsMulti = FeatsMulti)
    
    MeasMI_df = pd.concat([MeasMI_df,MeasMI_df2], ignore_index=True)
    MeasSpCorr_df = pd.concat([MeasSpCorr_df, MeasSpCorr_df2], ignore_index=True)  
    MeasCorr_df = pd.concat([MeasCorr_df, MeasCorr_df2], ignore_index=True)  

#Features x realizations x different simPars x measures

#% save features and measures  
MeasMI_df.to_pickle('./SimFeatures/%s-%s_MI.pkl'%(configProb,xvar))    #to save the dataframe, df to 123.pkl
MeasSpCorr_df.to_pickle('./SimFeatures/%s-%s_SpCorr.pkl'%(configProb,xvar))    #to save the dataframe, df to 123.pkl
MeasCorr_df.to_pickle('./SimFeatures/%s-%s_Corr.pkl'%(configProb,xvar))    #to save the dataframe, df to 123.pkl

np.save("./SimFeatures/%s-%s_Feats"%(configProb,xvar),Feats)
np.save("./SimFeatures/%s-%s_FeatsMulti"%(configProb,xvar),FeatsMulti)
    
    
#%% Figure 2 - simulate LFPs with and without active probing, from normal to ictal activity
  
stimAmp = [0,120]# Simulate with and without probing
Nmodels = 2
#
Pops = CA1popSet(coupling = "inter", #"inter" for inter-hemisphere (between PYR cells) or "intra" for intra-hemisphere coupling (between basket cells)
                 Nsets = Nmodels, A = 4.,B = 40, G = 20,K = 0.3)
PopsModel = [None]*len(stimAmp)
Feats = [None]*len(stimAmp) #[stimsAmp][ModelSubsets][Features][stim/time,realization]
FeatsMulti = [None]*len(stimAmp) #[stimsAmp][Features][stim/time,realization,Channel combination]
simulatedLFP = [None]*len(stimAmp)
for si,stimTemp in enumerate(stimAmp):# for each stimulus parameter
    PopsModel[si] = CA1simulation(Pops, #populations to simulate
                                 Fs = 512., #user-defined sampling frequency (typical: 512 Hz)
                                 finalTime=2000.,#total simulation time, in seconds
                                 stimAmp = stimTemp,#stimulus amplitude
                                 stimPeriod = 2.,#stimulus period
                                 stimDuration = 0.01,#stimulus duration (seconds)
                                 stimPos = "DG",#"DG" (sum stimulus to noise input to main cells - simulates stimulus to the DG) or "all" (sum stimulus to PSPs of all cells)
                                 biphasic = 0)# #biphasic (1) or monophasic (0) stimulation pulse  
    PopsModel[si].As[0,:] = np.linspace(2.5,5.3, PopsModel[si].nbSamples)
    #PopsModel[si].stimInput[1][:,:] = 0#No stimulus contralateral to ictal onset
    simulatedLFP[si] = PopsModel[si].simulateLFP(highpass = 1)
    tvec = np.arange(0,PopsModel[si].finalTime,1/PopsModel[si].Fs) 

#%
xLims = np.array([[1331,1351],[1505,1525],[1741,1721]])

# % plot Simulation epochs
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams.update({'font.size': 8})
popSet = 0#which population - 0 or 1
if popSet == 0:
    yLims = np.array([[-19,15],[-1.5,4]])
    yBar = np.array([[5.5,10.5],[1.7,2.7]])
else:
    yLims = np.array([[-19,15],[-1.5,4]])
    yBar = np.array([[5.5,10.5],[1.7,2.7]])    
    

for sb in range(2):
    if sb == 0:# control (no probing) shows transition + 2 subplots - before and during seizure
        grid = plt.GridSpec(2, 2)
    else:# probing shows transition + 3 subplots - way before seizure, during increased responses and during seizure
        grid = plt.GridSpec(2, 3)
    plt.figure(figsize=(3.72,3))
    ax1 = plt.subplot(grid[0,0:])
    ax1.plot(tvec,simulatedLFP[sb][popSet,:],'k',LineWidth=0.8)
    plt.xlim((1290,1750))
    plt.ylim(yLims[0,:])
    plt.plot([xLims[0,0],xLims[0,0]+50],[np.mean(yBar[0,:]),np.mean(yBar[0,:])],'k',LineWidth = 2) #x scale bar
    plt.plot([xLims[0,0],xLims[0,0]],yBar[0,:],'k',LineWidth = 2)
    plt.text(xLims[0,0]+12,np.mean(yBar[0,:])+1.5,'50 sec',fontsize = 8)
    plt.text(xLims[0,0]-30,np.mean(yBar[0,:])-1,'5 a.u.',fontsize = 8)
    if sb > 0:
        plt.plot(xLims[0,:],[yLims[0,0]+0.2,yLims[0,0]+0.2],'r',LineWidth=1.5)
    plt.plot(xLims[1,:],[yLims[0,0]+0.2,yLims[0,0]+0.2],'r',LineWidth=1.5)
    plt.plot(xLims[2,:],[yLims[0,0]+0.2,yLims[0,0]+0.2],'r',LineWidth=1.5)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().get_xaxis().set_ticks([])
    plt.gca().get_yaxis().set_ticks([])
    
    ax2 = plt.subplot(grid[1,0])
    ax2.plot(tvec,simulatedLFP[sb][popSet,:],'k',LineWidth=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().tick_params(labelbottom=False)  
    if sb == 0:
        plt.xlim(xLims[1,:]) 
        #plt.gca().locator_params(axis='x', nbins=9)   
        plt.plot([xLims[1,0]+2.45,xLims[1,0]+2.45],yBar[1,:],'k',LineWidth = 2)
        plt.plot([xLims[1,0]+2.45,xLims[1,0]+7.45],[np.mean(yBar[1,:]),np.mean(yBar[1,:])],'k',LineWidth = 2)
        plt.text(xLims[1,0]+0.75,np.mean(yBar[1,:]),'1 a.u.',fontsize = 8)
        plt.text(xLims[1,0]+2.75,np.mean(yBar[1,:])+0.2,'5 sec',fontsize = 8)
        plt.gca().get_xaxis().set_ticks([])
    else:
        plt.xlim(xLims[0,:])
        plt.plot([xLims[0,0]+3.5,xLims[0,0]+3.5],yBar[1,:],'k',LineWidth = 2)
        plt.plot([xLims[0,0]+3.5,xLims[0,0]+8.45],[np.mean(yBar[1,:]),np.mean(yBar[1,:])],'k',LineWidth = 2)
        plt.text(xLims[0,0]+0.75,2.9,'1 a.u.',fontsize = 8)
        plt.gca().locator_params(axis='x', nbins=10)   
    plt.ylim(yLims[1,:])
        
    ax2 = plt.subplot(grid[1,1])
    ax2.plot(tvec,simulatedLFP[sb][popSet,:],'k',LineWidth=1)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)
    plt.gca().get_yaxis().set_ticks([])
    plt.gca().tick_params(labelbottom=False)  
    if sb == 1:
        plt.xlim(xLims[1,:]) 
        plt.ylim(yLims[0,:])
        plt.gca().locator_params(axis='x', nbins=10)   
        plt.plot([xLims[1,1]-8,xLims[1,1]-8],yBar[0,:],'k',LineWidth = 2)
        #plt.plot([xLims[1,1]-8,xLims[1,1]-3],[4.5,4.5],'k',LineWidth = 2)
    else:
        plt.gca().get_xaxis().set_ticks([])
    
    if sb == 1:
        ax3 = plt.subplot(grid[1,2])
        ax3.plot(tvec,simulatedLFP[sb][popSet,:],'k',LineWidth=1)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['bottom'].set_visible(False)
        plt.gca().spines['left'].set_visible(False)
        plt.gca().get_yaxis().set_ticks([])
        plt.gca().tick_params(labelbottom=False) 
        plt.gca().locator_params(axis='x', nbins=10) 
    else:
        plt.gca().get_xaxis().set_ticks([])

    plt.plot([xLims[2,1]-8,xLims[2,1]-8],yBar[0,:],'k',LineWidth = 2)
    plt.xlim(xLims[2,:])  
    plt.ylim(yLims[0,:])  
    plt.tight_layout()
    
#%% Fig 2.c - each subplot with mean response for specific parameter

tprePEARP = 0.1 # pre-stimulus period
tposPEARP = 0.4 # post-stimulus period for feature extraction

stimAmp = 120 # Simulate with and without probing
Nmodels = 2
AAs = [2.5 , 3, 3.5, 4.0, 4.5, 5.0 ]
#
PopsModel = [None]*len(AAs)
Feats = [None]*len(AAs) #[stimsAmp][ModelSubsets][Features][stim/time,realization]
FeatsMulti = [None]*len(AAs) #[stimsAmp][Features][stim/time,realization,Channel combination]
simulatedLFP = [None]*len(AAs)

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams.update({'font.size': 8})


figWaves = [plt.figure(figsize = [3.7, 1.5]),plt.figure(figsize = [3.7, 1.5])]
for si,Aiter in enumerate(AAs):
    Pops = CA1popSet(coupling = "inter", #"inter" for inter-hemisphere (between PYR cells) or "intra" for intra-hemisphere coupling (between basket cells)
                     Nsets = Nmodels, A = Aiter,B = 40, G = 20,K = 0.3)

    PopsModel[si] = CA1simulation(Pops, #populations to simulate
                                 Fs = 512., #user-defined sampling frequency (typical: 512 Hz)
                                 finalTime=50.,#total simulation time, in seconds
                                 stimAmp = stimTemp,#stimulus amplitude
                                 stimPeriod = 2.,#stimulus period
                                 stimDuration = 0.01,#stimulus duration (seconds)
                                 stimPos = "DG",#"DG" (sum stimulus to noise input to main cells - simulates stimulus to the DG) or "all" (sum stimulus to PSPs of all cells)
                                 biphasic = 0)# #biphasic (1) or monophasic (0) stimulation pulse  
    #PopsModel[si].As[0,:] = np.linspace(2.5,5.3, PopsModel[si].nbSamples)
    #PopsModel[si].stimInput[1][:,:] = 0#No stimulus ipsilateral to ictal onset
    simulatedLFP[si] = PopsModel[si].simulateLFP(highpass = 0)
    tvec = np.arange(0,PopsModel[si].finalTime,1/PopsModel[si].Fs) 
    
for si,Aiter in enumerate(AAs):  
    stimTS = np.arange(int(PopsModel[si].Fs*PopsModel[si].stimPeriod),PopsModel[si].nbSamples,int(PopsModel[si].Fs*PopsModel[si].stimPeriod))
    allPEARPS = np.ndarray(shape=(len(stimTS),int((tprePEARP)*PopsModel[si].Fs)+int((tposPEARP)*PopsModel[si].Fs),Nmodels))#(Stims,timesamples,channel)
    for s_ts in range(len(stimTS)):
        indsERP = np.arange(stimTS[s_ts],stimTS[s_ts]+int(tposPEARP*PopsModel[si].Fs))#ERP indexes (only for "feature" averaging)
        #indsERP2 = range(max(0,si-avrgWinSize+1),si+1) #indexes - which ERPs to average (relative to allPEARPS) - mean ERP of last "avrgWinSize" stimuli
        for chi in range(Nmodels):#for each channel/model
            allPEARPS[s_ts,:,chi] = simulatedLFP[si][chi,stimTS[s_ts]-int(tprePEARP*PopsModel[si].Fs):stimTS[s_ts]+int(tposPEARP*PopsModel[si].Fs)]

    #
    tpearp = np.linspace(0,allPEARPS.shape[1]/PopsModel[si].Fs,allPEARPS.shape[1])
    tpearp = tpearp-tprePEARP

    for ch_i in [0,1]:
        plt.figure(figWaves[ch_i].number)
        plt.subplot(1,len(AAs),si+1)
        #ax = plt.axes()
        #ax.set_prop_cycle('color',[plt.cm.Blues(i) for i in np.linspace(0, 1, allPEARPS.shape[0])])
        plt.plot(tpearp,allPEARPS[:,:,ch_i].T,alpha = .2, color = 'k', linewidth = .5)
        plt.plot(tpearp,np.mean(allPEARPS[:,:,ch_i],axis = 0),color = 'k',linewidth = 2.5)
        plt.fill_between(tpearp, np.mean(allPEARPS[:,:,ch_i],axis = 0)-np.std(allPEARPS[:,:,ch_i],axis = 0),
                                 np.mean(allPEARPS[:,:,ch_i],axis = 0)+np.std(allPEARPS[:,:,ch_i],axis = 0), 
                                 color='k', alpha=.1)
        plt.axis('off')
        plt.ylim([-4,4])
        #plt.title('A$_1$ = %.1f'%Aiter)
plt.figure(figWaves[0].number)
plt.tight_layout()    
plt.figure(figWaves[1].number)
plt.tight_layout()    


#%%    
    
"""
if 0:
    %% Reduce feature file size - delete features not used in figure
    for arquivo in ["I-A","I-B","I-K","II-A","II-B","II-K"]: #which configuration (II-A,II-B,II-K,III-A,III-B,III-K)
        Feats = np.load("%s_Feats.npy"%arquivo,allow_pickle = True)
        FeatsMulti = np.load("%s_FeatsMulti.npy"%arquivo,allow_pickle = True)
        
        for ii in range(Feats.shape[0]):
            for jj in range(Feats.shape[1]):
                for kk in ["Ictal","Energy","PkAmp","PkLag","ValeAmp","ValeLag","Hmob","Hcomp"]:
                    del Feats[ii,jj][kk]
                    
        for ii in range(FeatsMulti.shape[0]):
            for kk in ["Coh","Corr","PLVphase","PLV"]:
                del FeatsMulti[ii][kk]
            
        Feats = Feats[:,:2]#delete mean LFP features
        np.save("%s_Feats.npy"%arquivo,Feats)
        np.save("%s_Feats.npy"%arquivo,FeatsMulti)
    
"""