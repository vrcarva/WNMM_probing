# -*- coding: utf-8 -*-
"""
Loads .pkl and .npy files from ./SimFeatures/ and plots figures from the manuscript.

@author: Vinícius Rezende Carvalho
vrcarva@ufmg.br
"""

import numpy as np
import matplotlib.pyplot as plt
from statsmodels.stats.multicomp import MultiComparison
from scipy import stats
import pandas as pd
from scipy.ndimage.filters import uniform_filter1d
#from mpl_toolkits.axes_grid1 import make_axes_locatable
#from sklearn.feature_selection import mutual_info_regression
#import seaborn as sns

    
def plotUniFigs(pFeats,xlabels,xvar,Fs = 512, zscore = 0,sb_row = 3, sb_col = 3, plotColor = [0,0,0],sb_offset = 0):
    #plot univariate feature series, with mean+confidence intervals across realizations
    plotFeatures =  ["Var","Skew","Kurt","lag1AC"]
    plotTitle = ["Variance","Skewness","Kurtosis","Lag-1 AC"]
    for sbNum,fkey in enumerate(plotFeatures):
        plt.subplot(sb_row,sb_col,sb_offset+sbNum+1)
        y = uniform_filter1d(pFeats[fkey], size=avrgWinSize,axis = 0)
        if zscore == 1:
            y = (y - np.mean(y,axis=0))/np.std(y,axis = 0)
        if fkey == "Var":
            y = np.log10(y)
        fSEM = np.std(y,axis = 1) 
        y = np.mean(y,axis = 1)
        plt.plot(xvar,y,c = plotColor)  
        plt.fill_between(xvar, y-fSEM, y+fSEM,alpha = 0.4, color = plotColor, linewidth = 0)
        if sb_offset == 0: #first row (set 1) plots
            plt.title(plotTitle[sbNum])
            plt.gca().spines['bottom'].set_visible(False) 
            plt.gca().get_xaxis().set_ticks([])  
        else:
            plt.gca().set_xticks(np.linspace(xvar[0],xvar[-1],3))
            plt.xlabel(xlabels)          
        plt.autoscale(tight=True)
        if fkey == "Var":
            plt.ylabel('log10')
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)  
         
        
        plt.gca().locator_params(axis='y', nbins=2)
def multiCompare(dfMeasure, selectedFeats):
    #Multiple comparisons between correlation measures - all stimuli amplitudes vs one(no stimulation)
    #dfMeasure: Dataframe with correlation measures
    #selectedFeats: Selected features to compare
    ps_, h_rej = [None]*2, [None]*2
    
    for chi in np.unique(dfMeasure["channel"]):
        ps_[chi],h_rej[chi] = {},{}
        
        for sbNum,fkey in enumerate(plotFeatLabels):
            mc = MultiComparison(dfMeasure.loc[dfMeasure["channel"]==chi,fkey],
                                   dfMeasure.loc[dfMeasure["channel"]==chi,"StimAmp"]
                                  )
            mc_results = mc.tukeyhsd()  
            ps_[chi][fkey] = mc_results.pvalues[:sAmps.size-1]
            h_rej[chi][fkey] = mc_results.reject[:sAmps.size-1]
        
        ps_[chi] = pd.DataFrame.from_dict(ps_[chi], orient='index',columns=sAmps[1:])
        h_rej[chi] = pd.DataFrame.from_dict(h_rej[chi], orient='index',columns=sAmps[1:])
    return ps_,h_rej                            
 



#%% Plot feature series and correlation measures figures 3 to 8 (change )

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams.update({'font.size': 8})

#simulation parameters
avrgWinSize = 20
Fs = 512
Nstim = 999 #number of features (or stimuli epochs) in each simulation 
stimAmp = np.linspace(0,200,11)

modelSet = "II-A" #which configuration (II-A,II-B,II-K,III-A,III-B,III-K)
Feats = np.load("./SimFeatures/%s_Feats.npy"%modelSet,allow_pickle = True)
FeatsMulti = np.load("./SimFeatures/%s_FeatsMulti.npy"%modelSet,allow_pickle = True)


xvarFeats = {"A":np.linspace(2.5,4.7,Nstim),"B":np.linspace(45,30,Nstim),'K':np.linspace(0,0.5,Nstim)}
xaxsLabel = {"A": 'EXC (A)',"B": "SDI (B)",'K':'Coupling gain (K)'}

uiStim = [0,4,8,10] #plot features w/ confidence intervals from which stimulation parameters? (index) of stimAmp/Feats/uinput
#uiStim = [0,3,6]

#cores = [plt.cm.Blues(int(ci)) for ci in np.linspace(110,200,len(uiStim))]
cores = [plt.cm.Blues(int(ci)) for ci in np.linspace(130,255,len(uiStim))]

xvar = xvarFeats[modelSet[-1]]
figFeat = plt.figure(figsize=(6.5,  3.8))
[plotUniFigs(Feats[ui][0],xlabels = xaxsLabel[modelSet[-1]],xvar = xvar,zscore = 0,sb_row = 3, sb_col = 5, plotColor = ci) for ui,ci in zip(uiStim,cores)]

plotFeaturesSynch = ["MI"]#which synchrony features
for ui,ci in zip(uiStim,cores):
    plt.subplot(3,5,5)
    y = uniform_filter1d(FeatsMulti[ui]["MI"][:,:,0], size=avrgWinSize,axis = 0)
    #y = FeatsMulti[ui][fkey][:,:,0]
    fSEM = np.std(y,axis = 1)
    plt.plot(xvar,np.mean(y,axis = 1), c = ci)  
    plt.tight_layout()
    plt.fill_between(xvar,np.mean(y,axis = 1)-fSEM, np.mean(y,axis = 1)+fSEM,alpha = 0.4, color = ci, linewidth = 0)
    plt.axis('tight')
    plt.title("MI")
    plt.ylabel('Bits')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().locator_params(axis='y', nbins=2)
    #plt.gca().get_xaxis().set_ticks([])
    plt.gca().set_xticks(np.linspace(xvar[0],xvar[-1],3))
    plt.xlabel(xaxsLabel[modelSet[-1]])    
    plt.autoscale(tight=True)
    _, _, ymin, ymax = plt.axis()

#plt.plot([150, 400],[ymin+0.05*(ymax-ymin), ymin+ 0.05*(ymax-ymin)],'k',Linewidth = 2)
#plt.text(100, ymin - (0.08)*(ymax-ymin), "250 s", fontsize=8)
#plt.legend(stimAmp[uiStim].astype(int))

[plotUniFigs(Feats[ui][1],xlabels = xaxsLabel[modelSet[-1]],xvar = xvar,zscore = 0,sb_row = 3, sb_col = 5, plotColor = ci,sb_offset=5) for ui,ci in zip(uiStim,cores)]
#correlation measurements
MeasMI_df = pd.read_pickle("./SimFeatures/%s_MI.pkl"%modelSet)
MeasSpCorr_df = pd.read_pickle("./SimFeatures/%s_SpCorr.pkl"%modelSet)
MeasCorr_df = pd.read_pickle("./SimFeatures/%s_Corr.pkl"%modelSet)  

plotFeatLabels = ["Var","Skew","Kurt","lag1AC","MI" ]
plotTitle = ["Variance","Skewness","Kurtosis","Lag-1 AC","MI"]
    
sAmps = np.unique(MeasMI_df["StimAmp"])
#Multiple comparisons  
psSpCorr, hSpCorr = multiCompare(MeasSpCorr_df, plotFeatLabels)   
chSet = [0,1]
eLineWs = [1, 2]
eAlphas = [1,0.75]
eColors = ['indianred','darkred']
for canal in chSet:
    for sbNum,fkey in enumerate(plotFeatLabels):
        ax1 = plt.subplot(3,5,10+sbNum+1)    
    #Spearman Correlation
        ymeans2 = np.array([np.mean(MeasSpCorr_df.loc[(MeasSpCorr_df["StimAmp"]==si) & (MeasSpCorr_df["channel"]==canal),fkey]) for si in sAmps])
        ystd2 = [np.std(MeasSpCorr_df.loc[(MeasSpCorr_df["StimAmp"]==si) & (MeasSpCorr_df["channel"]==canal),fkey]) for si in sAmps]
        if fkey == "PLV" or fkey == "MI":
            if canal == 0:
                ax1.plot(sAmps[1:][hSpCorr[canal].loc[fkey]],ymeans2[1:][hSpCorr[canal].loc[fkey]],'k*',label='_nolegend_', ms = 3) 
                ax1.errorbar(sAmps, ymeans2,yerr=ystd2,ms = 3,fmt='-D',color = "gray",zorder=1,elinewidth=eLineWs[canal],alpha = eAlphas[canal])  
                plt.ylabel('ρ')
        else:
            ax1.plot(sAmps[1:][hSpCorr[canal].loc[fkey]],ymeans2[1:][hSpCorr[canal].loc[fkey]],'k*',label='_nolegend_',ms = 3) 
            ax1.errorbar(sAmps, ymeans2,yerr=ystd2,ms = 3,fmt='-D',zorder=1,elinewidth=eLineWs[canal],alpha = eAlphas[canal])  
            if (canal == 0) & (sbNum == 0):
                plt.ylabel('ρ')
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.locator_params(axis='y', nbins=3)
        #ax1.locator_params(axis='x', nbins=8)
        plt.xticks([0,100,200])
        ax1.set_xticks([20,40,60,80,120,140,160,180], minor = True)
        
       # plt.title("%s"%(plotTitle[sbNum]))
        plt.xlabel('Stim. amplitude')
        #if sbNum == 0:
        #    plt.legend(["Set %d"%(legI+1) for legI in np.array(chSet)]) 

plt.tight_layout()
plt.subplots_adjust(wspace = 0.45)
plt.savefig("%s_Feats_Meas.pdf"%(modelSet), transparent=True)
plt.savefig("%s_Feats_Meas.png"%(modelSet), transparent=True)

#%% ICTAL ONSETS - figure 9
from scipy.stats import shapiro
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
plt.rc('font',**{'family':'serif','serif':['Arial']})
plt.rcParams.update({'font.size': 8})
Fs = 512
stimTS = np.arange(512*2,int(512*2000),512*2)    
stimAmp = np.linspace(0,200,11)

modelSets = ["II-A","III-A","II-B","III-B","II-K","III-K"] #load ictal onsets for each configuration

eLineWs = [1, 2]
eAlphas = [1,0.75]
   
chSet = [0,1]
fig1 = plt.figure(figsize=(4,  4.8))

for spi,iconf in enumerate(modelSets):
    plt.subplot(3,2,spi+1)
    if iconf[-1] == "A":
        FeatsLoad = np.load("./SimFeatures/%s_Feats_IctalOnset[2_5 5_5].npy"%iconf,allow_pickle = True)
    elif iconf[-1] == "B":
        FeatsLoad = np.load("./SimFeatures/%s_Feats_IctalOnset[45 20].npy"%iconf,allow_pickle = True)
    elif iconf[-1] == "K":
        FeatsLoad = np.load("./SimFeatures/%s_Feats_IctalOnset[0 1].npy"%iconf,allow_pickle = True)

    for canal in chSet:
        yAll =  np.array([FeatsLoad[ui][canal]["IctalOnset"] for ui in range(len(stimAmp))])/Fs
        ymean = np.array([np.nanmean(FeatsLoad[ui][canal]["IctalOnset"]) for ui in range(len(stimAmp))])/Fs
        ystd = np.array([np.nanstd(FeatsLoad[ui][canal]["IctalOnset"]) for ui in range(len(stimAmp))])/Fs
        #for ii in range(yAll.shape[0]):
        #    stat, p = shapiro(yAll[ii,:])
        #    print(p)    
        #one way anova         
        #one way anova
        F_statistic, pVal = stats.f_oneway(*[yAll[ii,:] for ii in range(len(stimAmp))])  
        print("%s set %d:%f"%(iconf,canal,pVal))
        #multiple comparisons
        yMC = np.reshape(yAll,[-1])
        MClabels = stimAmp[0]*np.ones(yAll.shape[1])
        for iLab in stimAmp[1:]: 
            MClabels = np.hstack((MClabels,iLab*np.ones(yAll.shape[1])))
        
        mc = MultiComparison(yMC,MClabels)   
        mc_results = mc.tukeyhsd()  
        ps_ = mc_results.pvalues[:stimAmp.size-1]
        h_rej = mc_results.reject[:stimAmp.size-1]         
        
        plt.plot(stimAmp[1:][h_rej],ymean[1:][h_rej],'k*',ms = 3,label='_nolegend_')
        plt.errorbar(stimAmp, ymean,yerr=ystd,fmt='-o',elinewidth=eLineWs[canal],alpha = eAlphas[canal],ms=3,zorder = -1)    
        
        dataTemp = {'ictalOnset': yMC,'amp':MClabels,"popSet":(canal+1)*np.ones(len(MClabels))}
        if canal >0:
            dataPlot = dataPlot.append(pd.DataFrame(dataTemp))
        else:
            dataPlot = pd.DataFrame(dataTemp)
    if spi == 0:
        plt.legend(["Set 1", "Set 2"])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    #plt.gca().locator_params(axis='x', nbins=11)
    plt.xticks([0,40,80,120,160,200])
    plt.gca().locator_params(axis='y', nbins=4)
    plt.xlabel("Stimulus amplitude")
    plt.ylabel("Ictal onset (s)")
    plt.title(iconf[1:])
plt.tight_layout()    

plt.savefig("%s_IctalOnsets.pdf"%(modelSets), transparent=True)
plt.savefig("%s_IctalOnsets.png"%(modelSets), transparent=True)

    
    

    