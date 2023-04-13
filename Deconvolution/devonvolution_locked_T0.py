# -*- coding: utf-8 -*-

# import packages

import csv
import time
from itertools import zip_longest as zip
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames

import matplotlib.pyplot as plt
import numpy as np
from numpy.core.numeric import NaN
from scipy.optimize import minimize

import re as re
from datetime import datetime
from pathlib import Path

# define molecule class

class FitResult:

    """This object holds all results of a deconvolution fit"""

    def unpackResults(self):

        self.fitAmplitudes = np.concatenate((self.parent.emptyAmps, self.fitResult['x']))

    def getResultStrings(self):

        self.resultStrings = []

        self.resultStrings.append(f'Local Results:')

        self.resultStrings.append(f'success: {self.fitResult.success}, message: {self.fitResult.message}, iterations: {self.fitResult.nit}, function evals: {self.fitResult.nfev}')
        self.resultStrings.append(f'squares: {self.SquareResult}, entropy: {self.EntropyResult}, entropyfactor: {self.entropyScale}, totalres: {self.TotalResult}')

        return self.resultStrings
    
    def __init__(self, parent, fitResult, fitDataCache:dict, isGlobal:bool = False):

        self.fitResult = fitResult

        self.SquareResult = fitDataCache['SquareResult']
        self.EntropyResult = fitDataCache['EntropyResult']
        self.TotalResult = fitDataCache['TotalResult']

        self.isGlobal = isGlobal
        self.entropyScale = entropyScale

        self.parent = parent

        self.unpackResults()
        self.getResultStrings()

class MoleculeData:
    """This object holds all data of an individual molecule"""

    def getKernel(self, x = None, Amp = None, x0 = None): # sum auf multiple gauss of same width but different centres, at offset x0 with amplitude Amp

        if x is None: x = self.bins
        if x0 is None: x0 = self.x0
        if Amp is None: Amp = 1

        p = 0

        for i, c in enumerate(self.centres):

            p += gauss(x, x0+c, Amp, self.gaussWidth)*self.baseAmps[i]

        return p

    #(bins, x0, bpConversion, gaussWidth, pGuesses2)

    def calcLandscape(self, x = None, x0 = None, Amplitudes = None): # sum of a series of gaussians starting at x0 with separation of their centers of dx

        if x is None: x = self.bins
        if x0 is None: x0 = self.x0
        if Amplitudes is None: Amplitudes = self.Amplitudes
        
        y = 0

        offset = x0    # initial value, then progress in bp steps
            
        for Amp in Amplitudes:
            
            y += self.getKernel(x, Amp, offset)
            offset += bpConversion

        return y

    def readHeader(self):

        # import kernel data from header

        self.path = Path(file)
        self.prefix = self.path.stem
        self.folder = self.path.parent

        headerdata = {}

        with open(self.file, "r") as read_file:

            for i in range(9): 

                line = read_file.readline()
                reg = r"\A(\w+)[ =]*([-0-9.Ee]*)\s*\Z"
                readout = re.findall(reg, line)[0]
                var = readout[0]
                value = float(readout[1])

                headerdata[var] = value

        self.gaussWidth = headerdata['w']

        baseAmps = np.array([headerdata['Al'], headerdata['Ac'], headerdata['Ar'], headerdata['Ae']])   # the kernel is given by the sum of three gaussians with different amplitude and centres but shared width

        self.baseAmps = baseAmps/np.sum(baseAmps)    #normalize
        self.centres = np.array([headerdata["xl"], headerdata["xc"], headerdata["xr"], headerdata["xe"]])
        
        AmpRes = minimize(self.getKernel, np.sum(self.centres * self.baseAmps), args=(-1, 0))  # find baseline peak

        self.kernelPeakSize = -AmpRes['fun']  # amplitude peak height of a normalized kernel distribution
        self.centres -= x0  # make centres relativ to offset

    def readFileData(self):

        with open(self.file, "r") as read_file:

            data = np.loadtxt(read_file, delimiter='\t', skiprows=12)

        self.bins = data[:,0]   # positions
        self.pvalues = data[:,1]    # apparent amplitudes
        self.indeces = np.arange(len(self.bins))

        self.binsize = np.round(self.bins[1] - self.bins[0], 10)

        self.pSum = np.sum(self.pvalues)
        self.pvaluesNorm = self.pvalues/(self.pSum*self.binsize)
        self.Evalues = -np.log(self.pvaluesNorm)

        self.minAmp = 0.1/self.pSum

        self.pGuesses = [self.minAmp] * (RLoopSize + 1)  # initial amplitudes for each position (per bp of Rloop lenght), start at max Energy for each

        self.bps = RloopPosArray

        self.bins_bp = (self.bins - self.x0)/bpConversion
        self.bps_rad = (self.bps * bpConversion) + self.x0

        self.firstElement = int((self.bins[0]-x0)/bpConversion)-1

        self.emptyAmps = np.full(self.firstElement, np.exp(-20))

    def getKernelOffsetLSScore(self, args): # Minimization function

        offset = args[0] # angle offset
        shift = args[1] # Amplitude of kernel in energy landscape

        binlow = np.round(-kernelRangeSize/2 + offset, 1)
        binhigh = binlow + kernelRangeSize

        lowIndex = int(round(np.interp(binlow, self.bins, self.indeces), 0))
        highIndex = int(round(np.interp(binhigh, self.bins, self.indeces), 0))+1

        kernelBins = np.arange(self.bins[lowIndex], self.bins[highIndex]-0.05, 0.1)   # Area to fit the initial kernal offset to

        self.EKernel = -np.log(self.getKernel(kernelBins, shift, offset))
        self.EKernelMeas = self.Evalues[lowIndex:highIndex]

        return np.sum((self.EKernel - self.EKernelMeas)**2)
    
    def precondition(self): # Preconditioning: iteratively close in on apparent energylandscape to create good parameter guesses

        rloopBpPos = np.arange(self.x0, self.x0 + (RLoopSize+1)*bpConversion, bpConversion)
        self.rloopBpPos = rloopBpPos[0:RLoopSize+1] #ensure correct length
        
        self.bpPositionBins = []

        for pos in self.rloopBpPos:

            found = np.interp(pos, self.bins, self.indeces)

            self.bpPositionBins.append(int(round(found, 0)))

        self.pApparent = np.array(self.pvaluesNorm)
        self.guessGaussSum = np.zeros(len(self.bins))

        # for i in range(2000):

        #     result = (self.pApparent - self.guessGaussSum)[max(self.bpPositionBins[0]-1,0):min(self.bpPositionBins[-1]+1,len(self.bins))]

        #     max_bin_index = result.argmax()+self.bpPositionBins[0]-1   # bin index, where the biggest difference is located
            
        #     max_pos = self.bins[max_bin_index]     # same, but in rad (pos on apparent energy landscape)

        #     max_pos_bp = (max_pos - self.x0)/bpConversion # same, but in bp (of Rloop length)

        #     frac = max_pos_bp%1

        #     if max_pos_bp >= rloopFitPositions - 1: # if the value to be edited is the max Rloop length's parameter, avoid running out of bounds of the amplitude array

        #         max_pos_bp = rloopFitPositions - 2
        #         frac = 1

        #     elif max_pos_bp < 0:

        #         max_pos_bp = 0
        #         frac = 0

        #     max_pos_bp = int(max_pos_bp)

        #     lower = max_pos_bp
        #     upper = max_pos_bp+1

        #     max_val_lower = result.max() / self.kernelPeakSize * 0.25 * (1-frac)    # kernelPeakSize is the amplitude peak height of a normalized kernel distribution
        #     max_val_upper = result.max() / self.kernelPeakSize * 0.25 * (frac)

        #     self.pGuesses[lower] += max_val_lower
        #     self.pGuesses[upper] += max_val_upper

        #     if upper == rloopFitPositions-1:

        #         for i in range(upper + 1, RLoopSize + 1):

        #             self.pGuesses[i] = self.pGuesses[upper] * np.exp(- dE_truncation * bpConversion * (i - upper))   # update truncation penalty to mismatch tail
            
        #     self.pGuesses = [1-1/self.pSum if x > 1-1/self.pSum else 1/self.pSum if x < 1/self.pSum else x for x in self.pGuesses]
           
        #     self.guessGaussSum = self.calcLandscape(None, None, self.pGuesses)

            # plot

            # guessplot = -np.log(self.guessGaussSum)

            # deconv_bar_pos = np.array(range(0,RLoopSize+1))
            # deconv_bar_val = -np.log(np.array(self.pGuesses)) # devide by (sqrt2pi*gaussWidth) to align "peak" amplitudes

            # bps = (self.bins - self.x0)/bpConversion

            # plt.figure(1).clear()
            # plt.plot(bps, guessplot, 'r-')
            # plt.plot(bps, self.Evalues, 'g-')
            # plt.bar(deconv_bar_pos, deconv_bar_val, width=0.8*bpConversion)
            # plt.draw()
            # plt.pause(.2)

            # if result.max() < 10/self.pSum: 

            #     print(f'Preconditioning Steps {self.prefix}: {i}')

            #     break

        self.pGuessesInt =  np.array(self.pGuesses)[0:rloopFitPositions]
        self.guessedFreeEnergy = -np.log(self.guessGaussSum)

    def setWeights(self):

        # define weights for fitting
    
        # weights = np.ones(len(self.bins))

        # ZeroPos = self.bpPositionBins[0]                    # baseline
        # MaxPos = self.bpPositionBins[-1]                    # minimum of rloop

        # FitRangeBins = int(np.round(FitRange/self.binsize,0))

        # for i, bin in enumerate(self.bins):

        #     if i <= (ZeroPos - FitRangeBins): # ignore anything before ~ 1.1 rad of the baseline peak

        #         weights[i] = 0

        #     elif bin < (areaofintereststart + self.x0):

        #         weights[i] = lowweight
            
        #     elif i > self.bpPositionBins[-2]: 

        #         weights[i] = 0

        #     elif i > MaxPos + FitRangeBins:

        #         weights[i] = lowweight

        weights = (self.pvaluesNorm)**(1/weightpower)

        self.weights = weights/np.sum(weights)*len(weights)

    def getFittingParameters(self):

        return self.x0, self.pGuessesInt, self.pSum

    def getApparentEntropy(self, EAmps):

        diffApparent = np.zeros(len(EAmps)) + 5
        rloopBpPos = np.arange(self.x0, self.x0 + (RLoopSize+1)*bpConversion, bpConversion)

        for i, Efit in enumerate(EAmps):

            pos = rloopBpPos[i+1]

            Eapparent = np.interp(pos, self.bins, self.Evalues)

            diffApparent[i] += Efit - Eapparent
        
        if (diffApparent < 0).any(): # protect from negative values to allow application of log --> shift all energies 

            diffApparent -= min(diffApparent) * 1.01

        EntropyAmps = diffApparent/np.sum(diffApparent) # normalize amplitudes to get a normalized entropy factor

        EntropyResult = -np.sum(EntropyAmps * np.log(EntropyAmps)) - np.log(RloopMatchingRange)

        return EntropyResult
    
    def getLeastSquareWithEntropyLocked(self, VarAmps):
        
        Amps = np.concatenate((self.emptyAmps, VarAmps))

        NormAmps = Amps/np.sum(Amps)

        fullAmplitudes = addMismatchAmplitudes(RLoopSize, NormAmps)
        
        p = self.calcLandscape(self.bins, self.x0, fullAmplitudes)    ## probability Dist. of the molecule with current fitting parameters

        E = -np.log(p)  ## Energy landscape of current parameters

        SquareResult = np.sum((E-self.Evalues)**2*self.weights)   ## E, Evalues and weights are all 1D arrays of same length

        EAmps = -np.log(Amps[1:RloopMatchingRange+1])   # Amplitude in Energy space        

        # EntropyResult = CalculateEntropy(EAmps, MismatchPos) # Amplitudes without (linear) bias
        EntropyResult = self.getApparentEntropy(EAmps) if useApparentEntropy else CalculateEntropy(EAmps) # Amplitudes without (linear) bias
        
        TotalResult = (SquareResult - (EntropyResult * entropyScale))        

        self.fitDataCache = {
            'SquareResult': SquareResult,
            'EntropyResult': EntropyResult,
            'TotalResult': TotalResult
        }

        return TotalResult

    def performLocalFit(self):

        guess = self.pGuessesInt[self.firstElement:]
        bounds = (len(guess))*[(self.minAmp,1-self.minAmp)]

        localFitResult = minimize(self.getLeastSquareWithEntropyLocked, guess, bounds=bounds, method='L-BFGS-B', options=fitOptions)

        self.localFitData = FitResult(self, localFitResult, self.fitDataCache)

        self.fullLocalAmplitudes = addMismatchAmplitudes(RLoopSize, self.localFitData.fitAmplitudes)

        self.fullLocalAmplitudes = self.fullLocalAmplitudes / np.sum(self.fullLocalAmplitudes)
         
        self.localFitEnergies = -np.log(np.array(self.fullLocalAmplitudes))

        self.localGaussFit = self.calcLandscape(self.bins, self.x0, self.fullLocalAmplitudes)
        self.localFreeEnergyFit = -np.log(self.localGaussFit)

    def plotData(self, ax):
        
        ax.plot(self.bins_bp, self.Evalues, '-', color = '#000000')
        ax.plot(self.bins_bp, self.guessedFreeEnergy, '-', color = '#00BB00')
        ax.plot(self.bins_bp, self.localFreeEnergyFit, '-', color = '#184ef2')
        ax.bar(self.bps+0.25, self.localFitEnergies, width=0.4, color = '#809efa')

        plt.draw()
        plt.pause(0.2)

    def plotTest(self, x0, Amps):

        plt.figure(2, clear=True)

        pFitTemp = self.calcLandscape(self.bins, x0, Amps)

        bins_bp = (self.bins - x0)/bpConversion

        ETemp = -np.log(addMismatchAmplitudes(RLoopSize, Amps))
        EFitTemp = -np.log(pFitTemp)

        EGuesses = -np.log(self.pGuesses)
        
        plt.plot(bins_bp, self.Evalues, '-', color = '#000000')
        plt.plot(bins_bp, self.guessedFreeEnergy, '-', color = '#00BB00')
        plt.plot(bins_bp, EFitTemp, '--', color = '#d50b0b')
        plt.bar(RloopPosArray - 0.25, EGuesses, width=0.4, color = '#cff3c9')
        plt.bar(RloopPosArray + 0.25, ETemp, width=0.4, color = '#f79c92')
        plt.draw()
        plt.pause(0.2)

    def writeReport(self):

        outPath = self.folder.parent / "Results" / f"{self.prefix}_{entropyScale:.2f}_deconv2.dat"

        rows = zip(self.bins, self.bins_bp, self.localFreeEnergyFit, self.Evalues, self.bps_rad, self.bps, self.fullLocalAmplitudes, self.localFitEnergies)

        with open(outPath, 'w') as outFile:
       
            wtr = csv.writer(outFile, delimiter=',', lineterminator='\n')

            for string in self.localFitData.resultStrings:

                wtr.writerow([string])

            wtr.writerow([ "ang (rad)", "ang (bp)", "E fit (kT)", "E data (kT)", "ang (rad)", "ang (bp)", "Amplitudes (%)", "Energy (kT)"])

            for row in rows:

                wtr.writerow (row)

    def printResults(self):

        print(f'Molecule: {self.prefix}')

        for string in self.localFitData.resultStrings:

            print(string)


    def __init__(self, file, index):

        self.file = file
        self.index = index
        self.Amplitudes = np.zeros(rloopFitPositions)
        self.x0 = x0

        self.readHeader()
        self.readFileData()
        self.precondition()
        self.setWeights()

# define functions

pi = np.pi
sqrt2pi = np.sqrt(2*pi)

def gauss(x, x0, A, sigma):
    
    y = A/(sqrt2pi*sigma) * np.exp(-1/2*(((x-x0)/sigma) ** 2))
    
    return y

def addMismatchAmplitudes(maxlength, Amps):    

    FullAmplitudes = np.zeros(maxlength+1)

    FullAmplitudes[0:len(Amps)] = Amps

    for bp in range(rloopFitPositions, maxlength + 1):

        FullAmplitudes[bp] = FullAmplitudes[bp-1] * np.exp(- dE_truncation * bpConversion)
    
    return FullAmplitudes

# define fitting functions

def CalculateEntropy(Amps):    # calculates entropy score

    biasSlope = np.mean(Amps[1:] - Amps[:-1]) # get the average change of Energy levels in R-loop

    arraySize = len(Amps)
    EAmpsNoBias = np.zeros(arraySize)

    for i in range(arraySize):

        EAmpsNoBias[i] = Amps[i] - biasSlope * i
    
    if (EAmpsNoBias < 0).any(): # protect from negative values to allow application of log --> shift all energies 

        EAmpsNoBias -= min(EAmpsNoBias) * 1.01

    EntropyAmps = EAmpsNoBias/np.sum(EAmpsNoBias) # normalize amplitudes to get a normalized entropy factor

    EntropyResult = -np.sum(EntropyAmps * np.log(EntropyAmps)) - np.log(RloopMatchingRange)

    return EntropyResult

totalIterations = 0

# parameters to edit

test = False                    # set True for debugging (limits optimization time of fits)

RloopTruncationLength = 0       # number of end mismatches

useApparentEntropy = True       # entropy method

# other global parameters

RLoopSize = 38                                              # total lenght of Rloop in bp
RloopMatchingRange = RLoopSize if useApparentEntropy else RLoopSize - RloopTruncationLength      # max pos of matching strand (bp)

RloopPosArray = np.arange(RLoopSize + 1)                    # used for plotting and results

maxIterations = 100 if test else 1000

dE_truncation = 2.29582 # in kT/rad: fixed penalty for end truncation mismatches of RLoop

x0range = 0.2   # allowed interval size to look for the angle offset
kernelRangeSize = 4

rloopFitPositions = RloopMatchingRange + 1  # how many amplitudes to modulate, +1 to include baseline, +2 to include first mismatch

EntropyScaleValues = [0, 1, 2, 4, 6, 10, 20, 40, 60, 100, 200, 400, 600, 1000]
EntropyScaleValues = [10] if test else EntropyScaleValues

entropyScale = 0 # initial value of the scaling factor for entropy in minimization

# params for weighting

weightpower = 4 # this works as a power since it is applied to the counts of each bin, so it scales line with the energy 

fitOptions = {'maxcor': 5000, 'ftol': 1e-15, 'gtol': 1e-09, 'eps': 1e-09, 'maxfun': 250000, 'maxiter': maxIterations, 'maxls': 20}

## Main

root = Tk() # init GUI
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

# import Energy Landscape

if test == False:

    file = askopenfilename(initialdir='Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data', title='Choose locked energy landscape file') # show an "Open" dialog box and return the path to the selected file
    paramfile = askopenfilename(initialdir='Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data', title='Choose deconvolution file') # show an "Open" dialog box and return the path to the selected file

else:

    file = 'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20210204_T2_locked.dat'
    paramfile = 'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/2021-02-04_T2/Deconvolution/20210204_T2_100.00_deconv2.dat'

headers = []

with open(paramfile, "r") as read_file:

    for i in range(7):

        headers.append(read_file.readline())

x0 = float(re.findall(r"Baseline: (-?\d+\.?\d*)", headers[1])[0])
bpConversion = float(re.findall(r"bpConversion: (\d+\.?\d*)", headers[1])[0])

mol = MoleculeData(file, 0) # create new object

Amps_argIndexStart = 0  # position of first baseline amplitude parameter
Rloop_argIndexStart = 1  # position of first rloop amplitude parameter

plt.ion()

for entropyScale in EntropyScaleValues:

    # perform minimization

    totalIterations = 0

    # calculate results

    mol.performLocalFit()

    # prepare plots

    plt.figure(1, figsize=(16, 9), clear=True)
    plt.subplot(1, 1, 1)

    # plot and write report to file

    mol.plotData(plt)
    
    plt.draw()
    plt.pause(1)

    # write to file
    
    if test != True: mol.writeReport()
    mol.printResults()
        
plt.show()
# input("Press Enter to continue...")
