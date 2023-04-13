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

        resultParameters = self.fitResult['x']

        if self.isGlobal == True:

            self.fitAmplitudes = resultParameters[Rloop_argIndexStart:]
            self.baselines = resultParameters[Amps_argIndexStart:Rloop_argIndexStart]

            if self.parent != None:
                self.x0 = resultParameters[self.parent.index]
                Amp0 = self.baselines[self.parent.index]
                self.fitAmplitudes = np.insert(self.fitAmplitudes, 0, Amp0)

            else:

                self.fitAmplitudes = np.insert(self.fitAmplitudes, 0, np.mean(self.baselines))

        else:
            
            self.fitAmplitudes = resultParameters

    def getResultStrings(self):

        self.resultStrings = []

        if self.isGlobal == True:

            x0 = '-' if self.parent == None else self.x0

            self.resultStrings.append(f'Global Fit Results:')
            self.resultStrings.append(f'Baseline: {x0}, bpConversion: {bpConversion}')

        else:

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
    
    def updateConversionParameters(self, x0):

        self.x0 = x0

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

            for i in range(7): 

                line = read_file.readline()
                reg = r"\A(\w+)[ =]*([-0-9.Ee]*)\s*\Z"
                readout = re.findall(reg, line)[0]
                var = readout[0]
                value = float(readout[1])

                headerdata[var] = value

        self.gaussWidth = headerdata['w']

        baseAmps = np.array([headerdata['Al'], headerdata['Ac'], headerdata['Ar']])   # the kernel is given by the sum of three gaussians with different amplitude and centres but shared width

        self.baseAmps = baseAmps/np.sum(baseAmps)    #normalize
        self.centres = np.array([headerdata["xl"], headerdata["xc"], headerdata["xr"]])
        
        AmpRes = minimize(self.getKernel, np.sum(self.centres * self.baseAmps), args=(-1, 0))  # find baseline peak
        x0 = AmpRes['x'][0]

        self.kernelPeakSize = -AmpRes['fun']  # amplitude peak height of a normalized kernel distribution
        self.centres -= x0  # make centres relativ to offset

        self.updateConversionParameters(x0)

    def readFileData(self):

        with open(self.file, "r") as read_file:

            data = np.loadtxt(read_file, delimiter='\t', skiprows=10)

        self.bins = data[:,0]   # positions
        self.pvalues = data[:,1]    # apparent amplitudes
        self.indeces = np.arange(len(self.bins))

        self.binsize = np.round(self.bins[1] - self.bins[0], 10)

        self.pSum = np.sum(self.pvalues)*self.binsize
        self.pvaluesNorm = self.pvalues/self.pSum
        self.Evalues = -np.log(self.pvaluesNorm)

        self.pGuesses = [1/self.pSum] * (RLoopSize + 1)  # initial amplitudes for each position (per bp of Rloop lenght), start at max Energy for each

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

    def getKernelOffset(self):  # sometimes the kernel is shifted against the energy landscape, so we do a quick fit to make sure the iitial offset is good.

        offsetResult = minimize(self.getKernelOffsetLSScore, [self.x0,1] , bounds=[(-5,5),(1E-6,10)], method='L-BFGS-B', options=fitOptions)

        self.x0 = offsetResult['x'][0]
        self.AmpKernel = offsetResult['x'][1]

        self.pGuesses[0] = self.AmpKernel
    
    def precondition(self): # Preconditioning: iteratively close in on apparent energylandscape to create good parameter guesses

        rloopBpPos = np.arange(self.x0, self.x0 + (RLoopSize+1)*bpConversion, bpConversion)
        self.rloopBpPos = rloopBpPos[0:RLoopSize+1] #ensure correct length
        
        self.bpPositionBins = []

        for pos in self.rloopBpPos:

            found = np.interp(pos, self.bins, self.indeces)

            self.bpPositionBins.append(int(round(found, 0)))

        self.pApparent = np.array(self.pvaluesNorm)
        self.guessGaussSum = np.zeros(len(self.bins))

        for i in range(2000):

            result = (self.pApparent - self.guessGaussSum)[self.bpPositionBins[1]-1:self.bpPositionBins[-1]+1]    # difference between current guess and apparent energy landscape, beginning and tail are truncated

            max_bin_index = result.argmax()+self.bpPositionBins[0]-1   # bin index, where the biggest difference is located
            
            max_pos = self.bins[max_bin_index]     # same, but in rad (pos on apparent energy landscape)

            max_pos_bp = (max_pos - self.x0)/bpConversion # same, but in bp (of Rloop length)

            frac = max_pos_bp%1

            if max_pos_bp >= rloopFitPositions - 1: # if the value to be edited is the max Rloop length's parameter, avoid running out of bounds of the amplitude array

                max_pos_bp = rloopFitPositions - 2
                frac = 1

            elif max_pos_bp < 0:

                max_pos_bp = 0
                frac = 0

            max_pos_bp = int(max_pos_bp)

            lower = max_pos_bp
            upper = max_pos_bp+1

            max_val_lower = result.max() / self.kernelPeakSize * 0.25 * (1-frac)    # kernelPeakSize is the amplitude peak height of a normalized kernel distribution
            max_val_upper = result.max() / self.kernelPeakSize * 0.25 * (frac)

            self.pGuesses[lower] += max_val_lower
            self.pGuesses[upper] += max_val_upper

            if upper == rloopFitPositions-1:

                for i in range(upper + 1, RLoopSize + 1):

                    self.pGuesses[i] = self.pGuesses[upper] * np.exp(- dE_truncation * bpConversion * (i - upper))   # update truncation penalty to mismatch tail
            
            self.pGuesses = [1-1/self.pSum if x > 1-1/self.pSum else 1/self.pSum if x < 1/self.pSum else x for x in self.pGuesses]
           
            self.guessGaussSum = self.calcLandscape(None, None, self.pGuesses)

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

            if result.max() < 10/self.pSum: 

                print(f'Preconditioning Steps {self.prefix}: {i}')

                break

        self.pGuessesInt =  np.array(self.pGuesses)[0:rloopFitPositions]
        self.guessedFreeEnergy = -np.log(self.guessGaussSum)

    def setWeights(self):

        # define weights for fitting
    
        weights = np.ones(len(self.bins))

        ZeroPos = self.bpPositionBins[0]                    # baseline
        MaxPos = self.bpPositionBins[-1]                    # minimum of rloop

        FitRangeBins = int(np.round(FitRange/self.binsize,0))

        for i, bin in enumerate(self.bins):

            if i <= (ZeroPos - FitRangeBins): # ignore anything before ~ 1.1 rad of the baseline peak

                weights[i] = 0

            elif bin < (areaofintereststart + self.x0):

                weights[i] = lowweight

            elif i > MaxPos + FitRangeBins:

                weights[i] = 0
            
            elif i > MaxPos: 

                weights[i] = lowweight

        self.weights = np.array(weights)

    def getFittingParameters(self):

        return self.x0, self.pGuessesInt, self.pSum

    def getApparentEnergies(self):
        
        rloopBpPos = np.arange(self.x0, self.x0 + (RLoopSize+1)*bpConversion, bpConversion)

        self.EApparentData = self.Evalues.copy()

        maxleft = int(np.interp(rloopBpPos[4], self.bins, range(len(self.bins))))
        maxright = int(np.interp(rloopBpPos[10], self.bins, range(len(self.bins))))

        minleft = int(np.interp(rloopBpPos[20], self.bins, range(len(self.bins))))
        minright = int(np.interp(rloopBpPos[32], self.bins, range(len(self.bins))))

        maxpos = np.argmax(self.EApparentData[maxleft:maxright]) + maxleft
        minpos = np.argmin(self.EApparentData[minleft:minright]) + minleft

        maxValue = np.mean(self.EApparentData[maxpos-2:maxpos+2])
        minValue = self.EApparentData[minpos]

        if MismatchPos > 10:

            max2left = int(np.interp(rloopBpPos[MismatchPos], self.bins, range(len(self.bins))))
            max2right = int(np.interp(rloopBpPos[MismatchPos+6], self.bins, range(len(self.bins))))

            jumpPos = int(np.interp(np.mean(rloopBpPos[MismatchPos-1:MismatchPos]), self.bins, range(len(self.bins))))

            max2pos = np.argmax(self.EApparentData[max2left:max2right]) + max2left

            max2Value = np.mean(self.EApparentData[max2pos-2:max2pos+2])

            slope = (maxValue - minValue + MismatchPenalty)/(self.bins[maxpos]-self.bins[minpos])

            self.EApparentData[jumpPos:max2pos] = (np.arange(max2pos - jumpPos) - (max2pos -jumpPos)) * slope * self.binsize + max2Value

        else:

            slope = (maxValue - minValue)/(self.bins[maxpos]-self.bins[minpos])

        self.EApparentData[0:maxpos] = (np.arange(maxpos) - maxpos) * slope * self.binsize + maxValue

    def getApparentEntropy(self, EAmps):
        
        rloopBpPos = np.arange(self.x0, self.x0 + (RLoopSize+1)*bpConversion, bpConversion)

        diffApparent = np.zeros(len(EAmps)) + 5

        for i, Efit in enumerate(EAmps):

            pos = rloopBpPos[i+1]

            Eapparent = np.interp(pos, self.bins, self.EApparentData)

            diffApparent[i] += Efit - Eapparent
        
        if (diffApparent < 0).any(): # protect from negative values to allow application of log --> shift all energies 

            diffApparent -= min(diffApparent) * 1.01

        EntropyAmps = diffApparent/np.sum(diffApparent) # normalize amplitudes to get a normalized entropy factor

        EntropyResult = -np.sum(EntropyAmps * np.log(EntropyAmps)) - np.log(RloopMatchingRange)

        return EntropyResult
    
    def getLeastSquareWithEntropy(self, Amps):

        NormAmps = Amps/np.sum(Amps)

        fullAmplitudes = addMismatchAmplitudes(RLoopSize, NormAmps)
        
        p = self.calcLandscape(self.bins, self.x0, fullAmplitudes)    ## probability Dist. of the molecule with current fitting parameters

        E = -np.log(p)  ## Energy landscape of current parameters

        SquareResult = np.sum((E-self.Evalues)**2*self.weights)   ## E, Evalues and weights are all 1D arrays of same length

        EAmps = -np.log(Amps[1:RloopMatchingRange+1])   # Amplitude in Energy space        

        # EntropyResult = CalculateEntropy(EAmps, MismatchPos) # Amplitudes without (linear) bias
        EntropyResult = self.getApparentEntropy(EAmps) if useApparentEntropy else CalculateEntropy(EAmps, MismatchPos) # Amplitudes without (linear) bias
        
        TotalResult = (SquareResult - (EntropyResult * entropyScale))        

        self.fitDataCache = {
            'SquareResult': SquareResult,
            'EntropyResult': EntropyResult,
            'TotalResult': TotalResult
        }

        return TotalResult

    def addGlobalFitResult(self, fitResult):

        self.globalFitData = FitResult(self, fitResult, self.fitResultFromGlobal, True)

        x0 = self.globalFitData.x0

        self.fullGlobalAmplitudes = addMismatchAmplitudes(RLoopSize, self.globalFitData.fitAmplitudes)

        self.fullGlobalAmplitudes = self.fullGlobalAmplitudes/np.sum(self.fullGlobalAmplitudes)
         
        self.globalFitEnergies = -np.log(np.array(self.fullGlobalAmplitudes))

        self.globalGaussFit = self.calcLandscape(self.bins, x0, self.fullGlobalAmplitudes)
        self.globalFreeEnergyFit = -np.log(self.globalGaussFit)

        # create conversions for result report

        self.bps = RloopPosArray

        self.bins_bp = (self.bins - x0)/bpConversion
        self.bps_rad = (self.bps * bpConversion) + x0

    def performLocalFit(self):

        self.updateConversionParameters(self.globalFitData.x0)

        bounds = rloopFitPositions*[(0.1/self.pSum,1-0.1/self.pSum)]

        localFitResult = minimize(self.getLeastSquareWithEntropy, self.globalFitData.fitAmplitudes, bounds=bounds, method='L-BFGS-B', options=fitOptions)

        self.localFitData = FitResult(self, localFitResult, self.fitDataCache)

        self.fullLocalAmplitudes = addMismatchAmplitudes(RLoopSize, self.localFitData.fitAmplitudes)

        self.fullLocalAmplitudes = self.fullLocalAmplitudes / np.sum(self.fullLocalAmplitudes)
         
        self.localFitEnergies = -np.log(np.array(self.fullLocalAmplitudes))

        self.localGaussFit = self.calcLandscape(self.bins, self.globalFitData.x0, self.fullLocalAmplitudes)
        self.localFreeEnergyFit = -np.log(self.localGaussFit)

    def plotData(self, ax):
        
        ax.plot(self.bins_bp, self.Evalues, '-', color = '#000000')
        ax.plot(self.bins_bp, self.guessedFreeEnergy, '-', color = '#00BB00')
        ax.plot(self.bins_bp, self.localFreeEnergyFit, '-', color = '#184ef2')
        ax.plot(self.bins_bp, self.globalFreeEnergyFit, '--', color = '#d50b0b')
        ax.bar(self.bps-0.25, self.globalFitEnergies, width=0.4, color = '#f79c92')
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

        rows = zip(self.bins, self.bins_bp, self.localFreeEnergyFit, self.Evalues, self.globalFreeEnergyFit, self.bps_rad, self.bps, self.fullLocalAmplitudes, self.localFitEnergies, self.fullGlobalAmplitudes, self.globalFitEnergies)

        with open(outPath, 'w') as outFile:
       
            wtr = csv.writer(outFile, delimiter=',', lineterminator='\n')

            for string in self.globalFitData.resultStrings:

                wtr.writerow([string])

            for string in self.localFitData.resultStrings:

                wtr.writerow([string])

            wtr.writerow([ "ang (rad)", "ang (bp)", "E fit (kT)", "E data (kT)", "global E fit (kT)", "ang (rad)", "ang (bp)", "Amplitudes (%)", "Energy (kT)", "Global Amplitudes (%)", "Global Energy (kT)"])

            for row in rows:

                wtr.writerow (row)

    def printResults(self):

        print(f'Molecule: {self.prefix}')

        for string in self.globalFitData.resultStrings:

            print(string)

        for string in self.localFitData.resultStrings:

            print(string)


    def __init__(self, file, index):

        self.file = file
        self.index = index
        self.Amplitudes = np.zeros(rloopFitPositions)

        self.readHeader()
        self.readFileData()
        self.getKernelOffset()
        self.precondition()
        self.setWeights()
        self.getApparentEnergies()

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

def getGuessesAndBounds(molecules):

    x0list = []
    x0BoundList = []
    baselines = []

    pguessList = np.zeros((rloopFitPositions-1, totalMolecules))

    pTotal = 0

    mol:MoleculeData    # type hint

    for i, mol in enumerate(molecules):

        x0, pGuesses, pSum =  mol.getFittingParameters()

        x0list.append(x0)

        baselines.append(pGuesses[0])

        pguessList[:,i] += (pGuesses[1:] * pSum)
        
        x0Bounds = (x0 - x0range/2, x0 + x0range/2)

        x0BoundList.append(x0Bounds)

        pTotal += mol.pSum

    pGuessAverage = np.average(pguessList, 1)/pTotal

    guesses = x0list + baselines + list(pGuessAverage)   # use individual baseline amplitudes for global fit

    bounds = x0BoundList + [(0.1/pTotal,1-(0.1/pTotal))]*(rloopFitPositions+totalMolecules-1)

    return guesses, bounds

# define fitting functions

def CalculateEntropy(Amps, MismatchPos):    # calculates entropy score

    if MismatchPos == -1:

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

    else:

        Amps1 = Amps[0:MismatchPos-1]
        Amps2 = Amps[MismatchPos-1:]

        arraySize1 = len(Amps1)
        arraySize2 = len(Amps2)

        biasSlope = (np.sum(Amps1[1:]-Amps1[:-1]) + np.sum(Amps2[1:]-Amps2[:-1]))/(arraySize1 + arraySize2-2) # get the average change of Energy levels in R-loop

        EAmpsNoBias1 = np.zeros(arraySize1)
        EAmpsNoBias2 = np.zeros(arraySize2)

        for i in range(arraySize1):

            EAmpsNoBias1[i] = Amps1[i] - biasSlope * i

        for i in range(arraySize2):

            EAmpsNoBias2[i] = Amps2[i] - biasSlope * i
        
        if (EAmpsNoBias1 < 0).any() or (EAmpsNoBias2 < 0).any(): # protect from negative values to allow application of log --> shift all energies 

            minvalue = min(min(EAmpsNoBias1), min(EAmpsNoBias2))

            EAmpsNoBias1 -= minvalue * 1.01
            EAmpsNoBias2 -= minvalue * 1.01

        totalsum = np.sum(EAmpsNoBias1)+np.sum(EAmpsNoBias2)

        EntropyAmps1 = EAmpsNoBias1/totalsum # normalize amplitudes to get a normalized entropy factor
        EntropyAmps2 = EAmpsNoBias2/totalsum # normalize amplitudes to get a normalized entropy factor

        EntropyResult = -np.sum(EntropyAmps1 * np.log(EntropyAmps1)) - np.log(arraySize1) - np.sum(EntropyAmps2 * np.log(EntropyAmps2)) - np.log(arraySize2)

        return EntropyResult

totalIterations = 0

def leastSquareWithEntropyGlobal(args): # Minimization function

    global globalFitDataCache
    
    RloopAmps = args[Rloop_argIndexStart:]

    SquareResult = 0
    EntropyResult = 0
    TotalSum = 0

    mol:MoleculeData    # type hint

    for i, mol in enumerate(moleculeList):

        arg_x0 = args[i]
        arg_Amp0 = args[i+Amps_argIndexStart]

        Amps = np.insert(RloopAmps, 0, arg_Amp0)

        mol.updateConversionParameters(arg_x0)
        mol.getLeastSquareWithEntropy(Amps)

        mol.fitResultFromGlobal = mol.fitDataCache
        
        SquareResult += mol.pSum * mol.fitResultFromGlobal['SquareResult']
        EntropyResult += mol.pSum * mol.fitResultFromGlobal['EntropyResult']
        TotalSum += mol.pSum

    globalFitDataCache['SquareResult'] = SquareResult / TotalSum
    globalFitDataCache['EntropyResult'] = EntropyResult / TotalSum
    globalFitDataCache['TotalResult'] = (globalFitDataCache['SquareResult'] - (globalFitDataCache['EntropyResult'] * entropyScale))

    return globalFitDataCache['TotalResult']

def fitCallback(args):
    
    global totalIterations

    totalIterations += 1

    if (totalIterations%100 != 0 and totalIterations < 1000) or totalIterations%10 != 0: return

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%H:%M:%S")

    print(f'[{timestampStr}] Global Fit Iteration {totalIterations}/{maxIterations}')
    
    # RloopAmps = args[Rloop_argIndexStart:]

    # mol:MoleculeData    # type hint    

    # showMol = 0

    # mol = moleculeList[showMol]
    # arg_x0 = args[showMol]
    # arg_Amp0 = args[showMol+Amps_argIndexStart]

    # Amps = np.insert(RloopAmps, 0, arg_Amp0)
    # NormAmps = Amps/np.sum(Amps)

    # mol.updateConversionParameters(arg_x0)

    # mol.plotTest(arg_x0, NormAmps)

# parameters to edit

test = False                    # set True for debugging (limits optimization time of fits)

constructPrefix = f"T6"
RloopTruncationLength = 0       # number of end mismatches

useApparentEntropy = True       # entropy method
MismatchPos = 17                # use -1 for no mismatch
MismatchPenalty = 4.5           # in kT, from apparent energy landscape analysis

# other global parameters

RLoopSize = 32                                              # total lenght of Rloop in bp
RloopMatchingRange = RLoopSize - RloopTruncationLength      # max pos of matching strand (bp)

RloopPosArray = np.arange(RLoopSize + 1)                    # used for plotting and results

if MismatchPos != -1: print(f'Applying {MismatchPenalty}kT mismatch at position {MismatchPos}')

maxIterations = 100 if test else 2000

dE_truncation = 2.29582 # in kT/rad: fixed penalty for end truncation mismatches of RLoop

x0range = 0.2   # allowed interval size to look for the angle offset
kernelRangeSize = 4

bpConversion = 0.5725        # initial value of bpConversion

rloopFitPositions = RloopMatchingRange + 1  # how many amplitudes to modulate, +1 to include baseline, +2 to include first mismatch

EntropyScaleValues = [0, 1, 2, 4, 6, 10, 20, 40, 60, 100, 200, 400, 600, 1000]
EntropyScaleValues = [100] if test else EntropyScaleValues

entropyScale = 0 # initial value of the scaling factor for entropy in minimization

# params for weighting

lowweight = 0.2             # weight of less important areas. important areas get weight of 1
areaofintereststart = 4.7   # (rad) Position where the important area starts
FitRange = 1.1              # extension beyond the actual rloop positions to also fit (before baseline / after full rloop)

fitOptions = {'maxcor': 5000, 'ftol': 1e-14, 'gtol': 1e-08, 'eps': 1e-08, 'maxfun': 250000, 'maxiter': maxIterations, 'maxls': 20}

## Main

root = Tk() # init GUI
root.withdraw() # we don't want a full GUI, so keep the root window from appearing

# import Energy Landscape

files = askopenfilenames(initialdir='Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data', title='Choose file') # show an "Open" dialog box and return the path to the selected file
# files = [
#     'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20190830_3_flipG.dat', 
#     'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20190830_4_flipG.dat', 
#     'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20190830_5_flipG.dat',
#     'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20200416_flipG.dat',
#     'Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/CRISPR Energy Landscape Deconvolution/Data/20200417_flipG.dat',
# ]

moleculeList = []   # list of all molecules of batch

for i, file in enumerate(files):

    mol = MoleculeData(file, i) # create new object

    moleculeList.append(mol) # add to storage


globalFitDataCache = {} # will hold detailed minimization scores
totalMolecules = len(moleculeList)  # total number of molecules in batch
Amps_argIndexStart = totalMolecules  # position of first baseline amplitude parameter
Rloop_argIndexStart = Amps_argIndexStart + totalMolecules  # position of first rloop amplitude parameter

guesses, bounds = getGuessesAndBounds(moleculeList)

plt.ion()

for entropyScale in EntropyScaleValues:

    # perform minimization

    totalIterations = 0

    fitResult = minimize(leastSquareWithEntropyGlobal, guesses, bounds=bounds, method='L-BFGS-B', options=fitOptions, callback = fitCallback)

    globalFitData = FitResult(None, fitResult, globalFitDataCache, True)

    guesses = fitResult['x']

    # prepare plots

    totalfigs = len(moleculeList) + 1
    figRows = int(np.round(np.sqrt(totalfigs), 0))
    figCols = int(np.ceil(totalfigs / figRows))

    fig, axes = plt.subplots(nrows=figRows, ncols=figCols, figsize=(16, 9), num=1, clear=True, squeeze=True)

    lowestBin = NaN
    highestBin = NaN
    binsize_bp = NaN
    
    mol:MoleculeData    # type hint

    for i, mol in enumerate(moleculeList):

        # calculate results

        mol.addGlobalFitResult(fitResult)
        mol.performLocalFit()

        # plot and write report to file

        row = int(np.floor(i/figCols))
        col = int(np.floor(i%figCols))
        ax = axes[row, col]

        mol.plotData(ax)

        # write to file

        if test != True: mol.writeReport()
        mol.printResults()

        # gather data for averaging

        lowestBin = np.nanmin((lowestBin, mol.bins_bp[0]))
        highestBin = np.nanmax((highestBin, mol.bins_bp[-1]))

        binsize_bp = np.nanmin((mol.binsize / bpConversion, binsize_bp))

    # generate new set of bins to allow averaging

    interpBins = np.arange(lowestBin+binsize_bp, highestBin-binsize_bp, binsize_bp)

    normStartindex = np.argmax(interpBins > 7)

    pData = np.zeros((len(interpBins), totalMolecules)) # orignial measured apparent energy landscapes
    pDataFit = np.zeros((len(interpBins), totalMolecules)) # global fit result
    Energies = np.zeros(((RLoopSize + 1), totalMolecules))
    
    # Aplitudes from global fit

    FullGlobalAmplitudes = addMismatchAmplitudes(RLoopSize, globalFitData.fitAmplitudes)
    FullGlobalAmplitudes = FullGlobalAmplitudes/np.sum(FullGlobalAmplitudes)
    FullGlobalEnergies = -np.log(FullGlobalAmplitudes)

    # interpolate to homogenize bin positions to allow averaging

    for i, mol in enumerate(moleculeList):

        pInterp = np.interp(interpBins, mol.bins_bp, mol.pvaluesNorm, left=NaN, right=NaN)
        pInterpNorm = pInterp/np.nansum(pInterp[normStartindex:])

        pFitInterp = np.interp(interpBins, mol.bins_bp, mol.globalGaussFit, left=NaN, right=NaN)
        pFitInterpNorm = pFitInterp/np.nansum(pFitInterp[normStartindex:])

        pData[:,i] = pInterpNorm
        pDataFit[:,i] = pFitInterpNorm
        Energies[:,i] = -np.log(mol.fullLocalAmplitudes/(np.sum(mol.fullLocalAmplitudes[1:])+FullGlobalAmplitudes[0]))

    # Average measurement data

    EData = -np.log(pData)
    pAverage = np.exp(-np.nanmean(EData, axis=1))
    pAverageSum = np.nansum(pAverage)
    EStd = np.nanstd(EData, axis=1)
    EAverageNorm = -np.log(pAverage/pAverageSum)
    
    # global fit (average to manage baseline and kernel)

    EFitAverage = np.nanmean(-np.log(pDataFit/pAverageSum), axis=1)

    # average deconvoluted Energies    
    
    EnergiesAvg = np.mean(Energies, axis=1)
    EnergiesStd = np.std(Energies, axis=1)

    # Summary plot

    row = int(np.floor(totalMolecules/figCols))
    col = int(np.floor(totalMolecules%figCols))
    ax = axes[row, col]    
    
    ax.plot(interpBins, EAverageNorm, '-', color = '#000000')
    ax.fill_between(interpBins, EAverageNorm+EStd, EAverageNorm-EStd, linestyle='--', edgecolor='#222222', facecolor='#21212185')
    ax.plot(interpBins, EFitAverage, '-', color = '#d50b0b')

    ax.bar(RloopPosArray - 0.25, FullGlobalEnergies, width=0.4, color='#d50b0b7d')
    ax.bar(RloopPosArray + 0.25, EnergiesAvg, width=0.4, yerr=EnergiesStd, color='#5f85f785', ecolor = '#0732b0')

    # '#000000'
    # '#00BB00'
    # '#184ef2'
    # '#d50b0b'
    # '#f79c92'
    # '#809efa'
    plt.draw()
    plt.pause(.2)

    if test != True:

        outPath = moleculeList[0].folder.parent / "Results" / f"summary_{constructPrefix}_{entropyScale:.2f}_deconv2.dat"

        rows = zip(interpBins * bpConversion, interpBins, EFitAverage, *EData.T, bpConversion * RloopPosArray, RloopPosArray, FullGlobalEnergies, *Energies.T)

        with open(outPath, 'w') as outFile:

            wtr = csv.writer(outFile, delimiter=',', lineterminator='\n')

            for string in globalFitData.resultStrings:

                wtr.writerow([string])

            molHeader1 = [f"E meas ({mol.prefix}) (kT)" for mol in moleculeList]
            molHeader2 = [f"Local Fit Energy ({mol.prefix}) (kT)" for mol in moleculeList]

            wtr.writerow(["ang (rad)", "ang (bp)", "E Global Fit (kT)"] + molHeader1 + ["ang (rad)", "ang (bp)", "Global Fit Energies (kT)"] + molHeader2)

            for row in rows:

                wtr.writerow (row)
        
plt.show()
input("Press Enter to continue...")
