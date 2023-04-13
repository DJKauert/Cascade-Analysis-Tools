import csv
import re as re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askdirectory, askopenfilenames
from itertools import zip_longest

import matplotlib.pyplot as plt
import matplotlib as mpl

import numpy as np
import math

# define some aliases

pi = np.pi

root = Tk()
root.withdraw()

# define filename patterns to distinguish filetypes

rxDiffusionMap = re.compile(r"_(\d+)\.csv$")
rxMSDMap = re.compile(r"_(MSD|Var|Count)Map\.csv$")
rxEntropy = re.compile(r"_S(\d+\.?\d*)_")
rxRate = re.compile(r"_k(\d+\.?\d*)_")

# define colormap

colors = 50
cmap = 'nipy_spectral'

# define diffusion map comparison range

minrow = 20
maxrow = 191
mincol = 9
maxcol = 21

# define plotted diffusion map positions

diffusionMapPositions = np.array([9,12,15,18,21,24,26])
diffusionMapPosOffset = -6
diffusionMapTotalPos = len(diffusionMapPositions)
diffusionMapAveragedFrames = [8, 20, 40, 100]
diffusionMapTotalAveragedFrames = len(diffusionMapAveragedFrames)

# define plotted MSD map positions

MSDMapPositions = np.array([15,18,21,24,26])
MSDMapAveragepos = np.array([15,21,26])
MSDMapAveragerange = np.arange(15,27)
MSDMapPosOffset = -7
MSDMapTotalPos = len(MSDMapPositions)
MSDMapTimeIntervals = np.array([[0, 4], [0, 10], [4,10], [4,25], [4,50], [10,25], [10,50], [0,25], [0,50]]) # indices in ms

# define dwelltime parameters

dwelltimeBinsize = 8
dwelltimeFitstart = 2
dwelltimeFitendmax = 21

dwelltimeIndexStart = 1
dwelltimeIndexEnd = -1

def calculateDiffusionMapsRMS(Map1, Map2, posIndex='None'):

    if posIndex == 'None':

        rms = ((Map1[minrow:maxrow, mincol:maxcol] - Map2[minrow:maxrow, mincol:maxcol])**2).sum()/(maxcol - mincol)

    else:

        rms = ((Map1[minrow:maxrow, posIndex] - Map2[minrow:maxrow, posIndex])**2).sum()

    return rms

def MakeSnapshotPlots(plotdata, plotdataTotal, frames, Title):

    fig, axes = plt.subplots(nrows=2, ncols=math.ceil((1+diffusionMapTotalPos)/2), sharex=True, figsize=(25, 10))

    minRange = 0

    maxRange = np.quantile(plotdataTotal, 0.5)

    levelsL = np.linspace(minRange, maxRange, num=colors+1)     ## Lines
    levelsC = np.linspace(minRange, maxRange, num=5*colors+1)    ## Color fill

    for pIndex, pos in enumerate(diffusionMapPositions):

        ax = axes[pIndex%2, math.floor(pIndex/2)]

        ax.contour(rindeces, eindeces, plotdata[pIndex,:,:], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
        ax.contourf(rindeces, eindeces, plotdata[pIndex,:,:], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange, extend='both')
            
        ax.set_title(f'Pos {pos:02}')

        if pIndex%2 == 1: 
            ax.set_xlabel('rate (1/s)')
            ax.set_xticklabels(rateValues, rotation=45)
        ax.set_xticks(rindeces)

        ax.set_ylabel('Entropy Factor')
        ax.set_yticks(eindeces)
        ax.set_yticklabels(entropyValues)

    ax = axes[diffusionMapTotalPos%2, math.floor(diffusionMapTotalPos/2)]   # number of elements is +1 to maximum index, -1 to get the last graph    

    bounds = np.linspace(minRange, maxRange, num=21)
    norm = mpl.colors.Normalize(vmin=minRange, vmax=maxRange)

    ax.contour(rindeces, eindeces, plotdataTotal, levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
    ax.contourf(rindeces, eindeces, plotdataTotal, levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
            
    ax.set_title(f'Summary')

    ax.set_xlabel('rate (1/s)')
    ax.set_xticks(rindeces)
    ax.set_xticklabels(rateValues, rotation=45)

    ax.set_ylabel('Entropy Factor')
    ax.set_yticks(eindeces)
    ax.set_yticklabels(entropyValues)

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.03, right=0.93, wspace=0.25, hspace=0.15)

    cbaxes = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, boundaries=levelsC, ticks=bounds)

    fig.suptitle(f'{Title}, dt = {frames/4} ms', fontsize=16)

    FileName = f'{prefix}_{Title}_T{frames:03}_ResultMap.png'
    FilePath = FileDir / FileName

    fig.savefig(FilePath)

    fig.clear()
    plt.close(fig)

def MakeAveragedSnapshotPlots(AbsData, NormData):

    AbsDataAvg = np.zeros(AbsData[0].shape)
    NormDataAvg = np.zeros(NormData[0].shape)

    AbsDataAvg2 = np.zeros(AbsData[0].shape)
    NormDataAvg2 = np.zeros(NormData[0].shape)

    for frameIndex, datasetAbs in enumerate(AbsData):

        datasetNorm = NormData[frameIndex]

        maxValueAbs = np.quantile(datasetAbs[np.isfinite(datasetAbs)], 0.75)
        maxValueNorm = np.quantile(datasetNorm[np.isfinite(datasetNorm)], 0.75)

        AbsDataAvg += datasetAbs / (maxValueAbs * diffusionMapTotalAveragedFrames)
        NormDataAvg += datasetNorm / (maxValueNorm * diffusionMapTotalAveragedFrames)

        AbsDataAvg2 += datasetAbs / diffusionMapTotalAveragedFrames
        NormDataAvg2 += datasetNorm / diffusionMapTotalAveragedFrames

    AbsDataAvg2 = AbsDataAvg2 / np.quantile(datasetAbs[np.isfinite(datasetAbs)], 0.75)
    NormDataAvg2 = NormDataAvg2 / np.quantile(datasetNorm[np.isfinite(datasetNorm)], 0.75)

    levelsL = np.linspace(0, 1, num=colors+1)     ## Lines
    levelsC = np.linspace(0, 1, num=5*colors+1)    ## Color fill

    # add to master fig

    mAx = mAxes[0, 0]

    mAx.set_title("DiffusionPlotAbs")

    mAx.contour(rindeces, eindeces, AbsDataAvg, levels=levelsL, linewidths=0.5, colors='k', vmin=0, vmax=1)
    mAx.contourf(rindeces, eindeces, AbsDataAvg, levels=levelsC, cmap=cmap, vmin=0, vmax=1)

    mAx.set_xlabel('rate (1/s)')
    mAx.set_xticks(rindeces)
    mAx.set_xticklabels(rateValues, rotation=45)

    mAx.set_ylabel('Entropy Factor')
    mAx.set_yticks(eindeces)
    mAx.set_yticklabels(entropyValues)

    mAx = mAxes[0, 1]

    mAx.set_title("DiffusionPlotNorm")
    
    mAx.contour(rindeces, eindeces, NormDataAvg, levels=levelsL, linewidths=0.5, colors='k', vmin=0, vmax=1)
    mAx.contourf(rindeces, eindeces, NormDataAvg, levels=levelsC, cmap=cmap, vmin=0, vmax=1)

    mAx.set_xlabel('rate (1/s)')
    mAx.set_xticks(rindeces)
    mAx.set_xticklabels(rateValues, rotation=45)

    mAx.set_ylabel('Entropy Factor')
    mAx.set_yticks(eindeces)
    mAx.set_yticklabels(entropyValues)

    mAx = mAxes[0, 2]

    mAx.set_title("DiffusionPlotNorm2")
    
    mAx.contour(rindeces, eindeces, NormDataAvg2, levels=levelsL, linewidths=0.5, colors='k', vmin=0, vmax=1)
    mAx.contourf(rindeces, eindeces, NormDataAvg2, levels=levelsC, cmap=cmap, vmin=0, vmax=1)

    mAx.set_xlabel('rate (1/s)')
    mAx.set_xticks(rindeces)
    mAx.set_xticklabels(rateValues, rotation=45)

    mAx.set_ylabel('Entropy Factor')
    mAx.set_yticks(eindeces)
    mAx.set_yticklabels(entropyValues)
            
    
    
                

def calculateMSDMapsStats(Map1, Map2, posIndex='None', posArray=np.empty(0), weighted='True'):

    indexM = measuredData['MSDMapIndex']
    indexS = simData['MSDMapIndex']

    if posIndex != 'None':

        pos = posIndex - MSDMapPosOffset

        msd = np.empty(len(MSDMapTimeIntervals))
        var = np.empty(len(MSDMapTimeIntervals))

        for timeIndex, timerange in enumerate(MSDMapTimeIntervals):

            valid = np.where((indexM>timerange[0]) & (indexM<timerange[1]))[0]
            timeindecesM = indexM[valid]

            msd1 = Map1['MSD'][valid, posIndex]
            msd2 = np.interp(timeindecesM, indexS, Map2['MSD'][:, posIndex])    # The times are alightly different, so use interpolation

            var1 = Map1['Var'][valid, posIndex]
            var2 = np.interp(timeindecesM, indexS, Map2['Var'][:, posIndex])    # The times are alightly different, so use interpolation

            msd[timeIndex] = ((msd2 - msd1)**2).mean()
            var[timeIndex] = ((var2 - var1)**2).mean()

        return msd, var

    elif len(posArray)>0:

        msd = np.empty(len(MSDMapTimeIntervals))
        var = np.empty(len(MSDMapTimeIntervals))

        for timeIndex, timerange in enumerate(MSDMapTimeIntervals):

            valid = np.where((indexM>timerange[0]) & (indexM<timerange[1]))[0]
            timeindecesM = indexM[valid]

            lenght = len(timeindecesM)

            msd1 = np.zeros(lenght)
            msd2 = np.zeros(lenght)
            var1 = np.zeros(lenght)
            var2 = np.zeros(lenght)
            totalCount1 = np.zeros(lenght) if weighted == True else 0
            totalCount2 = np.zeros(lenght) if weighted == True else 0

            for pos in posArray:

                posIndex = pos + MSDMapPosOffset

                count1 = Map1['Count'][valid, posIndex] if weighted == True else 1
                count2 = Map2['Count'][valid, posIndex] if weighted == True else 1

                msd1 += Map1['MSD'][valid, posIndex] * count1
                msd2 += np.interp(timeindecesM, indexS, Map2['MSD'][:, posIndex]) * count2

                var1 += Map1['Var'][valid, posIndex] * count1
                var2 += np.interp(timeindecesM, indexS, Map2['Var'][:, posIndex]) * count2

                totalCount1 += count1
                totalCount2 += count2
            
            if weighted == True:

                totalCount1[totalCount1==0] = 1
                totalCount2[totalCount2==0] = 1

            else:

                max(totalCount1, 1)
                max(totalCount2, 1)

            msd[timeIndex] = ((msd2/totalCount2 - msd1/totalCount1)**2).mean()
            var[timeIndex] = ((var2/totalCount2 - var1/totalCount1)**2).mean()

        return msd, var

def MakeMSDMapPlots(plotdata, plotdataTotal, plotdataTotalR, plotdataTotalRW, Title):

    for timeIndex, timeRange in enumerate(MSDMapTimeIntervals):

        mintime = timeRange[0]
        maxtime = timeRange[1]

        fig, axes = plt.subplots(nrows=2, ncols=1+math.ceil((1+MSDMapTotalPos)/2), sharex=True, figsize=(25, 10))

        data = plotdataTotal[:,:,timeIndex]

        minRange = 0
        maxRange = np.quantile(data, 0.5)

        levelsL = np.linspace(0, maxRange, num=colors+1)     ## Lines
        levelsC = np.linspace(0, maxRange, num=10*colors+1)    ## Color fill

        for pIndex, pos in enumerate(MSDMapPositions):

            ax = axes[pIndex%2, math.floor(pIndex/2)]

            ax.contour(rindeces, eindeces, plotdata[pIndex,:,:,timeIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
            ax.contourf(rindeces, eindeces, plotdata[pIndex,:,:,timeIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange, extend='both')
                
            ax.set_title(f'Pos {pos:02}')

            if pIndex%2 == 1: ax.set_xlabel('rate (1/s)')
            ax.set_xticks(rindeces)
            ax.set_xticklabels(rateValues, rotation=45)
            
            ax.set_ylabel('Entropy Factor')
            ax.set_yticks(eindeces)
            ax.set_yticklabels(entropyValues)

        ax = axes[1, math.floor(MSDMapTotalPos/2)]   # number of elements is +1 to maximum index, -1 to get the last graph    

        bounds = np.linspace(minRange, maxRange, num=21)
        norm = mpl.colors.Normalize(vmin=minRange, vmax=maxRange)

        ax.contour(rindeces, eindeces, plotdataTotal[:,:,timeIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
        ax.contourf(rindeces, eindeces, plotdataTotal[:,:,timeIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
                
        title = ', '.join(map(str, MSDMapAveragepos))
        
        ax.set_title(f'Summary ({title}, weighted)')

        ax.set_xlabel('rate (1/s)')
        ax.set_xticks(rindeces)
        ax.set_xticklabels(rateValues, rotation=45)
        
        ax.set_ylabel('Entropy Factor')
        ax.set_yticks(eindeces)
        ax.set_yticklabels(entropyValues)

        ax = axes[0, math.floor(MSDMapTotalPos/2)+1]   # number of elements is +1 to maximum index, -1 to get the last graph    

        ax.contour(rindeces, eindeces, plotdataTotalR[:,:,timeIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
        ax.contourf(rindeces, eindeces, plotdataTotalR[:,:,timeIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
        
        ax.set_title(f'Summary ({MSDMapAveragerange[0]} - {MSDMapAveragerange[-1]})')

        ax.set_xlabel('rate (1/s)')
        ax.set_xticks(rindeces)
        ax.set_xticklabels(rateValues, rotation=45)
        
        ax.set_ylabel('Entropy Factor')
        ax.set_yticks(eindeces)
        ax.set_yticklabels(entropyValues)        
        
        ax = axes[1, math.floor(MSDMapTotalPos/2)+1]   # number of elements is +1 to maximum index, -1 to get the last graph    

        ax.contour(rindeces, eindeces, plotdataTotalRW[:,:,timeIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
        ax.contourf(rindeces, eindeces, plotdataTotalRW[:,:,timeIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
        
        ax.set_title(f'Summary ({MSDMapAveragerange[0]} - {MSDMapAveragerange[-1]}, weighted)')

        ax.set_xlabel('rate (1/s)')
        ax.set_xticks(rindeces)
        ax.set_xticklabels(rateValues, rotation=45)
        
        ax.set_ylabel('Entropy Factor')
        ax.set_yticks(eindeces)
        ax.set_yticklabels(entropyValues)

        fig.subplots_adjust(top=0.90, bottom=0.06, left=0.03, right=0.93, wspace=0.25, hspace=0.15)

        cbaxes = fig.add_axes([0.95, 0.1, 0.02, 0.8])
        fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, boundaries=levelsC, ticks=bounds)

        fig.suptitle(f'{Title}, Times: {mintime} - {maxtime} ms', fontsize=16)

        FileName = f'{prefix}_{Title}_T{mintime:03}-T{maxtime:03}_ResultMap.png'
        FilePath = FileDir / FileName

        fig.savefig(FilePath)
        fig.clear()
        plt.close(fig)

        # also add to master fig

        if timeRange[0] == 4 and timeRange[1] == 50:

            title = "MSDPlot" if Title == "MSD" else "VarPlot"
            rowpos = 1 if Title == "MSD" else 0

            mAx = mAxes[1, rowpos]

            mAx.contour(rindeces, eindeces, plotdataTotalRW[:,:,timeIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
            mAx.contourf(rindeces, eindeces, plotdataTotalRW[:,:,timeIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)

            mAx.set_xlabel('rate (1/s)')
            mAx.set_xticks(rindeces)
            mAx.set_xticklabels(rateValues, rotation=45)
            
            mAx.set_ylabel('Entropy Factor')
            mAx.set_yticks(eindeces)
            mAx.set_yticklabels(entropyValues)
                    
            mAx.set_title(title)

def getDwelltimeRates(data, isMeas):

    dt = 1/3.947 if isMeas else 0.25

    cols = len(data)

    counts = np.zeros(cols)
    sumcounts = np.zeros(cols)
    decaytimes = np.zeros(cols)

    index = 0

    for col in data:

        colData = data[col]

        colCount = len(colData)
        counts[index] = colCount

        if colCount > 0:

            bins = np.arange(0, 401, dwelltimeBinsize)

            hist, bins = np.histogram(colData, bins)

            bincenters = (bins[:-1] + dwelltimeBinsize/2)*dt

            norm = hist/colCount

            survival = 1 - np.cumsum(norm)

            weight = np.sqrt(hist)

            lognorm = np.log(survival)

            validIndex = np.argmax(survival<0.5/colCount)

            fitend = min(validIndex, dwelltimeFitendmax) if validIndex > 0 else dwelltimeFitendmax

            if np.sum(hist[dwelltimeFitstart:fitend]) > 1:

                coef = np.polyfit(bincenters[dwelltimeFitstart:fitend], lognorm[dwelltimeFitstart:fitend], 1, w=weight[dwelltimeFitstart:fitend])

                decaytimes[index] = -1/(coef[0]) if coef[0] < 0 else math.inf

            else: decaytimes[index] = math.inf

        else: 

            decaytimes[index] = math.inf

        index += 1

    for i in range(len(counts)):

        index = int(i/2)*2

        sumcount = counts[index] + counts[index + 1]

        sumcounts[i] = max(sumcount, 1)

    rates = counts / sumcounts * 1000 / decaytimes

    return rates, counts

def getDwelltimeDiffScore(mRates, mCounts, sRates, sCounts):

    scores = (mRates[dwelltimeIndexStart:dwelltimeIndexEnd] - sRates[dwelltimeIndexStart:dwelltimeIndexEnd]) ** 2
    scoreSum = np.sum((mRates[dwelltimeIndexStart:dwelltimeIndexEnd] - sRates[dwelltimeIndexStart:dwelltimeIndexEnd]) ** 2 * mCounts[dwelltimeIndexStart:dwelltimeIndexEnd] * sCounts[dwelltimeIndexStart:dwelltimeIndexEnd] / (np.sum(mCounts[dwelltimeIndexStart:dwelltimeIndexEnd]) * np.sum(sCounts[dwelltimeIndexStart:dwelltimeIndexEnd])))

    return scores, scoreSum


def MakeDwelltimePlots(scores, scoreSums):

    scorePlots = scores.shape[2]

    fig, axes = plt.subplots(nrows=2, ncols=math.ceil((1+scorePlots)/2), sharex=True, figsize=(25, 10))

    minRange = 0    
    maxRange = np.quantile(scoreSums, 0.5)

    if not np.isfinite(maxRange): maxRange = 1

    header = measuredData['dwellTimeHeader']

    levelsL = np.linspace(minRange, maxRange, num=colors+1)     ## Lines
    levelsC = np.linspace(minRange, maxRange, num=10*colors+1)    ## Color fill

    for pIndex in range(scorePlots):

        ax = axes[pIndex%2, math.floor(pIndex/2)]

        ax.contour(rindeces, eindeces, scores[:,:,pIndex], levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
        ax.contourf(rindeces, eindeces, scores[:,:,pIndex], levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange, extend='both')
            
        ax.set_title(header[pIndex+dwelltimeIndexStart])

        if pIndex%2 == 1: ax.set_xlabel('rate (1/s)')
        ax.set_xticks(rindeces)
        ax.set_xticklabels(rateValues, rotation=45)
        
        ax.set_ylabel('Entropy Factor')
        ax.set_yticks(eindeces)
        ax.set_yticklabels(entropyValues)

    ax = axes[1, math.floor(scorePlots/2)]   # number of elements is +1 to maximum index, -1 to get the last graph    

    bounds = np.linspace(minRange, maxRange, num=21)
    norm = mpl.colors.Normalize(vmin=minRange, vmax=maxRange)

    ax.contour(rindeces, eindeces, scoreSums, levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
    ax.contourf(rindeces, eindeces, scoreSums, levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
    
    ax.set_title(f'Total (weighted)')

    ax.set_xlabel('rate (1/s)')
    ax.set_xticks(rindeces)
    ax.set_xticklabels(rateValues, rotation=45)
    
    ax.set_ylabel('Entropy Factor')
    ax.set_yticks(eindeces)
    ax.set_yticklabels(entropyValues) 

    fig.subplots_adjust(top=0.90, bottom=0.06, left=0.03, right=0.93, wspace=0.25, hspace=0.15)

    cbaxes = fig.add_axes([0.95, 0.1, 0.02, 0.8])
    fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cbaxes, boundaries=levelsC, ticks=bounds)

    fig.suptitle(f'Rates', fontsize=16)

    FileName = f'{prefix}_Rates_ResultMap.png'
    FilePath = FileDir / FileName

    fig.savefig(FilePath)

    fig.clear()
    plt.close(fig)

    # also add to master fig

    mAx = mAxes[1, 2]

    mAx.contour(rindeces, eindeces, scoreSums, levels=levelsL, linewidths=0.5, colors='k', vmin=minRange, vmax=maxRange)
    mAx.contourf(rindeces, eindeces, scoreSums, levels=levelsC, cmap=cmap, vmin=minRange, vmax=maxRange)
    
    mAx.set_title(f'Rates')

    mAx.set_xlabel('rate (1/s)')
    mAx.set_xticks(rindeces)
    mAx.set_xticklabels(rateValues, rotation=45)
    
    mAx.set_ylabel('Entropy Factor')
    mAx.set_yticks(eindeces)
    mAx.set_yticklabels(entropyValues) 


def readMeasSnapshotFile(path, frames):

    diffusionMapData = measuredData["diffusionMapData"]
    diffusionMapDataNorm = measuredData["diffusionMapDataNorm"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=0, missing_values=0)

        if not "snapShotHeader" in measuredData:
        
            measuredData["snapShotHeader"] = filedata[0,1:]

        index = np.transpose([filedata[1:,0]])
        rawdata = filedata[1:,1:]
        rawdatacopy = np.copy(rawdata)

        data = np.hstack((index, rawdata/max(rawdata.sum(),1)))

        for k, column in enumerate(rawdatacopy.T):

            rawdatacopy[:,k] = column/max(column.sum(),1)

        normdata = np.hstack((index, rawdatacopy))

        if frames in diffusionMapData:

            diffusionMapData[frames][entropy] = data
            diffusionMapDataNorm[frames][entropy] = normdata

        else:

            diffusionMapData[frames] = {entropy: data}
            diffusionMapDataNorm[frames] = {entropy: normdata}

def readSimSnapshotFile(path, frames):

    diffusionMapData = simData["diffusionMapData"]
    diffusionMapDataNorm = simData["diffusionMapDataNorm"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        rateRX = re.search(rxRate, path.stem)
        rate = float(rateRX[1])

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=1, missing_values=0)

        indexS = np.transpose([filedata[:,0]])
        rawdata = filedata[:,1:]
        rawdatacopy = np.copy(rawdata)

        data = np.hstack((indexS, rawdata/max(rawdata.sum(),1)))

        for k, column in enumerate(rawdatacopy.T):

            rawdatacopy[:,k] = column/max(column.sum(), 1)

        normdata = np.hstack((indexS, rawdatacopy))

        if not(frames in diffusionMapData):

            diffusionMapData[frames] = {}
            diffusionMapDataNorm[frames] = {}

        frameData = diffusionMapData[frames]
        frameDataNorm = diffusionMapDataNorm[frames]

        if not(entropy in frameData):

            frameData[entropy] = {rate: data}
            frameDataNorm[entropy] = {rate: normdata}

        else:         

            frameData[entropy][rate] = data
            frameDataNorm[entropy][rate] = normdata

def readMeasMSDMapFile(path, mapType):

    MSDMapData = measuredData["MSDMapData"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=0, missing_values=0)

        if not "MSDMapIndex" in measuredData:
        
            measuredData["MSDMapIndex"] = np.transpose(filedata[1:,1])

        if not(entropy in MSDMapData):

            MSDMapData[entropy] = {mapType: filedata[1:,2:]}

        else:         

            MSDMapData[entropy][mapType] = filedata[1:,2:]

def readSimMSDMapFile(path, mapType):

    MSDMapData = simData["MSDMapData"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        rateRX = re.search(rxRate, path.stem)
        rate = float(rateRX[1])

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=1, missing_values=0)

        if not "MSDMapIndex" in simData:
        
            simData["MSDMapIndex"] = np.transpose(filedata[1:,0])*0.25

        data = filedata[1:,2:]
        
        if not(entropy in MSDMapData): MSDMapData[entropy] = {}

        entropyData = MSDMapData[entropy]

        if not(rate in entropyData):

            entropyData[rate] = {mapType: data}

        else:         

            entropyData[rate][mapType] = data

def readMeasDwelltimeFile(path):

    dwellTimeData = measuredData["dwellTimeData"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        reader = csv.reader(file)
        header = next(reader, None)

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=1, missing_values=np.NaN)

        if not "dwellTimeHeader" in measuredData:
        
            measuredData["dwellTimeHeader"] = header

        columns = filedata.shape[1]

        if not (entropy in dwellTimeData): dwellTimeData[entropy] = {}

        for col in range(columns):

            colHeader = header[col]

            colData = filedata[:,col]

            dwellTimeData[entropy][colHeader] = colData[~np.isnan(colData)]

def readSimDwelltimeFile(path):

    dwellTimeData = simData["dwellTimeData"]
    dwellTimeDataRloop = simData["dwellTimeDataRloop"]

    with open(path, "r") as file:

        entropyRX = re.search(rxEntropy, path.stem)
        entropy = float(entropyRX[1])

        rateRX = re.search(rxRate, path.stem)
        rate = float(rateRX[1])

        reader = csv.reader(file)
        header = next(reader, None)

        filedata = np.genfromtxt(file, delimiter=', ', skip_header=1, missing_values=np.NaN)

        if len(filedata.shape) < 2 or filedata.shape[0] < 2: filedata = np.zeros((2,8))    

    rlooppath = path.parent / path.name.replace("Dwelltimes.csv", "Dwelltimes_Rloop.csv")
    
    with open(rlooppath, "r") as fileR:

        filedataR = np.genfromtxt(fileR, delimiter=', ', skip_header=1, missing_values=np.NaN)   

        if len(filedataR.shape) < 2 or filedataR.shape[0] < 2: filedataR = np.zeros((2,8))

    columns = filedata.shape[1]
    
    if not(entropy in dwellTimeData): 
        
        dwellTimeData[entropy] = {}
        dwellTimeDataRloop[entropy] = {}

    entropyData = dwellTimeData[entropy]
    entropyDataR = dwellTimeDataRloop[entropy]

    if not (rate in entropyData): 
    
        entropyData[rate] = {}
        entropyDataR[rate] = {}

    for col in range(columns):

        colHeader = header[col]

        colData = filedata[1:,col]
        colDataR = filedataR[1:,col]

        entropyData[rate][colHeader] = colData[~np.isnan(colData)]
        entropyDataR[rate][colHeader] = colDataR[~np.isnan(colDataR)] 


def readMeasurementFiles(dirPath):

    global measuredData

    measuredData = { 

        "diffusionMapData": {},
        "diffusionMapDataNorm": {},

        "MSDMapData": {},

        "dwellTimeData": {},

    }

    allFiles = dirPath.glob('*.csv')

    for path in allFiles:

        frameRX = re.search(rxDiffusionMap, path.name)

        if frameRX: 
            
            frames = int(frameRX[1])
            readMeasSnapshotFile(path, frames)

        mapTypeRX = re.search(rxMSDMap, path.name)

        if mapTypeRX:

            mapType = mapTypeRX[1]
            readMeasMSDMapFile(path, mapType)

        if path.name.endswith("_Dwelltimes.csv"):

            readMeasDwelltimeFile(path)

    measuredData["diffusionMapData"] = {k: measuredData["diffusionMapData"][k] for k in sorted(measuredData["diffusionMapData"])}
    measuredData["diffusionMapDataNorm"] = {k: measuredData["diffusionMapDataNorm"][k] for k in sorted(measuredData["diffusionMapDataNorm"])}
    measuredData["MSDMapData"] = {k: measuredData["MSDMapData"][k] for k in sorted(measuredData["MSDMapData"])}
    measuredData["dwellTimeData"] = {k: measuredData["dwellTimeData"][k] for k in sorted(measuredData["dwellTimeData"])}

    return measuredData

def readSimFiles(dirPath):

    global simData

    simData = { 

        "diffusionMapData": {},
        "diffusionMapDataNorm": {},

        "MSDMapData": {},

        "dwellTimeData": {},
        "dwellTimeDataRloop": {},

    }

    allFiles = dirPath.glob('*.csv')

    for path in allFiles:

        frameRX = re.search(rxDiffusionMap, path.name)

        if frameRX: 
            
            frames = int(frameRX[1])
            readSimSnapshotFile(path, frames)

        mapTypeRX = re.search(rxMSDMap, path.name)

        if mapTypeRX:

            mapType = mapTypeRX[1]
            readSimMSDMapFile(path, mapType)

        if path.name.endswith("_Dwelltimes.csv"):

            readSimDwelltimeFile(path)

    simData["diffusionMapData"] = {k: simData["diffusionMapData"][k] for k in sorted(simData["diffusionMapData"])}
    simData["diffusionMapDataNorm"] = {k: simData["diffusionMapDataNorm"][k] for k in sorted(simData["diffusionMapDataNorm"])}
    simData["MSDMapData"] = {k: simData["MSDMapData"][k] for k in sorted(simData["MSDMapData"])}
    simData["dwellTimeData"] = {k: simData["dwellTimeData"][k] for k in sorted(simData["dwellTimeData"])}
    simData["dwellTimeDataRloop"] = {k: simData["dwellTimeDataRloop"][k] for k in sorted(simData["dwellTimeDataRloop"])}

    return simData

def compareSnapshotFiles():

    mData = measuredData["diffusionMapData"]
    mDataNorm = measuredData["diffusionMapDataNorm"]
    sData = simData["diffusionMapData"]
    sDataNorm = simData["diffusionMapDataNorm"]

    resultMapTotalAveragingData = np.zeros((diffusionMapTotalAveragedFrames, len(entropyValues), len(rateValues)))
    resultMapNormTotalAveragingData = np.zeros((diffusionMapTotalAveragedFrames, len(entropyValues), len(rateValues)))

    fIndex = 0

    for frames in frameValues:

        frameMData = mData[frames]
        frameMDataNorm = mDataNorm[frames]
        frameSData = sData[frames]
        frameSDataNorm = sDataNorm[frames]

        resultMapPerPos = np.zeros((diffusionMapTotalPos, len(entropyValues), len(rateValues)))
        resultMapNormPerPos = np.zeros((diffusionMapTotalPos, len(entropyValues), len(rateValues)))

        resultMapTotal = np.zeros((len(entropyValues), len(rateValues)))
        resultMapNormTotal = np.zeros((len(entropyValues), len(rateValues)))

        eIndex = 0

        for entropy in entropyValues:    # measured data is per entropy (it affects mapping), Sim data is per entropy and per single stepping rate

            eMData = frameMData[entropy]
            eMDataNorm = frameMDataNorm[entropy]
            eSDataUnsorted = frameSData[entropy]
            eSDataNormUnsorted = frameSDataNorm[entropy]

            print(f'frames: {frames:03}, entropy:{entropy}')

            eSData = {k: eSDataUnsorted[k] for k in sorted(eSDataUnsorted)}
            eSDataNorm = {k: eSDataNormUnsorted[k] for k in sorted(eSDataNormUnsorted)}

            rIndex = 0

            for rate in rateValues:

                rSData = eSData[rate]
                rSDataNorm = eSDataNorm[rate]

                for pIndex, pos in enumerate(diffusionMapPositions):

                    posIndex = pos + diffusionMapPosOffset

                    resultMapPerPos[pIndex, eIndex, rIndex] = calculateDiffusionMapsRMS(eMData, rSData, posIndex) # RMS of relativ data
                    resultMapNormPerPos[pIndex, eIndex, rIndex] = calculateDiffusionMapsRMS(eMDataNorm, rSDataNorm, posIndex)


                resultMapTotal[eIndex, rIndex] = calculateDiffusionMapsRMS(eMData, rSData)
                resultMapNormTotal[eIndex, rIndex] = calculateDiffusionMapsRMS(eMDataNorm, rSDataNorm)

                rIndex += 1   

            eIndex += 1

        header = [f'{int(n):02}' for n in rateValues]
        headerStr = ", ".join(header)
        fmt = ", ".join(["%d"] * len(rateValues))

        FileName = f'{prefix}_Summary_T{frames:03}_ResultMap.csv'

        FilePath = FileDir / FileName

        with FilePath.open(mode='w') as file:    

            np.savetxt(file, resultMapTotal, delimiter=',', header=headerStr, comments='')

        FileName = f'{prefix}_Summary_T{frames:03}_ResultMapNorm.csv'

        FilePath = FileDir / FileName

        with FilePath.open(mode='w') as file:    

            np.savetxt(file, resultMapNormTotal, delimiter=',', header=headerStr, comments='')

        for pIndex, pos in enumerate(diffusionMapPositions):

            ### Write Files

            FileName = f'{prefix}_Pos{pos:03}_T{frames:03}_ResultMap.csv'

            FilePath = FileDir / FileName

            with FilePath.open(mode='w') as file:    

                np.savetxt(file, resultMapPerPos[pIndex,:,:], delimiter=',', header=headerStr, comments='')

            FileName = f'{prefix}_Pos{pos:03}_T{frames:03}_ResultMapNorm.csv'

            FilePath = FileDir / FileName

            with FilePath.open(mode='w') as file:    

                np.savetxt(file, resultMapNormPerPos[pIndex,:,:], delimiter=',', header=headerStr, comments='')

        ### Plots

        MakeSnapshotPlots(resultMapPerPos, resultMapTotal, frames, 'Absolute')
        MakeSnapshotPlots(resultMapNormPerPos, resultMapNormTotal, frames, 'Normalized')

        ### Collect Data for Averaging

        try:
            fIndex = diffusionMapAveragedFrames.index(frames)
        except ValueError:
            fIndex = -1

        if fIndex >= 0:

            resultMapTotalAveragingData[fIndex,:,:] = resultMapTotal
            resultMapNormTotalAveragingData[fIndex,:,:] = resultMapNormTotal

    MakeAveragedSnapshotPlots(resultMapTotalAveragingData, resultMapNormTotalAveragingData)

def compareMSDFiles():

    MSDMapPerPos = np.zeros((MSDMapTotalPos, len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))
    VarMapPerPos = np.zeros((MSDMapTotalPos, len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))

    MSDMapTotal = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))
    VarMapTotal = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))

    MSDMapTotalR = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))
    VarMapTotalR = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))

    MSDMapTotalRW = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))
    VarMapTotalRW = np.zeros((len(entropyValues), len(rateValues), len(MSDMapTimeIntervals)))

    eIndex = 0

    for entropy in entropyValues:    # measured data is per entropy (it affects mapping), Sim data is per entropy and per single stepping rate

        print(f'entropy:{entropy}')

        eMData = measuredData['MSDMapData'][entropy]
        eSData = simData['MSDMapData'][entropy]

        rIndex = 0

        for rate in rateValues:

            rSData = eSData[rate]

            for pIndex, pos in enumerate(MSDMapPositions):

                posIndex = pos + MSDMapPosOffset

                MSDDiff, varDiff = calculateMSDMapsStats(eMData, rSData, posIndex)

                MSDMapPerPos[pIndex, eIndex, rIndex] = MSDDiff
                VarMapPerPos[pIndex, eIndex, rIndex] = varDiff

            MSDDiffT, varDiffT = calculateMSDMapsStats(eMData, rSData, 'None', MSDMapAveragepos, 'True')

            MSDMapTotal[eIndex, rIndex] = MSDDiffT
            VarMapTotal[eIndex, rIndex] = varDiffT

            MSDDiffTR, varDiffTR = calculateMSDMapsStats(eMData, rSData, 'None', MSDMapAveragerange, 'False')

            MSDMapTotalR[eIndex, rIndex] = MSDDiffTR
            VarMapTotalR[eIndex, rIndex] = varDiffTR

            MSDDiffTRW, varDiffTRW = calculateMSDMapsStats(eMData, rSData, 'None', MSDMapAveragerange, 'True')

            MSDMapTotalRW[eIndex, rIndex] = MSDDiffTRW
            VarMapTotalRW[eIndex, rIndex] = varDiffTRW

            rIndex += 1   

        eIndex += 1

    ### Plots

    MakeMSDMapPlots(MSDMapPerPos, MSDMapTotal, MSDMapTotalR, MSDMapTotalRW, 'MSD')
    MakeMSDMapPlots(VarMapPerPos, VarMapTotal, VarMapTotalR, VarMapTotalRW, 'Var')

def compareDwelltimeFiles():

    MRateData = np.zeros((len(entropyValues), len(rateValues), 8))
    MCountData = np.zeros((len(entropyValues), len(rateValues), 8))

    SRateData = np.zeros((len(entropyValues), len(rateValues), 8))
    SCountData = np.zeros((len(entropyValues), len(rateValues), 8))

    RRateData = np.zeros((len(entropyValues), len(rateValues), 8))
    RCountData = np.zeros((len(entropyValues), len(rateValues), 8))

    Diff = np.zeros((len(entropyValues), len(rateValues), 8 - dwelltimeIndexStart + dwelltimeIndexEnd))
    DiffSum = np.zeros((len(entropyValues), len(rateValues)))

    eIndex = 0

    FileOutput = np.zeros((len(entropyValues)* len(rateValues), 26))

    rowindex = 0

    for entropy in entropyValues:    # measured data is per entropy (it affects mapping), Sim data is per entropy and per single stepping rate

        print(f'entropy:{entropy}')

        eMData = measuredData['dwellTimeData'][entropy]
        eSData = simData['dwellTimeData'][entropy]
        eRData = simData['dwellTimeDataRloop'][entropy]

        rIndex = 0

        mRates, mCounts = getDwelltimeRates(eMData, True)

        for rate in rateValues:

            rSData = eSData[rate]
            rRData = eRData[rate]

            sRates, sCounts = getDwelltimeRates(rSData, False)
            rRates, rCounts = getDwelltimeRates(rRData, False)

            diffScore, diffScoreSum = getDwelltimeDiffScore(mRates, mCounts, sRates, sCounts)

            MRateData[eIndex, rIndex] = mRates
            MCountData[eIndex, rIndex] = mCounts
            SRateData[eIndex, rIndex] = sRates
            SCountData[eIndex, rIndex] = sCounts
            RRateData[eIndex, rIndex] = rRates
            RCountData[eIndex, rIndex] = rCounts
            Diff[eIndex, rIndex, :] = diffScore
            DiffSum[eIndex, rIndex] = diffScoreSum

            FileOutput[rowindex,:] = np.array([entropy, rate] + list(mRates) + list(sRates) + list(rRates))

            rowindex += 1

            rIndex += 1   

        eIndex += 1

    # Plots

    MakeDwelltimePlots(Diff, DiffSum)

    # write File

    header = measuredData['dwellTimeHeader']

    outputHeader = ["entropy", "rate"] + [x + ' Meas' for x in header] + [x + ' Sim' for x in header] + [x + ' SimRloop' for x in header]

    outputHeaderStr = ", ".join(outputHeader)

    fmt = ", ".join(["%f"]*2 + ["%.2f"] * len(outputHeader))

    FileName = f'{prefix}_Rates_ResultMap.csv'

    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:
        
        np.savetxt(file, FileOutput, delimiter=',', header=outputHeaderStr, comments='')


## Main Program

Test = False

if Test != True:

    startPath = Path('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/')

    dirPath = Path(askdirectory(initialdir=startPath, title='Select Measurement folder'))
# Testdata
else:

    dirPath = Path('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/Test/MeasuredDiffusionMaps')

dataFolder = dirPath.parent
prefix = dataFolder.stem
simFileDir = dataFolder / 'BrownianDynamicsSimulations'

print(f'Starting Molecule: {dataFolder}')

measuredData = readMeasurementFiles(dirPath)
simData = readSimFiles(simFileDir)

FileDir = dataFolder / 'Comparison'

# get values for each covered parameter

frameValues = list(simData["diffusionMapData"].keys())
frameValues.sort()

entropyValues = list(simData["diffusionMapData"][frameValues[0]].keys())
entropyValues.sort()

eindeces = range(len(entropyValues))

rateValues = list(simData["diffusionMapData"][frameValues[0]][entropyValues[0]].keys())
rateValues.sort()

rindeces = range(len(rateValues))

maptypeValues = list(simData['MSDMapData'][entropyValues[0]][rateValues[0]].keys())
maptypeValues.sort()

# make master comparison plot

mFig, mAxes = plt.subplots(nrows=2, ncols=3, figsize=(25, 13))
mFig.subplots_adjust(top=0.90, bottom=0.1, left=0.03, right=0.93, wspace=0.20, hspace=0.25)

mAxes[1,2].axis('off')

compareSnapshotFiles()
compareMSDFiles()
compareDwelltimeFiles()

FileName = f'{prefix}_Summary_ResultMap.png'
FilePath = FileDir / FileName

mFig.savefig(FilePath)

mFig.clear()
plt.close(mFig)

# plt.show()

print(f'Finished Molecule: {dataFolder}')
