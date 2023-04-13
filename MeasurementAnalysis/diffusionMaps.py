import re as re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
from itertools import zip_longest

import numpy as np

# define some aliases

pi = np.pi

root = Tk()
root.withdraw()

startPath = Path('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data')

files = askopenfilenames(initialdir=startPath, title='Choose files', filetypes=[('MT files', '*EMCCD.dat')]) # show an "Open" dialog box and return the path to the selected file
# files = ('C:/Recent Data/2019-08-30/tmp_059+EMCCD.dat', 'C:/Recent Data/2019-08-30/tmp_060+EMCCD.dat')

dataFolder = Path(files[0]).parent.parent

deconvPath = dataFolder / 'Deconvolution'

DeconvFiles = askopenfilenames(initialdir=deconvPath, title='Choose files', filetypes=[('Deconv files', '*deconv2.dat')])

FileDir = dataFolder / 'MeasuredDiffusionMaps'

rawdata = np.array([])

for fileno, file in enumerate(files):	
    
    print(f'Opened: {file}')

    filedata = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=27, missing_values=np.nan)

    # input("Press Enter to continue...")

    rawdata = np.append(rawdata, filedata)

for deconvFile in DeconvFiles:

    with open(deconvFile, "r") as read_file:

        header = []

        header.append(read_file.readline())
        header.append(read_file.readline())
        header.append(read_file.readline())
        header.append(read_file.readline())

        entropy = float(re.findall(r"entropyfactor: (\d+\.?\d*)", header[3])[0])
        bpConversion = float(re.findall(r"bpConversion: (\d+\.?\d*)", header[1])[0])
        offset = float(re.findall(r"Baseline: (\-?\d+\.?\d*)", header[1])[0])

    Prefix = f"F3_S{entropy}"

    data = rawdata - offset
    
    RLoopSize = 32 # Total lenght of Rloop
    minMapPos = 7 # minBp to evaluate
    minRad = 7*bpConversion # pos. where to detect diffusion data
    distStep = 0.1 # max distance to determine (rad)

    radBp = np.arange(minMapPos, RLoopSize+1) * bpConversion

    def getDiffusionData(data, validDataLength, dists):

        msd = [0]

        for dist in dists:

            valid = np.where(validDataLength > dist)[0]
            diff = data[valid] - data[valid - dist]

            var = np.var(diff)

            msd.append(var)

        return msd

    def getDiffusionMaps(data, validDataLength, times):

        msdPlotData = np.zeros((len(radBp), len(times)))
        varPlotData = np.zeros((len(radBp), len(times)))
        msdPlotCountData = np.zeros((len(radBp), len(times)))

        for t, time in enumerate(times):

            valid = np.asarray((validDataLength > time).nonzero()[0])

            for bp, rad in enumerate(radBp):

                minR = rad - 0.5 * bpConversion
                maxR = rad + 0.5 * bpConversion

                diffData = data[valid-time]

                valid2 = np.asarray(((diffData >= minR) & (diffData < maxR)).nonzero()[0])

                destination = data[valid[valid2]]
                diffsq = (destination - data[valid[valid2]-time])**2

                var = 0 if len(destination) == 0 else np.var(destination)

                mean = 0 if len(diffsq) == 0 else np.mean(diffsq)
                count = len(diffsq)

                msdPlotCountData[bp][t] = count
                msdPlotData[bp][t] = mean
                varPlotData[bp][t] = var
        
        return msdPlotData, varPlotData, msdPlotCountData

    def makeTransitionMap(data, validDataLength, mapTimes):

        histCenters = np.linspace( minMapPos + distStep/2, RLoopSize - distStep/2, int((RLoopSize - minMapPos)/distStep))

        histData = {}

        for t, time in enumerate(mapTimes):

            valid = np.asarray((validDataLength > time).nonzero()[0])

            timeKey = f"{time:03}"

            currentData = {}

            for bp, rad in enumerate(radBp):

                posKey = f"{bp+minMapPos:02}"

                minR = rad - 0.5 * bpConversion
                maxR = rad + 0.5 * bpConversion

                diffData = data[valid-time]

                valid2 = np.asarray(((diffData >= minR) & (diffData < maxR)).nonzero()[0])

                result = (data[valid[valid2]])/bpConversion

                currentData[posKey] = result

            histData[timeKey] = np.zeros((RLoopSize + 1 - minMapPos, len(histCenters)), dtype=np.int32)

            for bp, rad in enumerate(radBp):

                posKey = f"{bp+minMapPos:02}"

                hist = np.histogram(currentData[posKey], bins=int((RLoopSize - minMapPos)/distStep), range=(minMapPos, RLoopSize))

                histData[timeKey][bp] = hist[0]

        return histCenters, histData

    def stack_padding(data):

        def resize(row, size):
            add = size - row.size
            new = np.append(row, [np.nan]*add)
            return new

        # find longest row length
        row_length = max(data, key=len).__len__()
        mat = np.array( [resize(row, row_length) for row in data] )

        return mat

    def getDwellTimeData(data):

        boundaryData = np.array([[8.5, 9.5, 3, 15], [14.5, 15.5, 9, 21], [20.5, 21.5, 15, 26], [25.5, 26.5, 21, 32]])*bpConversion
        startframe = np.zeros(4, dtype=int) - 1
        dwellTimeData = {
            'Rev2' : np.array([]),
            'For2' : np.array([]),
            'Rev3' : np.array([]),
            'For3' : np.array([]),
            'Rev4' : np.array([]),
            'For4' : np.array([]),
            'Rev5' : np.array([]),
            'For5' : np.array([]),
        }

        for i, value in enumerate(data):

            for k, boundaries in enumerate(boundaryData):

                if startframe[k] == -1 and value > boundaries[0] and value < boundaries[1]:

                    startframe[k] = i

                elif  startframe[k] != -1 and value < boundaries[2]: 

                    newDwellTime = i - startframe[k]
                    key = list(dwellTimeData.keys())[2*k]

                    dwellTimeData[key] = np.append(dwellTimeData[key], newDwellTime)
                    
                    startframe[k] = -1

                elif  startframe[k] != -1 and value > boundaries[3]: 

                    newDwellTime = i - startframe[k]
                    key = list(dwellTimeData.keys())[2*k+1]

                    dwellTimeData[key] = np.append(dwellTimeData[key], newDwellTime)
                    
                    startframe[k] = -1
        
        headers = list(dwellTimeData.keys())
        dwellTimes = stack_padding(dwellTimeData.values()).transpose()
        
        return headers, dwellTimes

    mapTimes = np.array([1, 2, 4, 6, 8, 12, 16, 20, 40, 60, 80, 100, 150, 200]) # in frames
    # mapTimes = np.array([2, 8, 20, 40, 100]) # in frames

    times2 = np.arange(1, 201, dtype=int)

    msdPlotData = np.zeros((len(radBp), len(times2)))
    varPlotData = np.zeros((len(radBp), len(times2)))
    msdPlotCountData = np.zeros((len(radBp), len(times2)))
    posData = np.array([])

    # determine valid data regions

    ValidDataLength = np.zeros(len(data), dtype=int)
    length = 0

    for i, value in enumerate(data):

        length = length + 1 if value > minRad else 0

        ValidDataLength[i] = length

    # Generate transistion Maps

    histCenters, histData = makeTransitionMap(data, ValidDataLength, mapTimes)

    # get position based MSD and Variance curves

    msdPlotData, varPlotData, msdPlotCountData = getDiffusionMaps(data, ValidDataLength, times2)

    dwellTimeHeaders, dwellTimes = getDwellTimeData(data)

    valid3 = np.asarray((ValidDataLength > times2[-1]).nonzero()[0])
    posData = data[valid3]

    histCentersPos = np.linspace( -3+distStep/2, RLoopSize + 3 - distStep/2, int((RLoopSize+6)/distStep))
    histDataPos = np.histogram(posData, bins=int((RLoopSize+6)/distStep), range=(-3, RLoopSize+3))[0]

    ## Write to file

    FirstFile = re.findall(r'\d+', Path(files[0]).stem)[0]
    LastFile = re.findall(r'\d+', Path(files[-1]).stem)[0]

    headerPosStr = ", ".join(dwellTimeHeaders)
    fmtPos = ", ".join(["%s"]*8)

    FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_Dwelltimes.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:    

        np.savetxt(FilePath, dwellTimes, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')

    header = ["Dist (rad)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)]
    headerStr = ", ".join(header)

    for timeKey in histData.keys():

        FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_{timeKey}.csv'
        FilePath = FileDir / FileName

        fmt = ", ".join(["%.2f"] + ["%d"] * (RLoopSize-minMapPos+1))

        with FilePath.open(mode='w') as file:

            allData = np.vstack((histCenters, histData[timeKey])).transpose()

            np.savetxt(FilePath, allData, delimiter=',', header=headerStr, fmt=fmt, comments='')

    headerPosStr = ", ".join(["Pos (bp)", "Count", "E/kT"])
    fmtPos = ", ".join(["%.2f", "%d", "%.6f"])

    FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_Edata.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allPosData = np.vstack((histCentersPos, histDataPos, -np.log(histDataPos/np.sum(histDataPos)))).transpose()

        np.savetxt(FilePath, allPosData, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')

    headerPosStr = ", ".join(["Time (Frames)", "Time (ms)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)])
    fmtPos = ", ".join(["%d", "%.3f"] + ["%.6f"] * (RLoopSize-minMapPos+1))

    FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_MSDMap.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allMSDData = np.vstack((times2, times2/3.975, msdPlotData)).transpose()

        np.savetxt(FilePath, allMSDData, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')

    headerPosStr = ", ".join(["Time (Frames)", "Time (ms)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)])
    fmtPos = ", ".join(["%d", "%.3f"] + ["%.6f"] * (RLoopSize-minMapPos+1))

    FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_CountMap.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allCountData = np.vstack((times2, times2/3.975, msdPlotCountData)).transpose()

        np.savetxt(FilePath, allCountData, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')

    headerPosStr = ", ".join(["Time (Frames)", "Time (ms)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)])
    fmtPos = ", ".join(["%d", "%.3f"] + ["%.6f"] * (RLoopSize-minMapPos+1))

    FileName = f'{Prefix}_DiffusionDataperPos_{FirstFile}-{LastFile}_VarMap.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allVarData = np.vstack((times2, times2/3.975, varPlotData)).transpose()

        np.savetxt(FilePath, allVarData, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')
