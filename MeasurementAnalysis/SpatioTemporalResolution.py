import re as re
from pathlib import Path
from tkinter import Tk
from tkinter.filedialog import askopenfilename, askopenfilenames
from itertools import zip_longest

import numpy as np

import matplotlib.pyplot as plt
import matplotlib as mpl

# calculate spatio-temporal resolution
dt = 0.000252
pointsPerDecade = 25
overlappingIntervals = True

def getWindowFilteredVariance(sumarray):

    print(f'--- calculate filtered RMS ---')

    datalength = len(sumarray)

    length = np.int(np.log10(datalength/4)*pointsPerDecade)+1

    sizes = np.unique((2 * 10 ** (np.arange(length)/pointsPerDecade)).astype(int))

    steps = len(sizes)

    cumsum = np.append(0,np.cumsum(sumarray))

    avgarray = np.zeros(datalength)
    rmsData = np.ones((steps+1,3))  # window size, rms, counts
    
    rmsData[1:,0] = sizes
    rmsData[0,1] = np.std(sumarray)
    rmsData[0,2] = datalength

    print(f'0 %')

    for i, size in enumerate(sizes):

        if overlappingIntervals: 

            newlength = datalength + 1 - size

            np.subtract(cumsum[size:], cumsum[0:newlength], out=sumarray[0:newlength]) 

        else: 

            newlength = np.int(datalength/size)

            maxpos = newlength * size + 1

            np.subtract(cumsum[size:maxpos:size], cumsum[0:maxpos-size:size], out=sumarray[0:newlength])

        np.divide(sumarray[0:newlength], size, out=avgarray[0:newlength])
        
        rmsData[i+1][1] = np.std(avgarray[0:newlength])
        rmsData[i+1][2] = newlength

        if i%pointsPerDecade == 0 and i>0:

            print(f'{100*i/steps:.2f} %')

    return rmsData

# define some aliases

pi = np.pi

root = Tk()
root.withdraw()

# Test Routine

if False: 

    testdata = np.random.standard_normal(2**20)

    rms1 = getWindowFilteredVariance(testdata)

    overlappingIntervals = True

    rms2 = getWindowFilteredVariance(testdata)

    fig, ax = plt.subplots()

    ax.plot(rms1[:,0], rms1[:,1], 'b-', rms2[:,0], rms2[:,1], 'r-')
    plt.show()

startPath = Path('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data')

files = askopenfilenames(initialdir=startPath, title='Choose files', filetypes=[('MT files', '*EMCCD.dat')]) # show an "Open" dialog box and return the path to the selected file
# files = ('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/2019-08-30_3/MT/tmp_053+EMCCD.dat','Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/2019-08-30_3/MT/tmp_058+EMCCD.dat')
# files = ('Z:/GroupMembers/Dominik/Projects/CRISPR-Twister/Data/2019-08-30_3/MT/tmp_053+EMCCD.dat',)

dataFolder = Path(files[0]).parent.parent

FileDir = dataFolder

rmsData = {}

maxLength = 0

for fileno, file in enumerate(files):	
    
    print(f'Open: {file}')

    filedata = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=27, missing_values=np.nan)

    if fileno == 0: 
        timedata = np.genfromtxt(file, delimiter='\t', skip_header=1, usecols=16, max_rows=10001, missing_values=np.nan)
        dt = (timedata[10000]-timedata[1000])/9000

    # input("Press Enter to continue...")

    rmsResult = getWindowFilteredVariance(filedata)

    rmsData[fileno] = rmsResult

    maxLength = max(maxLength, (rmsResult.shape)[0])

## Combine Data

nDatasets = len(rmsData)
rmsDataColumns = nDatasets + (1 if nDatasets > 1 else 0)  # one extra column for the average

rmsCombined = np.zeros((maxLength,2 + rmsDataColumns))
countSum = np.zeros(maxLength)

if nDatasets > 1:

    for i, rmsResult in enumerate(rmsData.values()):

        length = (rmsResult.shape)[0]

        if length == maxLength:

            rmsCombined[:,0] = rmsResult[:,0]
            rmsCombined[:,1] = rmsResult[:,0] * dt

        rmsCombined[:length,2] += rmsResult[:,1] * rmsResult[:,2]
        rmsCombined[:length,3+i] += rmsResult[:,1]
        countSum[:length] += rmsResult[:,2]

    rmsCombined[:,2] = rmsCombined[:,2] / countSum

else:

    rmsCombined[:,0] = rmsData[0][:,0]
    rmsCombined[:,1] = rmsData[0][:,0] * dt
    rmsCombined[:,2] = rmsData[0][:,1]


# Find max length

Prefix = dataFolder.stem

Filenumbers = np.array([int(re.findall(r'\d+', Path(f).stem)[0]) for f in files])

header = ["Window Size (#)", "Window Size (s)", "filtered RMS (rad)"]

FirstFile = Filenumbers.min()
LastFile = Filenumbers.max()

if nDatasets < 2:

    FileName = f'{Prefix}_Resolution_{FirstFile:03}.csv'
    headerStr = ", ".join(header)
    fmt = ", ".join(["%d", "%.6f", "%.6f"])
    
else:

    FilenumberHeader = [f'filtered RMS {fileNo:03} (rad)' for fileNo in Filenumbers]
    FileName = f'{Prefix}_Resolution_{FirstFile:03}-{LastFile:03}.csv'
    headerStr = ", ".join(header + FilenumberHeader)
    fmt = ", ".join(["%d", "%.6f", "%.6f"] + (["%.6f"] * len(FilenumberHeader)))

FilePath = FileDir / FileName

with FilePath.open(mode='w') as file:

    np.savetxt(FilePath, rmsCombined, delimiter=',', header=headerStr, fmt=fmt)
