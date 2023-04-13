import matplotlib.pyplot as plt
import numpy as np
import re as re
from time import time
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tkinter.filedialog import askopenfilenames
import csv
import multiprocessing as mp
from pathlib import Path

# define some aliases

rnorm = np.random.normal
rand = np.random.rand
pi = np.pi

Pool = mp.Pool
cpuCount = mp.cpu_count()

root = Tk()
root.withdraw()

simtime = 600  # length of simulated timetrace (s)
dt = 0.01  # timestep in ms of simulation
dt2 = 0.25  # timestep in ms to save data
averaging = 17 # number of frames to average

DiffMaxTime = 50.25  # Diffusion trace length in (ms)

RLoopSize = 32  # max pos of total R-loop

minMapPos = 7 # minBp to evaluate for diff maps
distStep = 0.1 # resolution (bp) for transition maps
mapTimes = np.array([1, 2, 4, 6, 8, 12, 16, 20, 40, 60, 80, 100, 150, 200]) # in frames

# Molecule Data

Rbead = 25.67534 # Radius of AuNP (nm). Use hydrodynamic radius values from measurements
Rarc = 83.7 # radius of the arc on which AuNP moves (nm)
ktor = 4.77905
Prefix = "T6_190422"

# Temperature Data

viscosity = 0.932e-9  # (pN s / nm²) Water, 23°C
kBT = 4.1  # (pN nm)

def getDiffusionData(data, validDataLength, timeDeltas):

    msd = [0]

    for timeDelta in timeDeltas:

        valid = np.where(validDataLength > timeDelta)[0]
        diff = data[valid] - data[valid - timeDelta]

        var = np.var(diff)

        msd.append(var)

    return msd

def getDiffusionMaps(data, validDataLength, times, bpConversion):

    radBp = np.arange(minMapPos, RLoopSize+1) * bpConversion

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

def makeTransitionMap(data, validDataLength, mapTimes, bpConversion):

    radBp = np.arange(minMapPos, RLoopSize+1) * bpConversion

    histCenters = np.linspace( minMapPos + distStep/2, RLoopSize - distStep/2, int((RLoopSize - minMapPos)/distStep))

    histData = {}

    for t, time in enumerate(mapTimes):

        valid = np.asarray((validDataLength > time).nonzero()[0])

        timeKey = f"{time:03}"

        currentData = {}

        histData[timeKey] = np.zeros((RLoopSize + 1 - minMapPos, len(histCenters)), dtype=np.int32)

        for bp, rad in enumerate(radBp):

            posKey = f"{bp+minMapPos:02}"

            minR = rad - 0.5 * bpConversion
            maxR = rad + 0.5 * bpConversion

            diffData = data[valid-time]

            valid2 = np.asarray(((diffData >= minR) & (diffData < maxR)).nonzero()[0])

            result = (data[valid[valid2]])/bpConversion

            currentData[posKey] = result

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

def getDwellTimeData(data, bpConversion):

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

def getRates(krLoop, Energies):

    kForData = np.zeros(RLoopSize+1)
    kRevData = np.zeros(RLoopSize+1)

    for r in range(RLoopSize+1):

        kForData[r] = 0 if r >= RLoopSize else krLoop * np.exp(-(Energies[r + 1] - Energies[r]) / 2)
        kRevData[r] = 0 if r <= 0 else krLoop * np.exp(-(Energies[r - 1] - Energies[r]) / 2)

    return kForData, kRevData

def simRloop(arguments):    ## Main Simulation Routine

    runtime = time()

    # simulation parameters

    Energies = arguments[0]
    krLoop = arguments[1]
    bpConversion = arguments[2]
    ktor = arguments[3]
    instance = arguments[5]
    FileDir = arguments[6]

    # calculated variables

    Drloop = krLoop * (bpConversion ** 2) / 2000    # in (rad²/ms)

    totalsteps = int(simtime * 1000 / dt)  # simulation steps
    totalvalues = int(simtime * 1000 / dt2)  # total amount of saved positions

    stepsPervalue = int(dt2 / dt)   # number of steps to collect and average for each saved position

    if not dt2 / dt > 0:

        print(f"stepsPervalue is invalid: {dt2/dt}")
        quit()

    drag = 8 * pi * viscosity * (Rbead ** 3) + 6 * pi * viscosity * Rbead * (Rarc ** 2)  # drag coeeficient for AuNP system (pN nm s)
    Dbead = kBT / (drag * 1000)

    torquestep = np.sqrt(2 * Dbead * dt)  # stdev of the torsional fluctuations of the bead (rad)

    ## setup functions

    def printTimeStats(Message):

        newtime = time()

        delta = newtime - runtime

        print(f"{instance}: {Message} {delta} s")

        return newtime

    ## do brownian dynamics iteration

    ## init variables and arrays

    r = 0  # Rloop (bp)
    g = 0  # torque fluctuations (rad)
    x = 0  # Rloop + torque fluctuations (rad)

    xarray = np.array(np.zeros(totalvalues), dtype=np.float32)
    rarray = np.array(np.zeros(totalvalues), dtype=np.float32)
    garray = np.array(np.zeros(totalvalues), dtype=np.float32)
    xbuffer = np.array(np.zeros(averaging), dtype=np.float32)
    rbuffer = np.array(np.zeros(averaging), dtype=np.float32)
    gbuffer = np.array(np.zeros(averaging), dtype=np.float32)
    validDataLength = np.array(np.zeros(totalvalues), dtype=np.int32)
    tarray = dt2 * np.arange(0, totalvalues, dtype=np.float32)

    # r is in bp here !!!

    cg = ktor * Dbead / kBT * dt  # factor for torque relaxation of bead

    print(f"{instance}: k_RLoop: {int(krLoop)}, D: {Drloop}")

    halfbp = bpConversion * 0.5
    quartbp = bpConversion * 0.25

    # biasFactor = np.exp( - Energy * halfbp )
    # biasFactor2 = np.exp( - (T6slope + Energy) * halfbp)
    ktorFactor = halfbp * ktor / kBT

    runtime = printTimeStats("Finished preparations in")

    stepcount = 0
    stepcountpos = 0

    randAllocLength = 1000000   # Determine random numbers in batches to avoid too frequent memory allocation

    kForData, kRevData = getRates(krLoop/2000, Energies)

    ## simulation loop

    for i in range(totalsteps):

        j = i % randAllocLength

        if j == 0:

            stepsDNATorque = rnorm(0.0, torquestep, randAllocLength)
            randoms1 = rand(randAllocLength)

        rad = r * bpConversion  # since r has unit [bp]

        pfor = (
            dt * kForData[r] * np.exp((x - rad - quartbp) * ktorFactor)
        )  # probability of stepping forward
        prev = (
            dt * kRevData[r] * np.exp((rad - x - quartbp) * ktorFactor)
        )  # probability of stepping backward

        randvalue = randoms1[j]

        dr = -1 if randvalue < prev else 1 if (1 - pfor) < randvalue else 0

        stepcount += 1 if dr != 0 and r > 0 else 0
        stepcountpos += 1 if r > 0 else 0

        g += -g * cg  # perform relaxation due to torque
        x += (rad - x) * cg  # perform relaxation for full model

        gstep = stepsDNATorque[j]  # torsional fluctuation due to brownian motion

        r += dr
        g += gstep  # brownian motion
        x += gstep

        istep = i % averaging

        xbuffer[istep] = x
        gbuffer[istep] = g
        rbuffer[istep] = r

        if istep == 0 and i>0:  # don't record every position, instead integrate over a time period, just like a camera. Also reduces memory consumption.

            k = int(i / stepsPervalue) - 1

            xarray[k] = xbuffer.mean() # save values
            garray[k] = gbuffer.mean()
            rarray[k] = rbuffer.mean() * bpConversion

            validDataLength[k] = validDataLength[k - 1] + 1 if x > 4 and k > 0 else 0

    print(
        f"{instance}: Effective stepping rate({stepcountpos}): {1000*stepcount/(stepcountpos * dt):.0f}"
    )

    runtime = printTimeStats("Finished simulating timetrace in")

    ## calculate diffusion parameters

    disttimes = list(
        np.arange(dt2, DiffMaxTime, dt2)
    )   # Make a list of time values over which to check Diffusion
    times = np.arange(1, len(disttimes) + 1, dtype=np.int32)    # convert time into indices

    msd = getDiffusionData(xarray, validDataLength, times)
    gvars = getDiffusionData(garray, validDataLength, times)
    rvars = getDiffusionData(rarray, validDataLength, times)

    # Calculate Diffusion Maps

    msdPlotData, varPlotData, msdPlotCountData = getDiffusionMaps(xarray, validDataLength, times, bpConversion)

    # Calculate Dwelltimes 

    dwellTimeHeaders, dwellTimes = getDwellTimeData(xarray, bpConversion)
    dwellTimeHeadersRloop, dwellTimesRloop = getDwellTimeData(rarray, bpConversion)

    sumvars = np.add(rvars, gvars)

    disttimes.insert(0, 0)

    diffInfinity = [
        2 * Drloop * t for t in disttimes
    ]  # plot line for free diffusion with Rloop constant
    
    histCenters, histData = makeTransitionMap(xarray, validDataLength, mapTimes, bpConversion)

    runtime = printTimeStats("Finished calculating diffusion data in")

    ## Write to files

    FileParams = f"{Prefix}_k{krLoop:.2f}_ktor{ktor:.2f}_S{instance}"

    headerPosStr = ", ".join(dwellTimeHeaders)
    fmtPos = ", ".join(["%s"]*8)

    FileName = f'{FileParams}_Dwelltimes.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:    

        np.savetxt(FilePath, dwellTimes, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')

    headerPosStr = ", ".join(dwellTimeHeadersRloop)
    fmtPos = ", ".join(["%s"]*8)

    FileName = f'{FileParams}_Dwelltimes_Rloop.csv'
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:    

        np.savetxt(FilePath, dwellTimesRloop, delimiter=',', header=headerPosStr, fmt=fmtPos, comments='')
    
    header = ["Dist (rad)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)]
    headerStr = ", ".join(header)

    for timeKey in histData.keys():

        FileName = f"{FileParams}_DiffMap_{timeKey}.csv"
        FilePath = FileDir / FileName

        fmt = ", ".join(["%.2f"] + ["%d"] * (RLoopSize-minMapPos+1))

        with FilePath.open(mode='w') as file:

            allData = np.vstack((histCenters, histData[timeKey])).transpose()

            np.savetxt(FilePath, allData, delimiter=',', header=headerStr, fmt=fmt, comments='')


    FileName = f"{FileParams}_BDStrace.csv"
    FilePath = FileDir / FileName

    with FilePath.open(mode="w") as file:

        rows1 = zip(tarray, xarray, rarray, garray)

        wtr = csv.writer(file, delimiter=",", lineterminator="\n")

        wtr.writerow(["t (ms)", "Total (rad)", "Rloop (rad)", "Baseline (rad)"])

        for i, row in enumerate(rows1):

            wtr.writerow(row)

    FileName = f"{FileParams}_BDS.csv"
    FilePath = FileDir / FileName

    with FilePath.open(mode="w") as file:

        rows2 = zip(disttimes, msd, rvars, gvars, sumvars, diffInfinity)

        wtr = csv.writer(file, delimiter=",", lineterminator="\n")

        wtr.writerow([f"Item = {Prefix}"])
        wtr.writerow([f"k_RLoop = {int(krLoop)}"])
        wtr.writerow([f"eff. k_RLoop = {1000*stepcount/(stepcountpos * dt):.0f}"])
        wtr.writerow([f"dt = {dt}"])
        wtr.writerow([f"ktor = {ktor}"])
        wtr.writerow([f"S = {instance}"])
        wtr.writerow(
            [
                "t (ms)",
                "Total (rad^2)",
                "Rloop (rad^2)",
                "Baseline (rad^2)",
                "Sum (rad^2)",
                "Linear (rad^2)",
                "Mean (rad^2)",
            ]
        )

        for row in rows2:

            wtr.writerow(row)

    headerParam = f"Item = {Prefix}, k_RLoop = {int(krLoop)}, eff. k_RLoop = {1000*stepcount/(stepcountpos * dt):.0f}, ktor = {ktor}, S = {instance}"
    headerPosStr = ", ".join(["Time (Frames)", "Time (ms)"] + [f'{int(n):02}' for n in range(minMapPos, RLoopSize + 1)])
    headerFull = "\n".join([headerParam, headerPosStr])
    fmtPos = ", ".join(["%d", "%.3f"] + ["%.6f"] * (RLoopSize-minMapPos+1))

    FileName = f"{FileParams}_MSDMap.csv"
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allMSDData = np.vstack((times, times/3.975, msdPlotData)).transpose()

        np.savetxt(FilePath, allMSDData, delimiter=',', header=headerFull, fmt=fmtPos, comments='')

    fmtPos = ", ".join(["%d", "%.3f"] + ["%d"] * (RLoopSize-minMapPos+1))

    FileName = f"{FileParams}_CountMap.csv"
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allCountData = np.vstack((times, times/3.975, msdPlotCountData)).transpose()

        np.savetxt(FilePath, allCountData, delimiter=',', header=headerFull, fmt=fmtPos, comments='')

    fmtPos = ", ".join(["%d", "%.3f"] + ["%.6f"] * (RLoopSize-minMapPos+1))

    FileName = f"{FileParams}_VarMap.csv"
    FilePath = FileDir / FileName

    with FilePath.open(mode='w') as file:

        allVarData = np.vstack((times, times/3.975, varPlotData)).transpose()

        np.savetxt(FilePath, allVarData, delimiter=',', header=headerFull, fmt=fmtPos, comments='')

    runtime = printTimeStats("Finished writing files")

def main():

    startPath = Path(
        "C:/"
    )

    files = askopenfilenames(
        initialdir=startPath,
        title="Choose files",
        filetypes=[("Deconvolution", "*.dat")],
    )  # show an "Open" dialog box and return the path to the selected file

    filedata = []
    EntropyData = []
    bpConvData = []

    FileDir = Path(files[0]).parent.parent / 'BrownianDynamicsSimulations'

    for file in files:

        print("Opened: " + file)

        header = []

        with open(file, "r") as read_file:

            header.append(read_file.readline())
            header.append(read_file.readline())
            header.append(read_file.readline())
            header.append(read_file.readline())

        entropy = float(re.findall(r"entropyfactor: (\d+\.?\d*)", header[3])[0])
        bpConversionFile = float(re.findall(r"bpConversion: (\d+\.?\d*)", header[1])[0])

        data = np.genfromtxt(
            file,
            delimiter=",",
            skip_header=8,
            usecols=8,
            max_rows=33,
            missing_values=-1,
        )

        data[0] = data[1]  # remove binding/dissociation rate

        filedata.append(data)
        EntropyData.append(entropy)
        bpConvData.append(bpConversionFile)

    i = 0

    args = []

    # dt = 0.01
    # Energy = filedata
    # krLoop = 2000

    # simRloop([filedata[0], 2000, dt, ktor, 0, 1, FileDir])

    for krLoop in [200, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000, 12000, 14000, 16000, 20000, 30000]:    ## single bp stepping rates to simulate

        for i, Energy in enumerate(filedata):

            args.append([Energy, krLoop, bpConvData[i], ktor, 0, EntropyData[i], FileDir])

    calcpool = Pool(processes=(cpuCount - 1))  # Create a multiprocessing Pool
    calcpool.map(simRloop, args)  # process data_inputs iterable with pool

if __name__ == "__main__":

    main()

