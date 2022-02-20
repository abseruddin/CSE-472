import numpy as np
import scipy.stats as stats
import math
import random
def readData():
    data = np.loadtxt(dataFileName)
    parameters = open(parameterFileName,'r')

    numberOfStates = int(parameters.readline())
    transitionMatrix = []
    mean = []
    sd = []
    for i in range(numberOfStates):
        transitionMatrix.append([float(x) for x in parameters.readline().split()])
    mean = [float(x) for x in parameters.readline().split()]
    sd = [math.sqrt(float(x)) for x in parameters.readline().split()]
    
    mean = np.asarray(mean)
    sd = np.asarray(sd)

    parameters.close()
    return data,mean,sd,transitionMatrix

def findIniialProbability(transitionMatrix):
    variables = np.asarray(transitionMatrix)
    
    for i in range(numberOfStates-1):
        variables[i][i] = transitionMatrix[i][i] - 1
    A = np.transpose(variables)
    for i in range(numberOfStates):
        A[numberOfStates-1][i] = 1
    B = np.zeros(numberOfStates)
    B[numberOfStates-1] = 1
    C = np.linalg.solve(A,B)
    # C = np.log(C)
    return C
    
def viterbi2(mean,sd,transitionMatrix,data,C):
    y = []
    for i in range(numberOfStates):
        yp = [stats.norm.pdf(x,mean[i],sd[i]) for x in data]
        yp = np.log(yp)
        y.append(yp)

    transitionMatrix = np.log(transitionMatrix)
    prevValue = []
    prevParents = []
    parents = []
    result = []
    
    for i in range(numberOfStates):
        
        prevValue.append(C[i]+y[i][0])
        prevParents.append(-1)
    
    parents.append( np.asarray(prevParents))
    m = len(data)

    for i in range(1,m):
        values = []
        stateParent = []
        for j in range(numberOfStates):
            stateValue = []
            for k in range(numberOfStates):
                v = transitionMatrix[k][j] + prevValue[k]
                stateValue.append(v)
            maxValue = max(stateValue)
            stateParent.append(np.argmax(stateValue))
            values.append(maxValue+y[j][i])
            
        
        parents.append( np.asarray(stateParent))
        prevValue = values
        
    
   
    prevP = np.argmax(prevValue)
    result = np.insert(result,0,prevP,axis=0)
    
    for i in range(m-1,0,-1):
        prevP = parents[i][prevP]
        result = np.insert(result,0,prevP,axis=0)
    return result

def checkAccuracy(outputFileName,resultFileName):
    File1 = open(outputFileName,"r")
    File2 = open(resultFileName,"r")
    Dict1 = File1.readlines()
    Dict2 = File2.readlines()
    count = 0
    for i in range(len(Dict1)):
        if Dict1[i] == Dict2[i]:
            count = count+1
    print(count)

def mainForViterbi():
    data,mean,sd,transitionMatrix = readData()
    global numberOfStates
    numberOfStates = len(transitionMatrix[0])

    C = findIniialProbability(transitionMatrix)
    C = np.log(C)
    m = len(data)

    result = viterbi2(mean,sd,transitionMatrix,data,C)

    with open("states_Viterbi_wo_learning.txt",'w') as f:
        for i in range(m):
            if result[i]==0:
                f.write('"El Nino"\n')
            else:
                f.write('"La Nina"\n')

    checkAccuracy("states_Viterbi_wo_learning.txt","./Output/states_Viterbi_wo_learning.txt")

def forward(mean,sd,transitionMatrix,y,C):
    result = []
    values = []
    m = len(y[0])

    for i in range(numberOfStates):
        values.append(C[i]*y[i][0])
    
    values = values/sum(values)
    result.append(np.asarray(values))
    
    for i in range(1,m):
        values = []
        for j in range(numberOfStates):
            v = 0
            for k in range(numberOfStates):
                v += result[i-1][k]*transitionMatrix[k][j]*y[j][i]
            values.append(v)
        values = values/sum(values)
        result.append(np.asarray(values))
    return result

def backward(mean,sd,transitionMatrix,y):
    result = []
    values = []
    m = len(y[0])
    C = np.ones(numberOfStates)
    
    result = np.zeros((m,numberOfStates))

    for i in range(numberOfStates):
        result[m-1][i] = 1
    
    for i in range(m-2,-1,-1):
        values = []
        for j in range(numberOfStates):
            v = 0
            for k in range(numberOfStates):
                v += result[i+1][k]*transitionMatrix[j][k]*y[k][i+1]
            result[i][j] = v
        result[i] = result[i]/sum(result[i])

    return result


def mainForBaum():
    data,mean,sd,transitionMatrix = readData()

    global numberOfStates
    numberOfStates = len(transitionMatrix[0])
    numberOfIteration = 0

    transitionMatrix[0][0] = random.random()
    transitionMatrix[0][1] = 1 - transitionMatrix[0][0]
    transitionMatrix[1][0] = random.random()
    transitionMatrix[1][1] = 1 - transitionMatrix[1][0]
    
    C = findIniialProbability(transitionMatrix)
    m = len(data)
    
    while True:
        numberOfIteration += 1
        #C = findIniialProbability(transitionMatrix)
        y = []
        for i in range(numberOfStates):
            yp = [stats.norm.pdf(x,mean[i],sd[i]) for x in data]
            y.append(yp)

        
        forwardResult = forward(mean,sd,transitionMatrix,y,C)
        backwardResult = backward(mean,sd,transitionMatrix,y)

        f = sum(forwardResult[m-1])
        phiStar = []
        for j in range(m):
            current = []
            for k in range(numberOfStates):
                current.append((forwardResult[j][k]*backwardResult[j][k])/f)
            current = current/sum(current)
            phiStar.append(current)
            
        phiStarStar = []
        for j in range(m-1):
            current = np.zeros((numberOfStates,numberOfStates))
            s = 0
            for k in range(numberOfStates):
                for l in range(numberOfStates):
                    current[k][l] = (forwardResult[j][k]*transitionMatrix[k][l]*y[l][j+1]*backwardResult[j+1][l])/f
                s+= sum(current[k])
            current = current/s
            phiStarStar.append(current)

        transitionNew = np.zeros((numberOfStates,numberOfStates))

        for j in range(m-1):
            for k in range(numberOfStates):
                for l in range(numberOfStates):
                    transitionNew[k][l] += phiStarStar[j][k][l]

        for k in range(numberOfStates):
            transitionNew[k] = transitionNew[k]/sum(transitionNew[k])
            
        meanNew = np.zeros(numberOfStates)
        sdNew = np.zeros(numberOfStates)

        c = np.zeros(numberOfStates)
        ## doing data.dot(phiStar)
        for j in range(m):
            for k in range(numberOfStates):
                meanNew[k] += data[j]*phiStar[j][k]
                c[k] += phiStar[j][k]
                    
        for k in range(numberOfStates):
            meanNew[k] = meanNew[k]/c[k]

        for j in range(m):
            for k in range(numberOfStates):
                sdNew[k] += (phiStar[j][k]*math.pow((data[j]-meanNew[k]),2))

        for k in range(numberOfStates):
            sdNew[k] = math.sqrt(sdNew[k]/c[k])
        
        if (mean==meanNew).all() and (sd==sdNew).all() and (transitionMatrix==transitionNew).all():
            break
        mean = meanNew
        sd = sdNew
        transitionMatrix = transitionNew
        print(sd)
        print(mean)
        print(transitionMatrix)
        print('############################################')
        
    print(numberOfIteration)

    sd = [x*x for x in sd]

    with open("parameters_learned.txt",'w') as f:
        f.write(str(numberOfStates)+'\n')
        for k in range(numberOfStates):
            for l in range(numberOfStates):
                f.write(str(round(transitionMatrix[k][l],7))+"\t")
            f.write('\n')

        for k in range(numberOfStates):
            f.write(str(round(mean[k],4))+'\t')
        f.write('\n')

        for k in range(numberOfStates):
            f.write(str(round(sd[k],6))+'\t')
        f.write('\n')

        for k in range(numberOfStates):
            f.write(str(round(C[k],2))+'\t')
        f.write('\n')

    sd = [math.sqrt(x) for x in sd]
    result = viterbi2(mean,sd,transitionMatrix,data,C)
    
    with open("states_Viterbi_after_learning.txt",'w') as f:
        for i in range(m):
            if result[i]==0:
                f.write('"El Nino"\n')
            else:
                f.write('"La Nina"\n')
    checkAccuracy("states_Viterbi_after_learning.txt","./Output/states_Viterbi_after_learning.txt")
    

dataFileName = './Input/data.txt'  
parameterFileName = './Input/parameters.txt.txt' 
mainForViterbi()
mainForBaum()

