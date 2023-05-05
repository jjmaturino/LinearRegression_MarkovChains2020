# Megan Vesta
# Created on 7/19/2020

import pandas as pd
import numpy as np
from statistics import mean
from statistics import mode

# Loading CSV files and initializing variables
#df = pd.read_csv('Pac12WinStreaks - good.csv')
#df = pd.read_csv('finalWinStreaks.csv')
df = pd.read_csv('Atlantic10WinStreaks - good.csv')
dfp = pd.read_csv('big10Probability.csv')
#teamData = pd.read_csv('Pac12Teams.csv')
#teamData = pd.read_csv('big10Teams.csv')
teamData = pd.read_csv('Atlantic10Teams.csv')
#initialData = pd.read_csv('Pac12Stats_modified.csv')
#initialData = pd.read_csv('big10Stats_modified.csv')
initialData = pd.read_csv('Atlantic10Stats_modified.csv')
prob = dfp['Probability']
winStreakProb = dfp['Win Streak']

#Need the following to get rid of endpoints
dfp.drop(dfp.head(1).index,inplace=True)
dfp.drop(dfp.tail(1).index,inplace=True)
#Create variables for graph from sorted probability data frame
prob = dfp['Probability']
winStreakProb = dfp['Win Streak']

#Sheila's code for best fit line
#Create variable array for graphing
xs = np.array(winStreakProb, dtype=np.float64)
ys = np.array(prob, dtype=np.float64)
#Best Fit Curve - Parabola (Quadratic : f = a*x^2 + b*x + c)
#Function to get coefficients a,b and c for each row in dataset and put into matrix
# with a's as the first column, b's as the second column and c's as the third colum
def get_matrixA(xs):
   matrixA = []
   #A = np.zeros((len(xs),3))
   for x in xs:
       a = x*x
       b = x
       c = 1
       matrixA.append([a,b,c])
   return matrixA
#The following function will solve the least squares matrix equation (A^T)*A*x = (A
# where A^T is the tranpose of matrix A, x is what we are solving for to get the co
# and b is the matrix composed of the y-values (the probabilities for eacah win str
def get_coeff(A,ys):
   A_t = np.transpose(A) #Find A^T
   B = []
   for y in ys: #Create matrix b
       B.append([y])

   #solve right and left hand sides of least squares matrix equation
   RHS = np.matmul(A_t, A)
   LHS = np.matmul(A_t, B)

   #Solve for x
   x = np.linalg.solve(RHS, LHS)
   return x
#Implement above functions
A = get_matrixA(xs)
coeff = get_coeff(A,ys)
#Put coefficients into own variables
a = coeff[0]
b = coeff[1]
c = coeff[2]

#predict_x2 = int(input("Enter the number for games in Win Streak: "))
#predict_p2 = a*(predict_x2**2) + (b*predict_x2) + c
#print(predict_p2)


# Function to calculate the best fit probability for each win streak
def calculateBestFitProb(winStreak):
    #print("Win Streak %d" % winStreak)
    probability = (a*(winStreak**2)) + (b*winStreak) + c
    #print("Probability %d" % probability)
    return probability

# Function to calculate the sum of non-diagonal entries
def calculateNonDiagonalEntries(listi, listj):
    sumi = 0
    sumj = 0
    sumationEntry = 0
    for i in listi:
        sumi += (1 - calculateBestFitProb(i))
    for j in listj:
        sumj += calculateBestFitProb(j)
    sumationEntry = sumi + sumj
    return sumationEntry

# More variable initialization
initialTeam1 = initialData['Team1']
#print(initialTeam1)
#initialDataResult1 = initialData['Result1']
initialTeam2 = initialData['Team2']
#print(initialTeam2)
#initialResult2 = initialData['Result2']
date = initialData['Date']
ni = []
totalRXGI = []
teamNum = teamData['TeamNumber']
teamName = teamData['TeamName']
winStreak = df['Numeric Win Streak']
teamNumbers = df['Team Number']
#nextGameOutcome = df['Next Game']
numericWinStreak = df['Numeric Win Streak']
bestFitProbability = pd.Series([])
wsDates = df['Date']
winStreak = df['Numeric Win Streak']
wsTeams = df['Team Number']
# Calculate total number of games played by each team
# Correspond to value N_i in our equation
for i in range(1, teamNumbers[len(df) -1] + 1):
    counter = 0
    teamCount = 0
    while counter < len(df):
        if teamNumbers[counter] == i:
            teamCount += 1
        counter += 1
    ni.append(teamCount)
    print("Team number %d and game counter %d" %(i, teamCount))

print("Values for N_i:")
print(ni)

# Calculate the best fit probability for each numeric streak, add to series
for j in range(len(numericWinStreak)):
    bestFitProbability[j] = calculateBestFitProb(numericWinStreak[j])
    #print(bestFitProbability[j])

# Calculate the sum of best fit probabilities for each team, ommiting last game value
# Value sum(r_x(g)^i) for diagonal entries of matrix
positionCounter = 0
for i in range(1, teamNumbers[len(df) -1] + 1):
    teamTotalProbability = 0
    while positionCounter < len(df) and teamNumbers[positionCounter] == i:
        if bestFitProbability[positionCounter] != -100:
            bestProbability = bestFitProbability[positionCounter]
            teamTotalProbability += bestProbability
            #print(teamTotalProbability)
        positionCounter += 1
    totalRXGI.append(teamTotalProbability)

print("Total Values of Team Probability (diagonals):")
print(totalRXGI)

# Creates a transition matrix of all 0, size ni x ni
transitionMatrix = np.zeros( (len(ni), len(ni)) )
print(transitionMatrix)

# Calculate and insert the diagonal values for the transition matrix
# t_ii values in the matrix
for i in range(len(totalRXGI)):
    diagonalValue = (1/ni[i]) * totalRXGI[i]
    transitionMatrix[i][i] = diagonalValue

print(transitionMatrix)

initialDataCounter = 0
entryCounter = 0
# Calculate the non-diagonal entries for the transition matrix
for a in range(len(ni)):
    for b in range(len(ni)):
        iList = []
        jList = []
        initialDataCounter = 0
        if a != b:
            print("Team %d and %d games: " % ((a + 1), (b+1))) 
            dateList = []
            while initialDataCounter < len(initialData):
                if initialTeam1[initialDataCounter] == (a + 1) and initialTeam2[initialDataCounter] == (b + 1):
                    dateList.append(date[initialDataCounter])
                    #print(date[initialDataCounter])
                elif initialTeam1[initialDataCounter] == (b + 1) and initialTeam2[initialDataCounter] == (a + 1):
                    dateList.append(date[initialDataCounter])
                    #print(date[initialDataCounter])
                initialDataCounter += 1
            print(dateList)
            position = 0
            for dates in dateList:
                while position < len(df):
                    if (wsTeams[position] == a + 1) and (wsDates[position] == dates):
                        iList.append(winStreak[position])
                        position += 1
                        break
                    position += 1
            print("Team %d and %d iList" % ((a+1),(b+1)))
            print(iList)
            position = 0
            for dates in dateList:
                while position < len(df):
                    if (wsTeams[position] == b + 1) and (wsDates[position] == dates):
                        jList.append(winStreak[position])
                        position += 1
                        break
                    position += 1
            print("Team %d and %d jList" % ((a+1),(b+1)))
            print(jList)
            sum = calculateNonDiagonalEntries(iList, jList)
            print("The sum %d" % sum)
            nonDiagonalValue = (1/ni[a]) * sum
            print('Non-diagonal value %f' % nonDiagonalValue)
            transitionMatrix[a][b] = nonDiagonalValue

print(transitionMatrix)

#w, v = np.linalg.eig(transitionMatrix)

#eigenVector = v[:,1]
#print("The eigenvector for the eigenvalue of 1: ")
#print(eigenVector)

#np.savetxt('matrix.csv', transitionMatrix, delimiter=',')
#print('done')
#go through lists, search, find values, create a list with streak values
#        sum = calculateNonDiagonalEntries(listi, listj)
#        nonDiagonalValue = (1/ni[a]) * sum
#        transitionMatrix[a][b] = nonDiagonalValue
def steady_state_prop(p):
    dim = p.shape[0]        #dim = number of rows in p
    q = (p-np.eye(dim))     #p minus matrix with all zeros except last diagonal entry is 1
    ones = np.ones(dim)     # creates matrix with ones 
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)

SSV = steady_state_prop(transitionMatrix)

print(SSV)

#Use Maria's code to match steady state vector with the teams
#Sort in descending order
#call it actual rank

steadyStateValues = pd.Series([])
SSV = pd.to_numeric(SSV)
dfe = pd.DataFrame ({'Team Number': teamNum,
                     'Team Name': teamName,
                     'Steady State Vector': SSV})
# Sort the data ftrame by the Steady State Vector column
dfs = dfe.sort_values(by=['Steady State Vector'], ascending=False)
print (dfs)
print ("")


TeamRank = dfs['Team Number']
TeamRankNames = dfs['Team Name']

ActualRank = []
for i in TeamRank:
    ActualRank.append(i)
print(ActualRank)

#Hard Coded 'True Rank' - Big 10
#TrueRankNames = ['Minnesota', 'Purdue', 'Michigan', 'Illinois', 'Indiana', 'Iowa', 'Ohio State', 'Michigan State', 
#                 'Maryland', 'Nebraska', 'Rutgers', 'Northwestern', 'Penn State']
#TrueRank = [7,12,5,1,2,3,10,6,4,8,13,9,11]

#Hard Coded "True Rank" - Pac 12
#TrueRankName = ['Stanford', 'Oregon State', 'Washington', 'UCLA', 'California', 'Arizona',
#                'Arizona State', 'Southern California', 'Oregon', 'Washington State', 'Utah']
#TrueRank = [6,5,10,7,3,1,2,8,4,11,9]

#Hard Coded "True Rank" - Atlantic 10
TrueRankName = ['Saint Louis', 'Fordham', 'George Mason', 'Richmond', 'Davidson', 'VCU'
                'George Washington', 'Rhode Island', 'Dayton', 'Saint Joseph\'s', 'St. Bonaventure', 'Massachusetts', 'La Salle']
TrueRank = [12,3,5,9,1,13,4,8,2,11,10,7,6]

#Calculate average between true rank and actual rank

averageDistance = 0
for i in range(len(ActualRank)):
    averageDistance += abs(ActualRank[i] - TrueRank[i])

averageDistance = averageDistance / len(ni)
print("Average distance between model and acutal results: %f" % averageDistance)

distanceList = []
for i in range(len(ActualRank)):
    distance = abs(ActualRank[i] - TrueRank[i])
    distanceList.append(distance)

#print(distanceList)
try:
    mode = mode(distanceList)
    print("Mode of data %s " % mode)
except:
    print("No mode possible")