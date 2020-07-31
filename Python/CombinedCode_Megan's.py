#!/usr/bin/env python
# coding: utf-8

# In[161]:


import pandas as pd

# Imports plotting functionality.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
from statistics import mean

import numpy as np
import scipy as scipy
import scipy.stats as stats
from scipy.optimize import curve_fit 
import scipy.linalg as la
from sympy import * 


# In[162]:


# Import data
#df = pd.read_csv('Big10Games.csv')
df1 = pd.read_csv('modifiedWinStreaks - good.csv')
df = pd.read_csv('finalCSV.csv')

teamData = pd.read_csv('big10teams.csv')
initialData = pd.read_csv('big10stats_modified.csv')

teamNum = teamData['TeamNum']
teamName = teamData['TeamName']


# In[163]:


#Function for calculating probability
def computeProb (counterWon, counterGame, streak):
    if streak != 0:
        prob = counterWon / counterGame
    else: 
        prob = 1/2
    return prob


# In[164]:


#Function to calculate the slope and y-intercept of best fit line
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b


# In[165]:


#The following function will solve the least squares matrix equation (A^T)*A*x = (A^T)*b 
# where A^T is the tranpose of matrix A, x is what we are solving for to get the coeffients, 
# and b is the matrix composed of the y-values (the probabilities for eacah win streak)
def get_coeff(A,ys):
    A_t = np.transpose(A)  #Find A^T
    B = []  
    for y in ys:           #Create matrix b
        B.append([y])
    
    #solve right and left hand sides of least squares matrix equation
    RHS = np.matmul(A_t, A)
    LHS = np.matmul(A_t, B)
    
    #Solve for x
    x = np.linalg.solve(RHS, LHS)
    return x


# In[166]:


# Function to calculate the best fit probability for each win streak using linear best fit line, 
# quadratic best fit line and cubic best fit line
def calculateBestFitProbLinear(winStreak):
    probability = m*(winStreak) + b
    return probability

def calculateBestFitProbQuadratic(winStreak):
    probability = a*(winStreak*winStreak) + (b*winStreak) + c
    return probability

def calculateBestFitProbCubic(winStreak):
    probability = (a1*(winStreak*winStreak*winStreak)) + (b1*(winStreak*winStreak)) + (c1*winStreak) + d
    return probability
    


# In[167]:


# Function to calculate the sum of non-diagonal entries
def calculateNonDiagonalEntries(listi, listj):
    sumi = 0
    sumj = 0
    sumationEntry = 0
    for i in listi:
        sumi += (1 - calculateBestFitProbLinear(i))
    for j in listj:
        sumj += calculateBestFitProbLinear(j)
    sumationEntry = sumi + sumj
    return sumationEntry


# In[168]:


# Function to calculate the sum of non-diagonal entries
def calculateNonDiagonalEntries(listi, listj):
    sumi = 0
    sumj = 0
    sumationEntry = 0
    for i in listi:
        sumi += (1 - calculateBestFitProbLinear(i))
    for j in listj:
        sumj += calculateBestFitProbLinear(j)
    sumationEntry = sumi + sumj
    return sumationEntry


# In[169]:


streaks = df['Numeric Win Streak']
nextResult = df['Next Game']

# Calculate the maximum win and loss streaks
maxWinStreak = 0
maxLossStreak = 0
for i in range(len(streaks)):
    if streaks[i] > maxWinStreak:
        maxWinStreak = streaks[i]
    elif streaks[i] < maxLossStreak:
         maxLossStreak = streaks[i]

print("The maximum win streak is: %d" % maxWinStreak)
print("The maximum loss streak is %d" % maxLossStreak)


# In[170]:


probabilities = pd.Series([])
winStreak = pd.Series([])

#Calculate probability of winning next game based on win streak value
probPositionCounter = 0
for i in range(maxLossStreak, maxWinStreak + 1):
    winCounter = 0
    totalGames = 0
    for j in range(len(df)):
        if i != 0:
            if streaks[j] == i and nextResult[j] == 'W':
                winCounter += 1
                totalGames += 1
            elif streaks[j] == i and nextResult[j] == 'L':
                totalGames += 1
            elif streaks[j] == i and nextResult[j] == 'T':
                totalGames += 1
            #else streaks[j] == i and nextResult[j] == 'N':
    probabilities[probPositionCounter] = computeProb(winCounter, totalGames, i)
    winStreak[probPositionCounter] = i
    probPositionCounter += 1
    


# In[171]:


#Creates the table with statistics per streak
dfp = pd.DataFrame ({'Win Streak': winStreak,
                     'Probability': probabilities})

#print(dfp)
#dfp.to_csv('big10Probability.csv')


# In[175]:


#Need the following to get rid of endpoints
dfp.drop(dfp.head(1).index,inplace=True)
dfp.drop(dfp.tail(1).index,inplace=True)

#Create variables for graph from sorted probability data frame
prob = dfp['Probability'] 
winStreak = dfp['Win Streak']


# plt.scatter(winStreak, prob, marker='o')
# plt.title('Win Streaks vs Probability')
# plt.ylabel('Probability')
# plt.xlabel('Win Streaks')
# plt.show()

# In[176]:


#Predictions with Best Fit LINE 

#Create variable array for xs (winstreaks) and ys (probabilities)
xs = np.array(winStreak, dtype=np.float64)
ys = np.array(prob, dtype=np.float64)


# In[177]:



#Implement best fit line function to find slope and y-intercept
m, b = best_fit_slope_and_intercept(xs,ys)
print(m,b)


# #Create line
# regression_line = [(m*x)+b for x in xs]
# 
# #Plot data and best fit line
# style.use('ggplot')
# plt.scatter(xs,ys,color='#003F72')
# plt.plot(xs, regression_line)
# plt.title('Win Streaks vs Probability with Best Fit Line')
# plt.ylabel('Probability')
# plt.xlabel('Win Streaks')
# plt.show()

# In[178]:


#Predicting the probability of winning next game given a certain win streak using best fit linear line.
predict_x = int(input("Enter the number for games in Win Streak: "))
predict_p = m*(predict_x) + b
print(predict_p)


# In[179]:


#Best Fit Curve - Parabola (Quadratic : f = a*x^2 + b*x + c)

#Function to get coefficients a,b and c for each row in dataset and put into matrix 
# with a's as the first column, b's as the second column and c's as the third column
def get_matrixA(xs):
    matrixA = []
    #A = np.zeros((len(xs),3))
    for x in xs:
        a = x*x
        b = x
        c = 1 
        matrixA.append([a,b,c])
    return matrixA

#Implement above function and least squares function to get coefficients for quadratic line
A = get_matrixA(xs)
coeff = get_coeff(A,ys)

#Put coefficients into own variables
a = coeff[0]
b1 = coeff[1]
c = coeff[2]
print(a,b,c)


# #Graph Quadratic Fit Line
# parabola_line = [a*(x*x)+(b1*x)+c for x in xs]
# 
# 
# style.use('ggplot')
# plt.scatter(xs,ys,color='#003F72')
# plt.plot(xs, parabola_line)
# plt.title('Win Streaks vs Probability with Parabolic Fit Line')
# plt.ylabel('Probability')
# plt.xlabel('Win Streaks')
# plt.show()

# In[180]:


#Predicting the probability of winning next game given a certain win streak.
predict_x2 = int(input("Enter the number for games in Win Streak: "))
predict_p2 = a*(predict_x2*predict_x2) + (b1*predict_x2) + c
print(predict_p2)


# In[181]:


#Best Fit Curve - Cubic (f = ax^3 + bx^2 + cx + d)

#Function to get coefficients a,b,c and d for each row in dataset and put into matrix 
# with a's as the first column, b's as the second column, c's as the third column, and d's as fourth column
def get_cubic_matrixA(xs):
    matrixA = []
    for x in xs:
        a = x*x*x
        b = x*x
        c = x
        d = 1
        matrixA.append([a,b,c,d])
    return matrixA

#Find coefficints using above function and least squares function in quadratic block
A2 = get_cubic_matrixA(xs)
cubic_coeffs = get_coeff(A2,ys)

#Put coefficients into own variables (used a1,b1,c1 because a,b,c are used above)
a1 = cubic_coeffs[0]
b2 = cubic_coeffs[1]
c1 = cubic_coeffs[2]
d = cubic_coeffs[3]
print(a1,b1,c1,d)


# #Graph Cubic Best Fit Line
# cubic_line = [a1*(x*x*x)+(b2*(x*x))+(c1*x)+d for x in xs]
# 
# style.use('ggplot')
# plt.scatter(xs,ys,color='#003F72')
# plt.plot(xs, cubic_line)
# plt.title('Win Streaks vs Probability with Cubic Fit Line')
# plt.ylabel('Probability')
# plt.xlabel('Win Streaks')
# plt.show()

# In[182]:


#Predicting the probability of winning next game given a certain win streak.
predict_x3 = int(input("Enter the number for games in Win Streak: "))
predict_p3 = (a1*(predict_x3*predict_x3*predict_x3)) + (b2*(predict_x3*predict_x3)) + (c1*predict_x3) + d
print(predict_p3)


# In[184]:


initialTeam1 = initialData['Team1']
#print(initialTeam1)
#initialDataResult1 = initialData['Result1']
initialTeam2 = initialData['Team2']
#print(initialTeam2)
#initialResult2 = initialData['Result2']
date = initialData['Date']
ni = []
totalRXGI = []
teamNumbers = df['Team Number']
nextGameOutcome = df['Next Game']
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
    bestFitProbability[j] = calculateBestFitProbLinear(numericWinStreak[j])
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

np.savetxt('matrix.csv', transitionMatrix, delimiter=',')
print('done')
#go through lists, search, find values, create a list with streak values
#        sum = calculateNonDiagonalEntries(listi, listj)
#        nonDiagonalValue = (1/ni[a]) * sum
#        transitionMatrix[a][b] = nonDiagonalValue


# In[185]:


evalues, evectors = scipy.sparse.linalg.eigs(transitionMatrix, k=1, sigma=1)
print("The eignenvalue is ", evalues)
print("The eigenvector is:")
print(evectors)


# In[186]:


#Calculate Stready State Vector
#Solve(T-I)x=0   where T is our transition matrix, I is the idenity matrix, 
# and x is the steady state vector we are solving for.
#Create identity matrix
I = np.identity(13, dtype = float)

#Take transpose of transition matrix
T = transitionMatrix.transpose()

#Calculate T-I where T is our transposed transition matrix and I is the idenity matrix
TI = np.subtract(T,I)

#Create a matrix of zeros for the left hand side of the equation
#zeroMatrix = np.zeros([13,1], dtype = float) 
#steady_x = np.linalg.solve(TI, vectors)
#print(steady_x)

#inv_TI = np.linalg.inv(TI)

#Row Reduce T-I matrix to solve for x
#Need to start by putting TI in matrix form that sympy will accept, then use rref from sympy
M = Matrix(TI)
#M_rref = M.rref() 
v = M.nullspace()
print(v)


# In[187]:


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


# In[188]:





# In[192]:


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


# In[193]:


#Hard Coded 'True Rank' 
TrueRankNames = ['Minnesota', 'Purdue', 'Michigan', 'Illinois', 'Indiana', 'Iowa', 'Ohio State', 'Michigan State', 
                 'Maryland', 'Nebraska', 'Rutgers', 'Northwestern', 'Penn State']
TrueRank = [7,12,5,1,2,3,10,6,4,8,13,9,11]

#Calculate average between true rank and actual rank

averageDistance = 0
for i in range(len(SSV)):
    averageDistance += abs(ActualRank[i] - TrueRank[i])

averageDistance = averageDistance / len(ni)

print(averageDistance)


# In[ ]:




