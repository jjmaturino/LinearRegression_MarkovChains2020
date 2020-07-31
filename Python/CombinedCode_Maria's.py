#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np

# Imports plotting functionality.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
from statistics import mean


# In[29]:


df = pd.read_csv('finalCSV.csv')
teamNum1 = df['Team Number']
date1 = df['Date']
numWinStreak = df['Numeric Win Streak']
nextGameOutcome = df['Next Game']

df1 = pd.read_csv('Big10Games.csv')
date2 = df1['Date']
team1Num = df1['Team1']
team2Num = df1['Team2']


# In[47]:


'''
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
'''


# In[35]:


ni = []
# Calculate total number of games played by each team, excluding last game
# Correspond to value N_i in our equation
for i in range(1, teamNum1[len(df) -1] + 1):
    counter = 0
    teamCount = 0
    while counter < len(df):
        if teamNum1[counter] == i and nextGameOutcome[counter] != 'N':
            teamCount += 1
        counter += 1
    ni.append(teamCount)
#    print("Team number %d and game counter %d" %(i, teamCount))

#print("Values for N_i:")
#print(ni)

# Initialize the transition matrix
t = []
for i in range (teamNum1[len(df)-1]):
    aux = []
    for j in range (teamNum1[len(df)-1]):
        aux.append (0.)
    t.append (aux)

# Coefficients of slope and offset of the best fit line
SLOPE = 0.03594576
OFFSET = 0.48069556

# Calculate the transition matrix coefficients
for i in range (teamNum1[len(df)-1]):
    for j in range (i+1):
#       Calculate diagonal
        if i == j:
            rxg = 0.
            for k in range (len(df1)-1):
                if team1Num[k]==i+1 or team2Num[k]==i+1:
                    date = date2[k]
                    m = 0
                    while not (date1[m]==date and teamNum1[m]==i+1):
                        m += 1
                    ws = numWinStreak[m]
                    rxg += SLOPE*ws + OFFSET    #Changed this to use functions instead
                    #rxg += calculateBestFitProb(ws)
#                    print (i, j, m, ws, rxg)
                    t[i][i] = rxg / ni[i]
#       Calculate the other coefficients (team i vs team j)
        else:
            rxgi = rxgj = rxgij = rxgji = rxg = 0.
            for k in range (len(df1)-1):
                if (team1Num[k] == i+1 or team1Num[k] == j+1) and (team2Num[k] == i+1 or team2Num[k] == j+1):
                    date = date2[k]
                    m = n = 0
                    end = False
                    while not end:
                        if date1[m] == date and (teamNum1[m] == i+1 or teamNum1[m] == j+1) and n == 0:
                            r1 = m
                            n += 1
                        elif date1[m] == date and (teamNum1[m] == i+1 or teamNum1[m] == j+1) and n == 1:
                            r2 = m
                            end = True
                        m += 1
                    if teamNum1[r1] == i+1:
                        ws = numWinStreak[r1]
                        rxgi += SLOPE*ws + OFFSET
                        #rxgi += calculateBestFitProbLinear(ws)
                        ws = numWinStreak[r2]
                        rxgj += SLOPE*ws + OFFSET
                        #rxgj += calculateBestFitProbLinear(ws)
                    else:
                        ws = numWinStreak[r2]
                        rxgi += SLOPE*ws + OFFSET
                        #rxgi += calculateBestFitProbLinear(ws)
                        ws = numWinStreak[r1]
                        rxgj += SLOPE*ws + OFFSET
                        #rxgj += calculateBestFitProbLinear(ws)
                    rxgij += rxgi + (1 - rxgj)
                    rxgji += rxgj + (1 - rxgi)
                    #print (i, j, rxgi, rxgj)
            t[i][j] = rxgij / ni[i]
            t[j][i] = rxgji / ni[j]
print ("Transition Matrix")
print ("=================")
print (t)
print ("")

#Convert t as list to matrix
t_matrix = np.asarray(t) 
print(t_matrix)


# In[43]:


#Calculate Stready State Vector
#Solve(T-I)x=0   where T is our transition matrix, I is the idenity matrix, 
# and x is the steady state vector we are solving for.
#Create identity matrix
I = np.identity(13, dtype = float)

#Take transpose of transition matrix
T = t_matrix.transpose()

#Calculate T-I where T is our transposed transition matrix and I is the idenity matrix
TI = np.subtract(T,I)


# In[44]:


def steady_state_prop(p):
    dim = p.shape[0]
    q = (p-np.eye(dim))
    ones = np.ones(dim)
    q = np.c_[q,ones]
    QTQ = np.dot(q, q.T)
    bQT = np.ones(dim)
    return np.linalg.solve(QTQ,bQT)

SSV = steady_state_prop(t_matrix)

print(SSV)


# In[45]:


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


# In[46]:


#Hard Coded 'True Rank' 
TrueRankNames = ['Minnesota', 'Purdue', 'Michigan', 'Illinois', 'Indiana', 'Iowa', 'Ohio State', 'Michigan State', 
                 'Maryland', 'Nebraska', 'Rutgers', 'Northwestern', 'Penn State']
TrueRank = [7,12,5,1,2,3,10,6,4,8,13,9,11]

#Calculate average between true rank and actual rank
#Average difference Norm: average ( |4-4|, |2-1|, |1-3|, |3-2|)=#

#Not Sure if the is correct
averageDistance = 0
for i in range(len(SSV)):
    print(ActualRank[i])
    averageDistance += abs(ActualRank[i] - TrueRank[i])

averageDistance = averageDistance / len(ni)

print(averageDistance)


# In[ ]:




