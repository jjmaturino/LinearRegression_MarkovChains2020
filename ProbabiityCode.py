# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 03:10:04 2020

@author: mvrcc
Sample code:
    We will calculate the P(A|B) being:
        A: Probability of win after a win streak of 4 (for example)
        B: Probability of a win streak of 4 (for example)
        
        P(A|B) = P(AꓵB) / P(B)

        P(B) = (# of times with a win streak of 4) / (# of Game)
        P(AꓵB) = (@ of wins after a win streak of 4) / (# of Game)
The number of consecutive wins to obtain the probability is a parameter
entered in the program
Other calculations, following the same structure, can be entered, like
the probability of lose conditioned to a lost streak, for instance.
"""

import pandas as pd

'''
Function to compute the conditional probability
'''
def computeProb (counterWon, counterWS, counterGame):
    pb = counterWS / counterGame
    paib = counterWon / counterGame
    if pb == 0:
        pab = 0
    else:
        pab = paib / pb

    return pab

# Main Module
#Import data
df = pd.read_csv('modifiedWinStreaks - good.csv')

teamNumberS = pd.Series([]) 
teamNameS = pd.Series([])
counterWonS = pd.Series([])
counterWSS = pd.Series([])
pabS = pd.Series([])

no_integer = True
while no_integer:
    try:
        wsValue = int(input("Enter the # of consecutive wins to be used as minimum Win Streak "))
        no_integer = False
    except:
        print ("Enter an integer value")

#determine who won Game
teamNumber = df['Team Number']
teamName = df['Team Name']
winStreak = df['Numeric Win Streak']   

# Initialize counters for next team
team = -1
counterWS = 0
counterWon = 0
counterGame = 0
j = 0
for i in range(len(df)):
    if teamNumber[i] != team:
        if team != -1:
            # Calculate probabilities
            pab = computeProb (counterWon, counterWS, counterGame)
            # Stores figures for this team
            teamNumberS[j] = team
            teamNameS[j] = teamName[i-1]
            counterWonS[j] = counterWon
            counterWSS[j] = counterWS
            pabS[j] = pab
            j += 1
            # Initialize counters for next team
            counterWS = 0
            counterWon = 0
            counterGame = 0
        team = teamNumber[i]
    if winStreak[i] >= wsValue:       
        counterWS += 1
        if winStreak[i] > wsValue:
            counterWon += 1
    counterGame += 1

# Stores figures for last team
pab = computeProb (counterWon, counterWS, counterGame)
teamNumberS[j] = team
teamNameS[j] = teamName[i-1]
counterWonS[j] = counterWon
counterWSS[j] = counterWS
pabS[j] = pab

#Creates the table with statistics per team
dft = pd.DataFrame ({'Team Number': teamNumberS,
                     'Team Name': teamNameS,
                     '#Wins WS': counterWonS,
                     '#Win Streak': counterWSS,
                     'Probability': pabS})

print(dft)
dft.to_csv('big10Probability.csv')
print('done')

