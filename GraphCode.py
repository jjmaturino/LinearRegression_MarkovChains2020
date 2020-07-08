#!/usr/bin/env python
# coding: utf-8

# ## College Baseball Prediction Model

# ### Starts by calculating the win streaks for a data set of a particular season's games. Then, calculates whether or not each team won or lost their next game. Then, using this information, calculate the probability of winning the next game given a certain number of games in win streak.

# In[110]:


# Imports pandas, which lets us use Data Frames. 
import pandas as pd

# Imports plotting functionality.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# We need this for our simulations. 
import numpy as np
import scipy.stats as stats


# In[111]:


# Import data
df = pd.read_csv('Big10Games.csv')
df1 = pd.read_csv('modifiedWinStreaks - good.csv')


# .head(10) shows the first ten entries of a data frame by default. 
df1.head(10)


# In[112]:


#df['Team1'].value_counts()


# In[113]:


#df['Team2'].value_counts()


# In[114]:


#Function to compute the conditional probability

def computeProb (counterWon, counterWS, counterGame):
    pb = counterWS / counterGame
    paib = counterWon / counterGame
    if pb == 0:
        pab = 0
    else:
        pab = paib / pb

    return pab


# In[115]:


#Create new columns in csv 
Result1 = pd.Series([])     #Column for whether team 1 won or lost
Result2 = pd.Series([])     #Column for whether team 2 won or lost
WinStreak1 = pd.Series([])  #Not sure if this is needed, Megan may have created elsewhere
LossStreak1 = pd.Series([]) #Not sure if this is needed, Megan may have created elsewhere
WinStreak2 = pd.Series([])  #Not sure if this is needed, Megan may have created elsewhere
LossStreak2 = pd.Series([]) #Not sure if this is needed, Megan may have created elsewhere
NextGame1 = pd.Series([])   #Column for whether team 1 won or lost next game
NextGame2 = pd.Series([])   #Column for whether team 2 won or lost next game

#Below was in Maria's code
teamNumberS = pd.Series([]) 
teamNameS = pd.Series([])
counterWonS = pd.Series([])
counterWSS = pd.Series([])
pabS = pd.Series([])


# Note sure if we even need the below cell anymore. But it is to calculate which team won each game

# In[116]:



#Create variables for the score columns
score1 = df['Score1']   
score2 = df['Score2']

#determine game results, fill Result 1 and Result 2 column with W or L
for i in range(len(df)): 
    if score1[i] > score2[i]: 
        Result1[i]="W"
        Result2[i]="L"
    elif score1[i] < score2[i]: 
        Result1[i]="L"
        Result2[i]='W'
    elif score1[i] == score2[i]: 
        Result1[i]="T"
        Result2[i]='T'
    else: 
        Result1[i]= ""

#Add new columns with data to csv    
#df.insert(8, 'Result1', Result1)
#df.insert(9, 'Result2', Result2)


# Maria's Code to Calculate Probabilities (function is above)

# In[117]:


no_integer = True
while no_integer:
    try:
        wsValue = int(input("Enter the # of consecutive wins to be used as minimum Win Streak "))
        no_integer = False
    except:
        print ("Enter an integer value")

#determine who won Game
teamNumber = df1['Team Number']
teamName = df1['Team Name']
winStreak = df1['Numeric Win Streak']   


# In[118]:


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


# In[119]:


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


# Start Graphs!

# In[121]:


#No longer needed, got probability to work
#numStreak = df1['Numeric Win Streak']
#minStreak = int(input('Enter the minimum number of games in streak: '))
#streak = df1.loc[numStreak >= minStreak]

#Need to sort the win streaks smallest to largest to get accurate graph
sortedDFT = dft.sort_values('#Win Streak', ascending = True)
print(sortedDFT)
print()

#Create variables for graph from sorted probability data frame
prob = sortedDFT['Probability']
winStreak = sortedDFT['#Win Streak']

plt.plot(winStreak, prob)
plt.title('Win Streaks Greater than %d vs Probability'%wsValue)
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()


# In[122]:


plt.bar(winStreak, prob)
plt.title('Win Streaks Greater than %d vs Probability'%wsValue)
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()


# In[ ]:


#team1.loc[team1 == 2]


# In[ ]:




