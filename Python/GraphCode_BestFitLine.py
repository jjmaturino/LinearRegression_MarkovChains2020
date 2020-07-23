#!/usr/bin/env python
# coding: utf-8

# In[74]:


import pandas as pd

# Imports plotting functionality.
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from matplotlib import style
import seaborn as sns
from statistics import mean

import numpy as np
import scipy.stats as stats
from scipy.optimize import curve_fit 

from sklearn.linear_model import LinearRegression


# In[75]:


# Import data
#df = pd.read_csv('Big10Games.csv')
df1 = pd.read_csv('modifiedWinStreaks - good.csv')
df = pd.read_csv('finalCSV.csv')


# In[76]:


#Function for calculating probability
def computeProb (counterWon, counterGame, streak):
    if streak != 0:
        prob = counterWon / counterGame
    else: 
        prob = 1/2
    return prob


# In[77]:


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


# In[78]:


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
    


# In[79]:


#Creates the table with statistics per streak
dfp = pd.DataFrame ({'Win Streak': winStreak,
                     'Probability': probabilities})

print(dfp)
dfp.to_csv('big10Probability.csv')


# In[80]:


#Need the following to get rid of endpoints
dfp.drop(dfp.head(1).index,inplace=True)
dfp.drop(dfp.tail(1).index,inplace=True)

#Create variables for graph from sorted probability data frame
prob = dfp['Probability'] 
winStreak = dfp['Win Streak']

plt.plot(winStreak, prob)
plt.title('Win Streaks Greater than vs Probability')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()


# In[81]:


plt.bar(winStreak, prob)
plt.title('Win Streaks Greater than  vs Probability')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()


# In[82]:


plt.scatter(winStreak, prob, marker='o')
plt.title('Win Streaks vs Probability')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()

#param, param_cov = curve_fit(test, winStreak, prob) 
#print(c)


# In[87]:

#Create variable array for graphing
xs = np.array(winStreak, dtype=np.float64)
ys = np.array(prob, dtype=np.float64)

#Function to calculate the slope and y-intercept of best fit line
def best_fit_slope_and_intercept(xs,ys):
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)*mean(xs)) - mean(xs*xs)))
    b = mean(ys) - m*mean(xs)
    return m, b

#Implement functiontion to find slope and y-intercept
m, b = best_fit_slope_and_intercept(xs,ys)
print(m,b)

#Create line
regression_line = [(m*x)+b for x in xs]
regression_line = []
for x in xs:
    regression_line.append((m*x)+b)

#Plot data and best fit line
style.use('ggplot')
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, regression_line)
plt.title('Win Streaks vs Probability with Best Fit Line')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()

#Predicting the probability of winning next game given a certain win streak using best fit line.
predict_x = int(input("Enter the number for games in Win Streak: "))
predict_p = m*(predict_x) + b
print(predict_p)

