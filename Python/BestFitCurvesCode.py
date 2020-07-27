#!/usr/bin/env python
# coding: utf-8

# In[166]:


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


# In[155]:


# Import data
#df = pd.read_csv('Big10Games.csv')
df1 = pd.read_csv('modifiedWinStreaks - good.csv')
df = pd.read_csv('finalCSV.csv')


# In[156]:


#Function for calculating probability
def computeProb (counterWon, counterGame, streak):
    if streak != 0:
        prob = counterWon / counterGame
    else: 
        prob = 1/2
    return prob


# In[157]:


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


# In[158]:


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
    


# In[159]:


#Creates the table with statistics per streak
dfp = pd.DataFrame ({'Win Streak': winStreak,
                     'Probability': probabilities})

print(dfp)
dfp.to_csv('big10Probability.csv')


# In[160]:


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


# In[161]:


plt.bar(winStreak, prob)
plt.title('Win Streaks Greater than  vs Probability')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()


# In[162]:


plt.scatter(winStreak, prob, marker='o')
plt.title('Win Streaks vs Probability')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()

#param, param_cov = curve_fit(test, winStreak, prob) 
#print(c)


# In[163]:


#Predictions with Best Fit LINE - taken from the following article: 
#https://pythonprogramming.net/how-to-program-best-fit-line-machine-learning-tutorial/

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


# In[172]:


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

#Implement above functions
A = get_matrixA(xs)
coeff = get_coeff(A,ys)

#Put coefficients into own variables
a = coeff[0]
b = coeff[1]
c = coeff[2]

parabola_line = [a*(x*x)+(b*x)+c for x in xs]
parabola_line = []
for x in xs:
    parabola_line.append((a*(x*x)+(b*x)+c))

style.use('ggplot')
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, parabola_line)
plt.title('Win Streaks vs Probability with Parabolic Fit Line')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()

#Predicting the probability of winning next game given a certain win streak.
predict_x2 = int(input("Enter the number for games in Win Streak: "))
predict_p2 = a*(predict_x2*predict_x2) + (b*predict_x2) + c
print(predict_p2)


# In[179]:


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
print(cubic_coeffs)

#Put coefficients into own variables (used a1,b1,c1 because a,b,c are used above)
a1 = cubic_coeffs[0]
b1 = cubic_coeffs[1]
c1 = cubic_coeffs[2]
d = cubic_coeffs[3]
print(a1)
print(b1)
print(c1)
print(d)

#cubic_line = [a*(x*x*x)+(b*(x*x))+(c*x)+d for x in xs]
cubic_line = []
for x in xs:
    cubic_line.append((a*(x*x*x)+(b*(x*x))+(c*x)+d))

style.use('ggplot')
plt.scatter(xs,ys,color='#003F72')
plt.plot(xs, cubic_line)
plt.title('Win Streaks vs Probability with Cubic Fit Line')
plt.ylabel('Probability')
plt.xlabel('Win Streaks')
plt.show()

#Predicting the probability of winning next game given a certain win streak.
predict_x3 = int(input("Enter the number for games in Win Streak: "))
predict_p3 = (a1*(predict_x3*predict_x3*predict_x3)) + (b1*(predict_x3*predict_x3)) + (c1*predict_x3) + d
print(predict_p3)


# In[ ]:




