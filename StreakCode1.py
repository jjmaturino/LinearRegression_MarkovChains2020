# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 03:10:04 2020

@author: Sheila
"""

import scipy.stats as stats   
import numpy as np
import pandas as pd
import random
import csv


#Import data
#df = pd.read_csv('CollegeBaseball2018Games.csv') #This will our test data

#Using Big10 data below for now since it is a smaller set
df = pd.read_csv('Big10Games.csv')
teamData = pd.read_csv('Big10Teams.csv')

#Create new columns in csv 
Result1 = pd.Series([])  #Column for whether team 1 won or lost
Result2 = pd.Series([])  #Column for whether team 2 won or lost
WinStreak1 = pd.Series([]) 
LossStreak1 = pd.Series([]) 
WinStreak2 = pd.Series([]) 
LossStreak2 = pd.Series([]) 
NextGame1 = pd.Series([])   #Column for whether team 1 won or lost next game
NextGame2 = pd.Series([])   #Column for whether team 2 won or lost next game


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
#df.insert(6, 'Result1', Result1)
#df.insert(9, 'Result2', Result2)

#Create variables for team1 and team2 columns
team1 = df['Team1']
team2 = df['Team2']

#Score difference = team1 score - team2 score
scoreDiff = score1-score2

team1games = []
team1games = df.loc[(df['Team1'] == 1) | (df['Team2'] == 1)]
#print(team1games)
for i in team1games.itertuples():  
    print(i)
    if team1games['Team1'][i] == 1:
        if team1games['Team1'][i+1] == 1:
            NextGame1[i] == 'W'
        elif team1games['Team2'][i+1] == 1:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team1games['Team2'][i] == 1:
        if team1games['Team1'][i+1] == 1:
            NextGame2[i] == 'W'
        elif team1games['Team2'][i+1] == 1:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
    
team2games = []
team2games = df.loc[(df['Team1'] == 2) | (df['Team2'] == 2)]
#print(team2games)
for i in team2games.itertuples():  
    print(i)
    if team2games['Team1'][i] == 2:
        if team2games['Team1'][i+1] == 2:
            NextGame1[i] == 'W'
        elif team2games['Team2'][i+1] == 2:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team2games['Team2'][i] == 2:
        if team2games['Team1'][i+1] == 2:
            NextGame2[i] == 'W'
        elif team2games['Team2'][i+1] == 2:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team3games = []
team3games = df.loc[(df['Team1'] == 3) | (df['Team2'] == 3)]
#print(team3games)
for i in team3games.itertuples():  
    print(i)
    if team3games['Team1'][i] == 3:
        if team3games['Team1'][i+1] == 3:
            NextGame1[i] == 'W'
        elif team3games['Team2'][i+1] == 3:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team3games['Team2'][i] == 3:
        if team3games['Team1'][i+1] == 3:
            NextGame2[i] == 'W'
        elif team3games['Team2'][i+1] == 3:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team4games = []
team4games = df.loc[(df['Team1'] == 4) | (df['Team2'] == 4)]
#print(team4games)
for i in team4games.itertuples():  
    print(i)
    if team4games['Team1'][i] == 4:
        if team4games['Team1'][i+1] == 4:
            NextGame1[i] == 'W'
        elif team4games['Team2'][i+1] == 4:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team3games['Team2'][i] == 4:
        if team4games['Team1'][i+1] == 4:
            NextGame2[i] == 'W'
        elif team4games['Team2'][i+1] == 4:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team5games = []
team5games = df.loc[(df['Team1'] == 5) | (df['Team2'] == 5)]
#print(team5games)
for i in team5games.itertuples():  
    print(i)
    if team5games['Team1'][i] == 5:
        if team5games['Team1'][i+1] == 5:
            NextGame1[i] == 'W'
        elif team5games['Team2'][i+1] == 5:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team5games['Team2'][i] == 5:
        if team5games['Team1'][i+1] == 5:
            NextGame2[i] == 'W'
        elif team5games['Team2'][i+1] == 5:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team6games = []
team6games = df.loc[(df['Team1'] == 6) | (df['Team2'] == 6)]
#print(team6games)
for i in team6games.itertuples():  
    print(i)
    if team6games['Team1'][i] == 6:
        if team6games['Team1'][i+1] == 6:
            NextGame1[i] == 'W'
        elif team6games['Team2'][i+1] == 6:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team6games['Team2'][i] == 6:
        if team6games['Team1'][i+1] == 6:
            NextGame2[i] == 'W'
        elif team6games['Team2'][i+1] == 6:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team7games = []
team7games = df.loc[(df['Team1'] == 7) | (df['Team2'] == 7)]
#print(team7games)
for i in team7games.itertuples():  
    print(i)
    if team7games['Team1'][i] == 7:
        if team7games['Team1'][i+1] == 7:
            NextGame1[i] == 'W'
        elif team7games['Team2'][i+1] == 7:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team7games['Team2'][i] == 7:
        if team7games['Team1'][i+1] == 7:
            NextGame2[i] == 'W'
        elif team7games['Team2'][i+1] == 7:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team8games = []
team8games = df.loc[(df['Team1'] == 8) | (df['Team2'] == 8)]
#print(team8games)
for i in team8games.itertuples():  
    print(i)
    if team8games['Team1'][i] == 8:
        if team8games['Team1'][i+1] == 8:
            NextGame1[i] == 'W'
        elif team8games['Team2'][i+1] == 8:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team8games['Team2'][i] == 8:
        if team8games['Team1'][i+1] == 8:
            NextGame2[i] == 'W'
        elif team8games['Team2'][i+1] == 8:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team9games = []
team9games = df.loc[(df['Team1'] == 9) | (df['Team2'] == 9)]
#print(team9games)
for i in team9games.itertuples():  
    print(i)
    if team9games['Team1'][i] == 9:
        if team9games['Team1'][i+1] == 9:
            NextGame1[i] == 'W'
        elif team9games['Team2'][i+1] == 9:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team9games['Team2'][i] == 9:
        if team9games['Team1'][i+1] == 9:
            NextGame2[i] == 'W'
        elif team9games['Team2'][i+1] == 9:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team10games = []
team10games = df.loc[(df['Team1'] == 10) | (df['Team2'] == 10)]
#print(team10games)
for i in team10games.itertuples():  
    print(i)
    if team10games['Team1'][i] == 10:
        if team10games['Team1'][i+1] == 10:
            NextGame1[i] == 'W'
        elif team10games['Team2'][i+1] == 10:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team10games['Team2'][i] == 10:
        if team10games['Team1'][i+1] == 10:
            NextGame2[i] == 'W'
        elif team10games['Team2'][i+1] == 10:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team11games = []
team11games = df.loc[(df['Team1'] == 11) | (df['Team2'] == 11)]
#print(team11games)
for i in team11games.itertuples():  
    print(i)
    if team11games['Team1'][i] == 11:
        if team11games['Team1'][i+1] == 11:
            NextGame1[i] == 'W'
        elif team11games['Team2'][i+1] == 11:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team11games['Team2'][i] == 11:
        if team11games['Team1'][i+1] == 11:
            NextGame2[i] == 'W'
        elif team11games['Team2'][i+1] == 11:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team12games = []
team12games = df.loc[(df['Team1'] == 12) | (df['Team2'] == 12)]
#print(team12games)
for i in team12games.itertuples():  
    print(i)
    if team12games['Team1'][i] == 12:
        if team12games['Team1'][i+1] == 12:
            NextGame1[i] == 'W'
        elif team12games['Team2'][i+1] == 12:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team12games['Team2'][i] == 12:
        if team12games['Team1'][i+1] == 12:
            NextGame2[i] == 'W'
        elif team12games['Team2'][i+1] == 12:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team13games = []
team13games = df.loc[(df['Team1'] == 13) | (df['Team2'] == 13)]
#print(team1games)
for i in team13games.itertuples():  
    print(i)
    if team13games['Team1'][i] == 13:
        if team13games['Team1'][i+1] == 13:
            NextGame1[i] == 'W'
        elif team13games['Team2'][i+1] == 13:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team13games['Team2'][i] == 13:
        if team13games['Team1'][i+1] == 13:
            NextGame2[i] == 'W'
        elif team13games['Team2'][i+1] == 13:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

df.insert(7, 'NextGame1', NextGame1)
df.insert(10, 'NextGame2', NextGame2)

'''
#Determines which team won/lost, add to df and prints to new csv
#Not necessary for masseyratings data, but may be for differently arranged data sets
team1 = df['Team1']
team2 = df['Team2']
team1wins = pd.Series([])
team2wins = pd.Series([])
for i in range(len(df)):
    counter = 1
    while(counter <= 13):
        if team1[i] == counter:
            if Result1[i] == 'W':
                team1wins[i] = "+"
            elif Result1[i] == "L":
                team1wins[i] = "-"
        elif team2[i] == counter:
            if Result2[i] == 'W':
                team2wins[i] = "+"
            elif Result2[i] == "L":
                team2wins[i] = "-"
        counter += 1

df.insert(10, "Team 1 Winstreak", team1wins)
df.insert(11, "Team 2 Winstreak", team2wins)
print(df)
df.to_csv('big10stats_modified.csv')
print('done')

# Create a win dictionary to keep track of team win/loss streak
winDictionary = {}
#teamNames = pd.Series([])
name = teamData['TeamName']
for i in range(len(teamData)):
    #teamNames[i] = name[i]
    winDictionary[name[i]] = ""

    
#teamData.insert(2, 'Added Team Name', teamNames)
#print(teamData)

#Determines the win/loss streaks for each team throughout the season
for i in range(len(df)):
    counter = 1
    while(counter <= len(teamData)):
        currentTeam = name[counter - 1]
        if team1[i] == counter:
            if Result1[i] == 'W':
                if '+' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += '+'
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '+'
                    #print(winDictionary[currentTeam])
            elif Result1[i] == "L":
                if '-' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += '-'
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '-'
                    #print(winDictionary[currentTeam])
            #add else here?
        elif team2[i] == counter:
            if Result2[i] == 'W':
                if '+' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += "+"
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '+'
                    #print(winDictionary[currentTeam])
            elif Result2[i] == "L":
                if '-' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += "-"
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '-'
                    #print(winDictionary[currentTeam])
        counter += 1

print("The dictionary: ")
print(winDictionary)

ws = pd.DataFrame.from_dict(winDictionary, orient='index')
print(ws)
ws.to_csv('big10winStreaks.csv', header=['Streak'])
print('done')
>>>>>>> f3dcb87c3e7dc54f06450da3e5db3ab94c6c2ff5
'''
=======
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 03:10:04 2020

@author: Sheila
"""

import scipy.stats as stats   
import numpy as np
import pandas as pd
import random
import csv


#Import data
#df = pd.read_csv('CollegeBaseball2018Games.csv') #This will our test data

#Using Big10 data below for now since it is a smaller set
df = pd.read_csv('Big10Games.csv')
teamData = pd.read_csv('Big10Teams.csv')

#Create new columns in csv 
Result1 = pd.Series([])  #Column for whether team 1 won or lost
Result2 = pd.Series([])  #Column for whether team 2 won or lost
WinStreak1 = pd.Series([]) 
LossStreak1 = pd.Series([]) 
WinStreak2 = pd.Series([]) 
LossStreak2 = pd.Series([]) 
NextGame1 = pd.Series([])   #Column for whether team 1 won or lost next game
NextGame2 = pd.Series([])   #Column for whether team 2 won or lost next game


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
#df.insert(6, 'Result1', Result1)
#df.insert(9, 'Result2', Result2)

#Create variables for team1 and team2 columns
team1 = df['Team1']
team2 = df['Team2']

#Score difference = team1 score - team2 score
scoreDiff = score1-score2

team1games = []
team1games = df.loc[(df['Team1'] == 1) | (df['Team2'] == 1)]
#print(team1games)
for i in team1games.itertuples():  
    print(i)
    if team1games['Team1'][i] == 1:
        if team1games['Team1'][i+1] == 1:
            NextGame1[i] == 'W'
        elif team1games['Team2'][i+1] == 1:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team1games['Team2'][i] == 1:
        if team1games['Team1'][i+1] == 1:
            NextGame2[i] == 'W'
        elif team1games['Team2'][i+1] == 1:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
    
team2games = []
team2games = df.loc[(df['Team1'] == 2) | (df['Team2'] == 2)]
#print(team2games)
for i in team2games.itertuples():  
    print(i)
    if team2games['Team1'][i] == 2:
        if team2games['Team1'][i+1] == 2:
            NextGame1[i] == 'W'
        elif team2games['Team2'][i+1] == 2:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team2games['Team2'][i] == 2:
        if team2games['Team1'][i+1] == 2:
            NextGame2[i] == 'W'
        elif team2games['Team2'][i+1] == 2:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team3games = []
team3games = df.loc[(df['Team1'] == 3) | (df['Team2'] == 3)]
#print(team3games)
for i in team3games.itertuples():  
    print(i)
    if team3games['Team1'][i] == 3:
        if team3games['Team1'][i+1] == 3:
            NextGame1[i] == 'W'
        elif team3games['Team2'][i+1] == 3:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team3games['Team2'][i] == 3:
        if team3games['Team1'][i+1] == 3:
            NextGame2[i] == 'W'
        elif team3games['Team2'][i+1] == 3:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team4games = []
team4games = df.loc[(df['Team1'] == 4) | (df['Team2'] == 4)]
#print(team4games)
for i in team4games.itertuples():  
    print(i)
    if team4games['Team1'][i] == 4:
        if team4games['Team1'][i+1] == 4:
            NextGame1[i] == 'W'
        elif team4games['Team2'][i+1] == 4:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team3games['Team2'][i] == 4:
        if team4games['Team1'][i+1] == 4:
            NextGame2[i] == 'W'
        elif team4games['Team2'][i+1] == 4:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team5games = []
team5games = df.loc[(df['Team1'] == 5) | (df['Team2'] == 5)]
#print(team5games)
for i in team5games.itertuples():  
    print(i)
    if team5games['Team1'][i] == 5:
        if team5games['Team1'][i+1] == 5:
            NextGame1[i] == 'W'
        elif team5games['Team2'][i+1] == 5:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team5games['Team2'][i] == 5:
        if team5games['Team1'][i+1] == 5:
            NextGame2[i] == 'W'
        elif team5games['Team2'][i+1] == 5:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team6games = []
team6games = df.loc[(df['Team1'] == 6) | (df['Team2'] == 6)]
#print(team6games)
for i in team6games.itertuples():  
    print(i)
    if team6games['Team1'][i] == 6:
        if team6games['Team1'][i+1] == 6:
            NextGame1[i] == 'W'
        elif team6games['Team2'][i+1] == 6:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team6games['Team2'][i] == 6:
        if team6games['Team1'][i+1] == 6:
            NextGame2[i] == 'W'
        elif team6games['Team2'][i+1] == 6:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team7games = []
team7games = df.loc[(df['Team1'] == 7) | (df['Team2'] == 7)]
#print(team7games)
for i in team7games.itertuples():  
    print(i)
    if team7games['Team1'][i] == 7:
        if team7games['Team1'][i+1] == 7:
            NextGame1[i] == 'W'
        elif team7games['Team2'][i+1] == 7:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team7games['Team2'][i] == 7:
        if team7games['Team1'][i+1] == 7:
            NextGame2[i] == 'W'
        elif team7games['Team2'][i+1] == 7:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team8games = []
team8games = df.loc[(df['Team1'] == 8) | (df['Team2'] == 8)]
#print(team8games)
for i in team8games.itertuples():  
    print(i)
    if team8games['Team1'][i] == 8:
        if team8games['Team1'][i+1] == 8:
            NextGame1[i] == 'W'
        elif team8games['Team2'][i+1] == 8:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team8games['Team2'][i] == 8:
        if team8games['Team1'][i+1] == 8:
            NextGame2[i] == 'W'
        elif team8games['Team2'][i+1] == 8:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team9games = []
team9games = df.loc[(df['Team1'] == 9) | (df['Team2'] == 9)]
#print(team9games)
for i in team9games.itertuples():  
    print(i)
    if team9games['Team1'][i] == 9:
        if team9games['Team1'][i+1] == 9:
            NextGame1[i] == 'W'
        elif team9games['Team2'][i+1] == 9:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team9games['Team2'][i] == 9:
        if team9games['Team1'][i+1] == 9:
            NextGame2[i] == 'W'
        elif team9games['Team2'][i+1] == 9:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team10games = []
team10games = df.loc[(df['Team1'] == 10) | (df['Team2'] == 10)]
#print(team10games)
for i in team10games.itertuples():  
    print(i)
    if team10games['Team1'][i] == 10:
        if team10games['Team1'][i+1] == 10:
            NextGame1[i] == 'W'
        elif team10games['Team2'][i+1] == 10:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team10games['Team2'][i] == 10:
        if team10games['Team1'][i+1] == 10:
            NextGame2[i] == 'W'
        elif team10games['Team2'][i+1] == 10:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'
            
team11games = []
team11games = df.loc[(df['Team1'] == 11) | (df['Team2'] == 11)]
#print(team11games)
for i in team11games.itertuples():  
    print(i)
    if team11games['Team1'][i] == 11:
        if team11games['Team1'][i+1] == 11:
            NextGame1[i] == 'W'
        elif team11games['Team2'][i+1] == 11:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team11games['Team2'][i] == 11:
        if team11games['Team1'][i+1] == 11:
            NextGame2[i] == 'W'
        elif team11games['Team2'][i+1] == 11:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team12games = []
team12games = df.loc[(df['Team1'] == 12) | (df['Team2'] == 12)]
#print(team12games)
for i in team12games.itertuples():  
    print(i)
    if team12games['Team1'][i] == 12:
        if team12games['Team1'][i+1] == 12:
            NextGame1[i] == 'W'
        elif team12games['Team2'][i+1] == 12:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team12games['Team2'][i] == 12:
        if team12games['Team1'][i+1] == 12:
            NextGame2[i] == 'W'
        elif team12games['Team2'][i+1] == 12:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

team13games = []
team13games = df.loc[(df['Team1'] == 13) | (df['Team2'] == 13)]
#print(team1games)
for i in team13games.itertuples():  
    print(i)
    if team13games['Team1'][i] == 13:
        if team13games['Team1'][i+1] == 13:
            NextGame1[i] == 'W'
        elif team13games['Team2'][i+1] == 13:
            NextGame1[i] == 'L'
        else:
            NextGame1[i] == 'N'
    elif team13games['Team2'][i] == 13:
        if team13games['Team1'][i+1] == 13:
            NextGame2[i] == 'W'
        elif team13games['Team2'][i+1] == 13:
            NextGame2[i] == 'L'
        else:
            NextGame2[i] == 'N'

df.insert(7, 'NextGame1', NextGame1)
df.insert(10, 'NextGame2', NextGame2)

'''
#Determines which team won/lost, add to df and prints to new csv
#Not necessary for masseyratings data, but may be for differently arranged data sets
team1 = df['Team1']
team2 = df['Team2']
team1wins = pd.Series([])
team2wins = pd.Series([])
for i in range(len(df)):
    counter = 1
    while(counter <= 13):
        if team1[i] == counter:
            if Result1[i] == 'W':
                team1wins[i] = "+"
            elif Result1[i] == "L":
                team1wins[i] = "-"
        elif team2[i] == counter:
            if Result2[i] == 'W':
                team2wins[i] = "+"
            elif Result2[i] == "L":
                team2wins[i] = "-"
        counter += 1

df.insert(10, "Team 1 Winstreak", team1wins)
df.insert(11, "Team 2 Winstreak", team2wins)
print(df)
df.to_csv('big10stats_modified.csv')
print('done')

# Create a win dictionary to keep track of team win/loss streak
winDictionary = {}
#teamNames = pd.Series([])
name = teamData['TeamName']
for i in range(len(teamData)):
    #teamNames[i] = name[i]
    winDictionary[name[i]] = ""

    
#teamData.insert(2, 'Added Team Name', teamNames)
#print(teamData)

#Determines the win/loss streaks for each team throughout the season
for i in range(len(df)):
    counter = 1
    while(counter <= len(teamData)):
        currentTeam = name[counter - 1]
        if team1[i] == counter:
            if Result1[i] == 'W':
                if '+' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += '+'
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '+'
                    #print(winDictionary[currentTeam])
            elif Result1[i] == "L":
                if '-' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += '-'
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '-'
                    #print(winDictionary[currentTeam])
            #add else here?
        elif team2[i] == counter:
            if Result2[i] == 'W':
                if '+' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += "+"
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '+'
                    #print(winDictionary[currentTeam])
            elif Result2[i] == "L":
                if '-' in winDictionary[currentTeam]:
                    winDictionary[currentTeam] += "-"
                    #print(winDictionary[currentTeam])
                else:
                    winDictionary[currentTeam] = '-'
                    #print(winDictionary[currentTeam])
        counter += 1

print("The dictionary: ")
print(winDictionary)

ws = pd.DataFrame.from_dict(winDictionary, orient='index')
print(ws)
ws.to_csv('big10winStreaks.csv', header=['Streak'])
print('done')
'''
>>>>>>> 87b5289dac4cbfd0fbd4597d2dec7f418348044f:StreakCode1.py
