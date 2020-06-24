# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 03:10:04 2020

@author: Sheila
"""

#import scipy.stats as stats   
import numpy as np
import pandas as pd
import random
import csv

#Import data
#df = pd.read_csv('CollegeBaseball2018Games.csv')
df = pd.read_csv('big10stats.csv')
teamData = pd.read_csv('big10teams.csv')

Result1 = pd.Series([]) 
Result2= pd.Series([])
WinStreak1 = pd.Series([])

#Determine if Team Home or Away and Make new column for that

#determine who won Game
score1 = df['Score1']   
score2 = df['Score2']

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

#Add new column with game results    
df.insert(6, 'Result1', Result1)
df.insert(9, 'Result2', Result2)
#print(df)

#Determines which team won/lost, add to df and prints to new csv
#Probably not necessary for masseyratings.com data, but may be for differently arranged data sets
team1 = df['Team1']
team2 = df['Team2']
team1wins = pd.Series([])
team2wins = pd.Series([])
for i in range(len(df)):
    counter = 1
    while(counter <= len(teamData)):
        if team1[i] == counter:
            if Result1[i] == 'W':
                team1wins[i] = "W"
            elif Result1[i] == "L":
                team1wins[i] = "L"
        elif team2[i] == counter:
            if Result2[i] == 'W':
                team2wins[i] = "W"
            elif Result2[i] == "L":
                team2wins[i] = "L"
        counter += 1

df.insert(10, "Team 1 Result", team1wins)
df.insert(11, "Team 2 Result", team2wins)
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

dates = df['Date']
dates = pd.to_numeric(dates)
print("The dates are: ")
print(dates)


#Takes the date you want as input, will be used to find games through this day
lowBound = dates[0]
print("The low bound is: %d" % lowBound)
upTo = int(input("Enter a date in the following format YYYYMMDD: "))


#teamData.insert(2, 'Added Team Name', teamNames)
#print(teamData)
ws = pd.DataFrame.from_dict(winDictionary, orient='index')
#ws = pd.DataFrame.from_records(winDictionary, columns=['Team', 'Streak', 'Date'])

names = pd.Series([])
#Determines the win/loss streaks for each team based on date
#Prints each team to a dictionary that stores the team win/loss streak
for i in range(len(df)):
    #counter += 1
    for counter in range(len(teamData)):
        if upTo >= dates[i]:
            currentDate = dates[i]
            currentTeam = name[counter]
            if team1[i] == counter + 1:
                if Result1[i] == 'W':
                    if '+' in winDictionary[currentTeam]:
                        winDictionary[currentTeam] += '+'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                    else:
                        winDictionary[currentTeam] = '+'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                elif Result1[i] == "L":
                    if '-' in winDictionary[currentTeam]:
                        winDictionary[currentTeam] += '-'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                    else:
                        winDictionary[currentTeam] = '-'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
            elif team2[i] == counter + 1:
                if Result2[i] == 'W':
                    if '+' in winDictionary[currentTeam]:
                        winDictionary[currentTeam] += "+"
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                    else:
                        winDictionary[currentTeam] = '+'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                elif Result2[i] == "L":
                    if '-' in winDictionary[currentTeam]:
                        winDictionary[currentTeam] += "-"
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
                    else:
                        winDictionary[currentTeam] = '-'
                        print(winDictionary[currentTeam])
                        names[i] = currentTeam
            #print("The dictionary: ")
            #print(winDictionary)
    ws = pd.DataFrame(list(winDictionary.items()))#, index=['Team Name', 'Win Streak'])
    #ws = pd.DataFrame.from_dict(winDictionary, orient='index')
    #ws = pd.DataFrame.from_records(winDictionary, columns=['Team', 'Streak'])
    ws.insert(1, "Date", currentDate)
    #ws.insert(0, "Team Name", names)
    #ws.insert(0, "Team Name", currentTeam)
    ws.to_csv('big10winStreaks.csv', mode = 'a', header=False)

#ws.insert(0, "Team Name", names)
ws.columns = ['Team Name', 'Date', 'Win Streak']
print(ws)
print('done')

#Deletes duplicate win/loss entries and any entries that do not contain a streak
streaks = pd.read_csv('big10WinStreaks.csv')
streaks.columns = ['Team Number', 'Team Name', 'Date', 'Win Streak']
#del streaks['Team Number']
streaks.dropna(axis=0, subset=['Win Streak'], inplace=True)
streaks.drop_duplicates(keep='first', inplace=True)
indexing = []
for i in range(len(streaks)):
    indexing.append(i)
streaks.index = indexing
print(streaks)
teamNumbers = streaks['Team Number']
teamNumbers = pd.to_numeric(teamNumbers)
print("Team Numbers: ")
print(teamNumbers)
print("Team Streaks: ")
teamStreaks = streaks['Win Streak']
teamStreaks.to_string()
print(teamStreaks)
print(streaks)
streaks.to_csv("checkwinstreaks.csv")
# CODE I AM NOT SURE ABOUT
yesCount = 0
for i in range(len(streaks)):
    number = teamNumbers[i]
    print("Number: %d" % number)
    streak = teamStreaks[i]
    print("Streak: %s" % streak)
    checkPosition = i + 1
    while checkPosition < (len(streaks) - 1)  and teamNumbers[checkPosition] != number:
        checkPosition += 1
    if checkPosition < len(streaks) and teamStreaks[checkPosition] == streak:
        print("yes")
        yesCount += 1
        print(checkPosition)
        streaks = streaks.drop(streaks.index[checkPosition])
        delStreak = streak


print("yesCount: %d" % yesCount)

print("Streaks: ")
print(streaks)
streaks.to_csv('newWinStreaks.csv')
print('done')

# OLD COMMENTS/NOTES
#ws.columns = ['Streak', 'Date']
#ws.drop_duplicates(keep='last',inplace=True)
#ws.to_csv('newWinStreaks.csv')


#print(ws)
#print('done')

#Teams are 1 through 297
#Should i sort the team row by number then compare dates W or L?
#Make a list of each teams games then create counter that way
#or dictionary with team

#for i in range(len(teamData)):
#I'm just going to hard code this part for now...
#Illinois = df[df['TeamName']=='Illinois']
'''
countWins = 0
countLoss = 0
team1 = df['Team1']
team2 = df['Team2']
result = df['Result1']
for i in range(len(df)): 
    if team1[i] == 1:
      if result[i] == 'W':
          countWins += 1
    if team2[i] == 1:
        countLoss += 1

#print(countWins)
#print(countLoss)
#print(df[df['Team1'] == 2]) #This has 18 rows which matches countWins
#print(df[df['Team2'] == 2]) #This has 12 rows which matches countLoss

IllinoisWin = df[df['Team1'] == 1]
IllinoisLoss = df[df['Team2'] == 1]
IndianaWin = df[df['Team1'] == 2]
IndianaLoss = df[df['Team2'] == 2]
IowaWin = df[df['Team1'] == 3]
IowaLoss = df[df['Team2'] == 3]

ilGames = []
for i in range(len(df)):
    if team1[i] == 1:
        ilGames.append(i)
    if team2[i] == 1:
        ilGames.append(i)
 
print(ilGames) #This is list is just the row numbers where 1 appears
'''
#Team 1 seems to always be the winner so when a Team shows up in Team2
            # column they lost so start losing streak



