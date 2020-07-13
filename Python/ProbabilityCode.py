def computeProb(counterWon, counterWS, counterGame):
    pb = counterWS / counterGame
    paib = counterWon / counterGame
    if pb == 0:
        pab = 0
    else:
        pab = paib / pb

    return pab


import pandas as pd

df = pd.read_csv("modifiedWinStreaks - good.csv")

streaks = df["Numeric Win Streak"]
# nextOutcome = df['Next game']

# Calculate the maximum (win) and minimum (lose) streaks
maxStreak = 0
minStreak = 0
for i in range(len(streaks)):
    if streaks[i] > maxStreak:
        maxStreak = streaks[i]
    elif streaks[i] < minStreak:
        minStreak = streaks[i]

print("The maximum win streak is: %d" % maxStreak)
print("The minimum win streak is %d" % minStreak)

probabilities = pd.Series([])

# This isn't complete, but I started to try to go by win streak instead of team
for i in range(minStreak, maxStreak):
    positionCounter = 0
    seriesCounter = 0
    winCounter = 0
    totalGames = 0
    if i != 0:
        while positionCounter > len(df):
            if df[positionCounter] == i:
                winCounter += 1
                totalGames += 1
            else:
                totalGames += 1
        # This part would use the computeProbability function.
        # Later it will add to our probability table
        # probabilities[seriesCounter] = computeProb(winCounter, totalGames, )


# This is Maria's code

teamNumberS = pd.Series([])
teamNameS = pd.Series([])
counterWonS = pd.Series([])
counterWSS = pd.Series([])
pabS = pd.Series([])

# no_integer = True
# while no_integer:
#    try:
#        wsValue = int(input("Enter the # of consecutive wins to be used as minimum Win Streak "))
#        no_integer = False
#    except:
#        print ("Enter an integer value")

# determine who won Game
teamNumber = df["Team Number"]
teamName = df["Team Name"]
winStreak = df["Numeric Win Streak"]

# Initialize counters for next team
team = -1
counterWS = 0
counterWon = 0
counterGame = 0
j = 0
wsValue = minStreak
for i in range(len(df)):
    if teamNumber[i] != team:
        if team != -1:
            print("Checking probability of %d games" % wsValue)
            # Calculate probabilities
            pab = computeProb(counterWon, counterWS, counterGame)
            # Stores figures for this team
            teamNumberS[j] = team
            teamNameS[j] = teamName[i - 1]
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
pab = computeProb(counterWon, counterWS, counterGame)
teamNumberS[j] = team
teamNameS[j] = teamName[i - 1]
counterWonS[j] = counterWon
counterWSS[j] = counterWS
pabS[j] = pab

# Creates the table with statistics per team
dft = pd.DataFrame(
    {
        "Team Number": teamNumberS,
        "Team Name": teamNameS,
        "#Wins WS": counterWonS,
        "#Win Streak": counterWSS,
        "Probability": pabS,
    }
)

print(dft)
dft.to_csv("big10Probability.csv")
print("done")
