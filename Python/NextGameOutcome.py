import pandas as pd

# Reading the file
df = pd.read_csv("modifiedWinStreaks - good.csv")

# Initializing variables
teams = df["Team Number"]
teams = pd.to_numeric(teams)
streaks = df["Numeric Win Streak"]
streaks = pd.to_numeric(streaks)
# print(streaks)

# Correcting team numbers
for i in range(len(teams)):
    teams[i] = teams[i] + 1
# print(teams)

# Determining outcome of next game: W, L, or N (no future game)
nextOutcome = pd.Series([])
for i in range(len(df)):
    if i < len(df) - 1:
        currentTeam = teams[i]
        currentStreak = streaks[i]
        if teams[i + 1] == currentTeam:
            if streaks[i + 1] > 0:
                nextOutcome[i] = "W"
            elif streaks[i + 1] < 0:
                nextOutcome[i] = "L"
        else:
            nextOutcome[i] = "N"
    else:
        nextOutcome[i] = "N"

# Inserting data, dropping extra column, printing to new CSV file
df.insert(6, "Next Game", nextOutcome)
df.drop("Unnamed: 0", axis=1, inplace=True)
df.to_csv("finalCSV.csv")
print("done")

