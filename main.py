import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from steam import Steam
from decouple import config

# API Key Stored in environmental variables
KEY = config("STEAM_API_KEY")  # Link: https://pypi.org/project/steam/
steam = Steam(KEY)  # Wrapper for API

# ref
account = "Kingmanis1929"
accountID = "76561198052658415"
# App id: to get specific apps
terraria_app_id = 105600

# arguments: steamid
user = steam.users.search_user(account)  # User details / Easy and quick way to obtain user ID
# User game details
userGames = steam.users.get_owned_games(accountID)
# Searches for a game
searchGame = steam.apps.get_app_details(terraria_app_id)

# Data set Ini
gamesDataset = pd.read_csv('steam_games.csv')
userDataset = pd.DataFrame.from_dict(userGames)

# Labels
games = gamesDataset.loc[:, ['name', 'recent_reviews', 'all_reviews', 'popular_tags']]

# Split comma-separated values and create new rows
new_rows = userDataset['games'].apply(lambda x: pd.Series(x))
# Concatenate new rows with the original DataFrame
userDataset = pd.concat([userDataset.drop('games', axis=1), new_rows], axis=1)
# Select and display specific columns
selected_columns = ['name', 'playtime_forever']  # playtime_forever is total number of minutes played on a game

print(userDataset[selected_columns])
print(games)
