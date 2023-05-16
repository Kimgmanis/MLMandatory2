import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from steam import Steam
from decouple import config

KEY = config("STEAM_API_KEY")
steam = Steam(KEY)

# ref
account = "Kingmanis1929"
accountID = "76561198052658415"
terraria_app_id = 105600

# arguments: steamid
# Retrieves user details
user = steam.users.search_user(account)
# Retrieves user game details
userGames = steam.users.get_owned_games(accountID)
# Searches for a game
searchGame = steam.apps.get_app_details(terraria_app_id)
# print(user)
# print(userGames)
# print(searchGame)

gamesDataset = pd.read_csv('steam_games.csv')
userDataset = pd.DataFrame.from_dict(userGames)
games = gamesDataset.loc[:, ['name', 'recent_reviews', 'all_reviews', 'popular_tags']]
# Split comma-separated values and create new rows
new_rows = userDataset['games'].apply(lambda x: pd.Series(x))
# Concatenate new rows with the original DataFrame
userDataset = pd.concat([userDataset.drop('games', axis=1), new_rows], axis=1)
# playtime_forever is total number of minutes played on a game
# Select and display specific columns
selected_columns = ['name', 'playtime_forever']
print(userDataset[selected_columns])
print(games)
