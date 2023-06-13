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

# Data ini
gamesDataset = pd.read_csv('steam_games.csv')
userDataset = pd.DataFrame.from_dict(userGames)

# Formatting Labels
games = gamesDataset.loc[:, ['name', 'recent_reviews', 'all_reviews', 'popular_tags']]

# Split the 'all_reviews' column into separate columns
games[['all_review_sentiment', 'all_review_count', 'all_review_percentage']] =\
    games['all_reviews'].str.extract(r'([A-Za-z\s]+),\(([\d,]+)\),-\s([\d%]+)')
games = games.drop('all_reviews', axis=1)  # Used drop to remove the original 'popular_tags' column

# Split the 'recent_reviews' column into separate columns
games[['recent_review_sentiment', 'recent_review_count', 'recent_review_percentage']] =\
    games['recent_reviews'].str.extract(r'([A-Za-z\s]+),\(([\d,]+)\),-\s([\d%]+)')
games = games.drop('recent_reviews', axis=1)  # Remove the original 'recent_reviews' column

# str.split based on ',' and assigning the resulting list to a column called 'tags'
games['tags'] = games['popular_tags'].str.split(',')
games = games.drop('popular_tags', axis=1)  # Used drop function to remove the original 'popular_tags'

# Split comma-separated values and create new rows
new_rows = userDataset['games'].apply(lambda x: pd.Series(x))
# Concatenate new rows with the original DataFrame
userDataset = pd.concat([userDataset.drop('games', axis=1), new_rows], axis=1)
# Select and display specific columns
selected_columns = ['name', 'playtime_forever']  # playtime_forever is total number of minutes played on a game

print(userDataset[selected_columns])
print(games)
