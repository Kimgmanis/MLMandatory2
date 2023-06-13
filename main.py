import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from minisom import MiniSom
from pylab import bone, pcolor, colorbar, plot, show
from steam import Steam
from decouple import config
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MultiLabelBinarizer

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
games[['all_review_sentiment', 'all_review_count', 'all_review_percentage']] = \
    games['all_reviews'].str.extract(r'([A-Za-z\s]+),\(([\d,]+)\),-\s([\d%]+)')
games = games.drop('all_reviews', axis=1)  # Used drop to remove the original 'popular_tags' column

# Split the 'recent_reviews' column into separate columns
games[['recent_review_sentiment', 'recent_review_count', 'recent_review_percentage']] = \
    games['recent_reviews'].str.extract(r'([A-Za-z\s]+),\(([\d,]+)\),-\s([\d%]+)')
games = games.drop('recent_reviews', axis=1)  # Remove the original 'recent_reviews' column

# str.split based on ',' and assigning the resulting list to a column called 'tags'
games['tags'] = games['popular_tags'].str.split(',')
games = games.drop('popular_tags', axis=1)  # Used drop function to remove the original 'popular_tags'
games.dropna(inplace=True)  # drops rows with nan

# Split comma-separated values and create new rows
new_rows = userDataset['games'].apply(lambda x: pd.Series(x))

# Concatenate new rows with the original DataFrame
userDataset = pd.concat([userDataset.drop('games', axis=1), new_rows], axis=1)

# Add a new column 'ownedGame' and set it to True initially
userDataset['ownedGame'] = True

# Strip leading and trailing whitespaces from game names in both datasets
userDataset['name'] = userDataset['name'].str.strip()
games['name'] = games['name'].str.strip()

# Convert game names to lowercase in both datasets
userDataset['name'] = userDataset['name'].str.lower()
games['name'] = games['name'].str.lower()

# Merge userDataset with games based on the game name
combined_data = pd.merge(userDataset, games, on='name', how='right')

# Replaces NaN to false for ownedGame column
combined_data['ownedGame'].fillna(False, inplace=True)

# Select the desired columns
all_columns = ['name', 'playtime_forever', 'all_review_sentiment', 'all_review_count', 'all_review_percentage',
               'recent_review_sentiment', 'recent_review_count', 'recent_review_percentage', 'ownedGame']

# Assign X and y dataframe
nameTimeCol = ['name', 'playtime_forever']  # X
reviewTagCol = ['name', 'all_review_sentiment', 'all_review_count', 'all_review_percentage',  # y
                'recent_review_sentiment', 'recent_review_count', 'recent_review_percentage', 'tags']

# Filter the combined data based on the selected columns
filtered_data = combined_data[all_columns]
# X = combined_data[nameTimeCol]
X = combined_data[combined_data['ownedGame']][nameTimeCol]
# y = using reviewTagCol
y = combined_data[reviewTagCol]

# print(combined_data)
# print(filtered_data)  # Prints all_columns = (name, playtime_forever,..)
# y processing
# create binary columns for each tag
mlb = MultiLabelBinarizer()
y = y.join(pd.DataFrame(mlb.fit_transform(y.pop('tags')),
                        columns=mlb.classes_,
                        index=y.index))

# Numeric features
numeric_features = ['all_review_count', 'recent_review_count', 'all_review_percentage', 'recent_review_percentage']
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])

# Convert columns to numeric and replace ',' with nothing
for feature in numeric_features:
    y[feature] = pd.to_numeric(y[feature].str.replace(',', ''), errors='coerce')

# Categorical features
categorical_features = ['all_review_sentiment', 'recent_review_sentiment', 'name']
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Processed Data
X = pd.get_dummies(X)
X = X.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y_preprocessed = preprocessor.fit_transform(y)

print(X)
print(y_preprocessed)
