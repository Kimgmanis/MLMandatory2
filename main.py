import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
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
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dense
from tensorflow.python.keras.optimizer_v2.adam import Adam

# API Key Stored in environmental variables
KEY = config("STEAM_API_KEY")  # Link: https://pypi.org/project/steam/
steam = Steam(KEY)  # Wrapper for API

# ref
account = "Kingmanis1929"
accountID = "76561198052658415"
# App id: to get specific apps
# terraria_app_id = 105600

# arguments: steamid
user = steam.users.search_user(account)  # User details / Easy and quick way to obtain user ID
# User game details
userGames = steam.users.get_owned_games(accountID)
# Searches for a game
# searchGame = steam.apps.get_app_details(terraria_app_id)

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
               'recent_review_sentiment', 'recent_review_count', 'recent_review_percentage', 'ownedGame', 'tags']

# Assign X and y dataframe
nameTimeCol = ['name', 'playtime_forever']  # X
reviewTagCol = ['name', 'all_review_sentiment', 'all_review_count', 'all_review_percentage',  # y
                'recent_review_sentiment', 'recent_review_count', 'recent_review_percentage', 'tags']

# Filter the combined data based on the selected columns
filtered_data = combined_data[all_columns]
# X = combined_data[nameTimeCol]
X = combined_data[combined_data['ownedGame']][nameTimeCol]
# y = using reviewTagCol
y = combined_data[combined_data['ownedGame']][reviewTagCol]
# Saving column names before preprocessing.
XColumns = X.columns
yColumns = list(y.columns)

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
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse=False))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)])

# Processed Data
X = pd.get_dummies(X)
X = X.values
scaler = StandardScaler()
X = scaler.fit_transform(X)
y = preprocessor.fit_transform(y)

# print(X)
# print(y_preprocessed)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.2, random_state = 42 )
# Model
model = Sequential()
model.add(Dense(64,activation='relu')) # 4 outputs. It will automatically adapt to number inputs
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(183,activation='softmax'))

adam = Adam(learning_rate=0.001) # you may have to change learning_rate, if the model does not learn.
model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

# Train
model.fit(X_train,y_train,epochs=500, verbose=1)
# Show loss vs. epochs
loss = model.history.history['loss']
sns.lineplot(x=range(len(loss)),y=loss)
model.evaluate(X_test,y_test,verbose=1)

# Generate the confusion matrix
y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_test_classes = np.argmax(y_test, axis=1)
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Convert the 'tags' column from lists to tuples
combined_data['tags'] = combined_data['tags'].apply(tuple)

# Get the unique class names from your 'tags' field.
class_names = combined_data['tags'].unique().tolist()

# Convert the confusion matrix to a DataFrame
cm_df = pd.DataFrame(cm, index=class_names[:cm.shape[0]], columns=class_names[:cm.shape[1]])
cm_df['tags'] = class_names[:cm.shape[0]]  # Add 'tags' column

# Convert the 'tags' column to string type
cm_df['tags'] = cm_df['tags'].astype(str)

# Reshape the DataFrame to explode the nested tags
cm_df_exp = cm_df['tags'].apply(lambda x: x.split(',')).explode().reset_index(drop=True).to_frame()
cm_df_exp.columns = ['tag']

## Convert the values in the confusion matrix DataFrame to numeric
cm_df_numeric = cm_df.apply(lambda x: pd.to_numeric(x, errors='coerce'))

# Create the heatmap using seaborn with the numeric values
plt.figure(figsize=(10, 7))
sns.heatmap(cm_df_numeric, annot=True, fmt='g')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()