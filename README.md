### Machine Learning Mandatory 2

Requirements. You must:

- Briefly describe the problem you want to solve

I would like to generate a list of potential games a user may buy from steam based on their play time data. This could be potentially useful for marketing and providing the right games to user to improve customer satisfaction.

- Find a dataset (f.ex. from kaggle.com)

Dataset from steam API
Link: https://pypi.org/project/steam/

Steam games complete dataset | Kaggle
Link: https://www.kaggle.com/datasets/trolukovich/steam-games-complete-dataset

- Prepare the data

Data structure example:

userGames format
{'game_count': 359, 'games': [{'appid': 4000, 'name': "Garry's Mod", 'playtime_forever': 7254, 'img_icon_url': '4a6f25cfa2426445d0d9d6e233408de4d371ce8b', 'has_community_visible_stats': True, 'playtime_windows_forever': 165, 'playtime_mac_forever': 0, 'playtime_linux_forever': 0, 'rtime_last_played': 1610650969},

- Train a model or Self Organizing Map (SOM)

- Evaluate the quality of the model / SOM.

Possible methods:
Supervised Learning:
- Deep Neural Network (previous example: bankcustomer, stay or leave)
- Convolutional Neural Network (previous example: recognize handwritten digits)
- Recurrent Neural Network (previous example: predict Google stock price)
Unsupervised Learning:
- Self Organizing Maps (previous example: detect fraud in Credit Card applications)
- Autoencoder (previous example: movie rating prediction)
