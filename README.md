# Movie Recommendation System
This Jupyter notebook contains code for building a movie recommendation system using various algorithms such as K-Nearest Neighbors, Pearson Correlation and Singular Value Decomposition.

## Getting Started
To run the code in this notebook, you will need to have the following libraries installed:

```numpy``` <br>
```pandas```<br>
```surprise```<br>
```scikit-learn```<br>
You can install these libraries by running the following command in a code cell:

```!pip install numpy pandas surprise scikit-learn``` <br>
You will also need to have the movies.csv, ratings.csv, and tags.csv datasets in the same directory as this notebook.

### Recommendation System using K-Nearest Neighbors algorithm
This section of the notebook demonstrates how to build a recommendation system using the K-Nearest Neighbors algorithm. The code loads the necessary datasets and libraries, normalizes the data using MinMaxScaler, and splits it into training and testing sets.

The data is then vectorized using tfidf and a tfidf matrix is created. The KNN algorithm is used to find k nearest neighbors and cosine similarity. The code then predicts ratings for test data and calculates the root mean squared error (RMSE) and mean absolute error (MAE).

### Recommendation System with Collaborative Filtering using User Ratings
This section of the notebook demonstrates how to build a recommendation system using collaborative filtering with user ratings. The code loads the necessary datasets and performs preprocessing steps such as extracting the year from movie titles and dropping unnecessary columns.

The code then takes user input in the form of movie ratings and preprocesses it by merging it with the movies dataset. The users who have seen the same movies as the input user are then selected and grouped by userId.

The Pearson Correlation Coefficient is calculated for each user to determine their similarity to the input user. The top 50 most similar users are then selected and their ratings are used to calculate a weighted average recommendation score for each movie.

The top 10 recommended movies are then displayed using collaborative filtering.

### Recommendation System using Singular Value Decomposition
This section of the notebook demonstrates how to build a recommendation system using Singular Value Decomposition (SVD). The code loads the necessary datasets and libraries, performs preprocessing steps such as merging dataframes and splitting data into training and testing sets.

The SVD algorithm is then used to fit the training data and make predictions on the test data. The RMSE and MAE are calculated to measure the accuracy of the predictions.

A function is also provided to measure the accuracy of different algorithms such as SVD and BaselineOnly. The results are displayed in a dataframe showing the RMSE, MAE, precision, recall, F-measure, and NDCG for each algorithm.

### Conclusion
This Jupyter notebook provides an example of how to build a movie recommendation system using various algorithms such as K-Nearest Neighbors, Collaborative Filtering with User Ratings, and Singular Value Decomposition. The code demonstrates how to load and preprocess data, fit models, make predictions, and evaluate their accuracy.
