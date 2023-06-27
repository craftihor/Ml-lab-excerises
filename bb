plt.figure(figsize=(10, 6))
plt.barh(attribute_contributions['Attribute'], attribute_contributions['Contribution'])
plt.xlabel('Contribution')
plt.ylabel('Attribute')
plt.title('Attribute Contributions')
plt.tight_layout()
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Assuming you have attribute contributions stored in a DataFrame called 'attribute_contributions'

# Pivot the attribute contributions DataFrame for visualization
heatmap_data = attribute_contributions.pivot(index=None, columns='Attribute', values='Contribution')

# Create a heatmap of attribute contributions
plt.figure(figsize=(10, 6))
sns.heatmap(heatmap_data, cmap='coolwarm', annot=True, fmt=".2f", cbar=False)
plt.title('Attribute Contributions Heatmap')
plt.tight_layout()
plt.show()



from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with attributes and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Create and fit a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X, y)

# Calculate attribute contributions (probability ratios)
attribute_contributions = pd.DataFrame({'Attribute': X.columns, 'Contribution': nb.feature_log_prob_[1] - nb.feature_log_prob_[0]})

# Sort attribute contributions by descending order
attribute_contributions = attribute_contributions.sort_values(by='Contribution', ascending=False)

# Print attribute contributions
print(attribute_contributions)



from sklearn.linear_model import Lasso
import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with attributes and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Create and fit a Lasso Regression model
lasso = Lasso(alpha=0.01)
lasso.fit(X, y)

# Get attribute contributions from the non-zero coefficients
attribute_contributions = pd.DataFrame({'Attribute': X.columns, 'Contribution': lasso.coef_})

# Sort attribute contributions by descending order
attribute_contributions = attribute_contributions.sort_values(by='Contribution', ascending=False)

# Print attribute contributions
print(attribute_contributions)


from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with attributes and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Create and fit a Decision Tree classifier
dt = DecisionTreeClassifier()
dt.fit(X, y)

# Calculate permutation importances
result = permutation_importance(dt, X, y, n_repeats=10, random_state=0)

# Get attribute importances from the permutation importances
attribute_importances = pd.DataFrame({'Attribute': X.columns, 'Importance': result.importances_mean})





import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with columns 'attribute1', 'attribute2', and 'target'

# Group the data by the target column and calculate the count for each attribute combination
grouped = df.groupby('target')[['attribute1', 'attribute2']].count()

# Calculate the total count for each target
target_counts = df['target'].value_counts().rename('total_count')

# Calculate the percentage of each attribute combination for each target
percentage_df = grouped.div(target_counts, axis=0) * 100

# Print the percentage of attribute combinations for each target
print(percentage_df)


import pandas as pd

# Assuming you have a DataFrame called 'original_df'

# Create an empty DataFrame to store the rows with non-null values
new_df = pd.DataFrame(columns=original_df.columns)

# Iterate over the rows of the original DataFrame
for index, row in original_df.iterrows():
    if pd.notnull(row['column_name']):  # Replace 'column_name' with the name of your column
        new_df = new_df.append(row)

# Reset the index of the new DataFrame
new_df = new_df.reset_index(drop=True)


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Assuming you have preprocessed and scaled unsupervised data in a dataframe called 'df'

# Apply dimensionality reduction using PCA
pca = PCA(n_components=2)  # You can adjust the number of components as needed
pca_result = pca.fit_transform(df)

# Apply different clustering algorithms
kmeans = KMeans(n_clusters=3)
dbscan = DBSCAN(eps=0.5, min_samples=5)
agg_cluster = AgglomerativeClustering(n_clusters=3)

# Fit the clustering models
kmeans_labels = kmeans.fit_predict(pca_result)
dbscan_labels = dbscan.fit_predict(pca_result)
agg_cluster_labels = agg_cluster.fit_predict(pca_result)

# Evaluate the clustering results using silhouette score
kmeans_score = silhouette_score(pca_result, kmeans_labels)
dbscan_score = silhouette_score(pca_result, dbscan_labels)
agg_cluster_score = silhouette_score(pca_result, agg_cluster_labels)

# Print the silhouette scores for each clustering algorithm
print(f"KMeans Silhouette Score: {kmeans_score}")
print(f"DBSCAN Silhouette Score: {dbscan_score}")
print(f"Agglomerative Clustering Silhouette Score: {agg_cluster_score}")


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Assuming you have preprocessed and scaled unsupervised data in a dataframe called 'df'

# Apply standardization to the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df)

# Apply PCA
pca = PCA()
pca.fit(scaled_data)

# Calculate the cumulative explained variance ratio
cumulative_variance_ratio = np.cumsum(pca.explained_variance_ratio_)

# Plot the explained variance ratio
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), cumulative_variance_ratio, marker='o', linestyle='--')
plt.xlabel('Number of Principal Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Cumulative Explained Variance Ratio by Principal Components')
plt.grid(True)
plt.show()


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import VotingClassifier

# Assuming you have preprocessed and label-encoded data in a dataframe called 'df'

# Split the data into features (X) and target variable (y)
X = df.drop('target_variable', axis=1)  # Replace 'target_variable' with the name of your target variable column
y = df['target_variable']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a list of models to use
models = [
    ('Random Forest', RandomForestClassifier()),
    ('Gradient Boosting', GradientBoostingClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('Logistic Regression', LogisticRegression()),
    ('SVM', SVC())
]

# Train and evaluate each model
for model_name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f'{model_name} Accuracy: {accuracy}')

    # Get feature importances if available
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
        print(f'{model_name} Feature Importances:')
        print(feature_importances)

# Optimize the ensemble
parameters = {
    'Random Forest__n_estimators': [100, 200, 300],  # Example hyperparameters for Random Forest
    'Gradient Boosting__n_estimators': [100, 200, 300],  # Example hyperparameters for Gradient Boosting
}

ensemble_model = GridSearchCV(
    VotingClassifier(estimators=models),  # Use VotingClassifier for ensemble
    param_grid=parameters,
    cv=5
)

ensemble_model.fit(X_train, y_train)
y_pred = ensemble_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Ensemble Accuracy: {accuracy}')



conda install -c conda-forge jupyter_contrib_nbextensions
import math
import numpy as np

def get_optimal_bins(prices):
    # Determine the optimal number of bins using the square root rule
    num_bins = int(round(math.sqrt(len(prices))))

    # Create the optimal bins
    _, bin_edges = np.histogram(prices, bins=num_bins)

    # Replace the bin edges with bin labels
    bin_labels = [f'Bin{i}' for i in range(1, num_bins+1)]

    # Substitute the bin labels in the prices list
    binned_prices = []
    for price in prices:
        bin_idx = np.searchsorted(bin_edges, price, side='right')
        if bin_idx == len(bin_edges):
            binned_prices.append(bin_labels[bin_idx-2])
        elif bin_idx == 0:
            binned_prices.append(bin_labels[bin_idx])
        else:
            binned_prices.append(bin_labels[bin_idx-1])

    # Return the binned prices and the optimal number of bins
    return binned_prices, num_bins

# Example list of prices
prices = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]

# Obtain the binned prices and number of bins
binned_prices, num_bins = get_optimal_bins(prices)

# Display the binned prices and number of bins
print("Binned Prices:", binned_prices)
print("Optimal Number of Bins:", num_bins)
