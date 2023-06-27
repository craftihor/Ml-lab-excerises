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
