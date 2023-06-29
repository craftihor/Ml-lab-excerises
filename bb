import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Assuming you have a preprocessed DataFrame called 'df' with features and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Convert the encoded labels to one-hot encoded format
y_encoded_onehot = np_utils.to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_onehot, test_size=0.2, random_state=42)

# Define the Deep Learning model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Extract feature importance
importance_scores = np.abs(model.layers[0].get_weights()[0]).mean(axis=1)

# Create a DataFrame to store feature importance
feature_importance = pd.DataFrame({'Feature': X.columns, 'Importance': importance_scores})

# Sort feature importance in descending order
feature_importance = feature_importance.sort_values(by='Importance', ascending=False)

# Print feature importance
print(feature_importance)





import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

# Assuming you have a preprocessed DataFrame called 'df' with features and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Encode the target variable
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
# Convert the encoded labels to one-hot encoded format
y_encoded_onehot = np_utils.to_categorical(y_encoded)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded_onehot, test_size=0.2, random_state=42)

# Define the Deep Learning model
model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)






from sklearn.naive_bayes import MultinomialNB
import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with attributes and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Create and fit a Naive Bayes classifier
nb = MultinomialNB()
nb.fit(X, y)

# Create a DataFrame to store the attribute contributions
attribute_contributions = pd.DataFrame(index=X.columns)

# Iterate over each class
for class_label in nb.classes_:
    # Calculate attribute contributions (probability ratios) for the current class
    class_contributions = nb.feature_log_prob_[class_label] - nb.feature_log_prob_.sum(axis=0)
    # Assign the contributions to the corresponding class column in the DataFrame
    attribute_contributions[class_label] = class_contributions

# Sort attribute contributions by descending order for each class
attribute_contributions = attribute_contributions.sort_values(by=nb.classes_[0], ascending=False)

# Print attribute contributions
print(attribute_contributions)



from itertools import combinations
import numpy as np

# Assuming you have already trained a Random Forest model 'rf' on your data
# Assuming you have a list of column names 'column_names'

# Get the feature importances from the Random Forest
feature_importances = rf.feature_importances_

# Get the indices of the top-k most important features
top_k_features = np.argsort(feature_importances)[::-1][:k]

# Create a list to store the top feature combinations
top_feature_combinations = []

# Iterate through different combination lengths
for r in range(1, k+1):
    # Generate all possible combinations of length r using the top-k features
    combinations_r = combinations(top_k_features, r)
    
    # Store the combination, its corresponding importance score, and column names
    for combination in combinations_r:
        combination_column_names = [column_names[idx] for idx in combination]
        top_feature_combinations.append((combination_column_names, combination, np.sum(feature_importances[list(combination)])))

# Sort the feature combinations based on their importance scores in descending order
top_feature_combinations.sort(key=lambda x: x[2], reverse=True)

# Print the top feature combinations, column names, and importance scores
for column_names, combination, importance_score in top_feature_combinations:
    print(f"Feature Combination: {column_names}")
    print(f"Feature Indices: {combination}")
    print(f"Importance Score: {importance_score}")
    print("--------------")



from itertools import combinations
import numpy as np

# Assuming you have already trained a Random Forest model 'rf' on your encoded data
# And you have an encoding map 'encoding_map' that maps encoded feature indices to their original feature names

# Get the feature importances from the Random Forest
feature_importances = rf.feature_importances_

# Get the indices of the top-k most important features
top_k_features = np.argsort(feature_importances)[::-1][:k]

# Map the feature indices to their original feature names
top_k_feature_names = [encoding_map[i] for i in top_k_features]

# Create a list to store the top feature combinations
top_feature_combinations = []

# Iterate through different combination lengths
for r in range(1, k+1):
    # Generate all possible combinations of length r using the top-k features
    combinations_r = combinations(top_k_feature_names, r)
    
    # Store the combination and its corresponding importance score
    for combination in combinations_r:
        top_feature_combinations.append((combination, np.sum(feature_importances[top_k_features])))

# Sort the feature combinations based on their importance scores in descending order
top_feature_combinations.sort(key=lambda x: x[1], reverse=True)

# Print the top feature combinations and their importance scores
for combination, importance_score in top_feature_combinations:
    print(f"Feature Combination: {combination}, Importance Score: {importance_score}")



from itertools import combinations
import numpy as np

# Assuming you have already trained a Random Forest model 'rf' on your data

# Get the feature importances from the Random Forest
feature_importances = rf.feature_importances_

# Get the indices of the top-k most important features
top_k_features = np.argsort(feature_importances)[::-1][:k]

# Create a list to store the top feature combinations
top_feature_combinations = []

# Iterate through different combination lengths
for r in range(1, k+1):
    # Generate all possible combinations of length r using the top-k features
    combinations_r = combinations(top_k_features, r)
    
    # Store the combination and its corresponding importance score
    for combination in combinations_r:
        top_feature_combinations.append((combination, np.sum(feature_importances[list(combination)])))

# Sort the feature combinations based on their importance scores in descending order
top_feature_combinations.sort(key=lambda x: x[1], reverse=True)

# Print the top feature combinations and their importance scores
for combination, importance_score in top_feature_combinations:
    print(f"Feature Combination: {combination}, Importance Score: {importance_score}")




import json
from sklearn.preprocessing import LabelEncoder

# Load the JSON object containing the data to be encoded
with open('data.json') as json_file:
    data = json.load(json_file)

# Load the existing encoding dictionary from a JSON file
with open('encoding.json') as json_file:
    encoding_dict = json.load(json_file)

# Create a new LabelEncoder instance
label_encoder = LabelEncoder()

# Iterate through each key-value pair in the data JSON
for key, value in data.items():
    # Check if the key is present in the encoding dictionary
    if key in encoding_dict:
        # Use the existing encoding for the value
        encoded_value = encoding_dict[key][value]
    else:
        # Fit the label encoder on the new values and update the encoding dictionary
        label_encoder.fit([value])
        encoding_dict[key] = {value: label_encoder.transform([value])[0]}
        encoded_value = encoding_dict[key][value]

    # Assign the encoded value back to the data JSON
    data[key] = encoded_value

# Save the updated encoding dictionary to a JSON file
with open('encoding.json', 'w') as json_file:
    json.dump(encoding_dict, json_file)

# Save the encoded data JSON to a file
with open('encoded_data.json', 'w') as json_file:
    json.dump(data, json_file)



from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

# Assuming you have a DataFrame 'data' containing the features and a Series 'target' containing the multinomial target values

# Create a Random Forest classifier
rf = RandomForestClassifier()

# Perform Recursive Feature Elimination (RFE) with the Random Forest classifier
rfe = RFE(estimator=rf, n_features_to_select=10)
rfe.fit(data, target)

# Get the selected features
selected_features = data.columns[rfe.support_]

# Combine selected features with their respective target values
selected_data = pd.concat([data[selected_features], target], axis=1)

# Display the top 10 combinations of features and their target values
top_10_combinations = selected_data.head(10)
print(top_10_combinations)



from catboost import CatBoostClassifier

# Assuming you have trained a CatBoost classifier and stored it in the variable 'model'

# Get the feature importance
feature_importance = model.get_feature_importance(prettified=True)

# Specify the target attribute feature
target_attribute = 'target_feature_name'  # Replace with the actual name of the target attribute feature

# Filter the feature importance by the target attribute
target_feature_importance = feature_importance[feature_importance['Feature Id'].str.contains(target_attribute)]

# Sort the feature importance in descending order
sorted_target_feature_importance = target_feature_importance.sort_values(by='Importances', ascending=False)

# Print the sorted feature importance
print(sorted_target_feature_importance)



from catboost import CatBoostClassifier
import matplotlib.pyplot as plt

# Assuming you have trained a CatBoost classifier and stored it in the variable 'model'

# Print the classification tree
model.plot_tree(tree_idx=0, pool=None, figsize=(20, 20), feature_names=None, plot=True)

# Get the feature importance
feature_importance = model.get_feature_importance(prettified=True)

# Print the feature importance
print(feature_importance)

# Visualize the feature importance
plt.figure(figsize=(10, 6))
plt.barh(feature_importance['Feature Id'], feature_importance['Importances'])
plt.xlabel('Feature Importance')
plt.ylabel('Features')
plt.title('CatBoost Feature Importance')
plt.show()



import pandas as pd
from sklearn.model_selection import train_test_split
from catboost import CatBoostClassifier

# Assuming you have a DataFrame 'df' with features and 'target' column

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(df.drop('target', axis=1), df['target'], test_size=0.2, random_state=42)

# Identify the categorical features
categorical_features = ['cat_feature1', 'cat_feature2']  # Replace with the actual column names of categorical features

# Create a CatBoost classifier
model = CatBoostClassifier(iterations=100, learning_rate=0.1, depth=6, loss_function='MultiClass')

# Fit the model to the training data
model.fit(X_train, y_train, cat_features=categorical_features, verbose=100)

# Make predictions on the test data
predictions = model.predict(X_test)

# Evaluate the model
accuracy = (predictions == y_test).mean()
print("Accuracy:", accuracy)


import math
import numpy as np

def get_optimal_bins(prices):
    # Determine the optimal number of bins using the square root rule
    num_bins = int(round(math.sqrt(len(prices))))

    # Create the optimal bins
    counts, bin_edges = np.histogram(prices, bins=num_bins)

    # Replace the bin edges with bin labels
    bin_labels = [f'Bin{i}' for i in range(1, num_bins + 1)]

    # Substitute the bin labels in the prices list
    binned_prices = []
    for price in prices:
        bin_idx = np.searchsorted(bin_edges, price, side='right')
        if bin_idx == len(bin_edges):
            binned_prices.append(bin_labels[bin_idx - 2])
        elif bin_idx == 0:
            binned_prices.append(bin_labels[bin_idx])
        else:
            binned_prices.append(bin_labels[bin_idx - 1])

    # Return the binned prices, optimal number of bins, and bin edges
    return binned_prices, num_bins, bin_edges

# Example usage
prices = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
binned_prices, num_bins, bin_edges = get_optimal_bins(prices)

# Save the bin range
bin_range = [bin_edges[0], bin_edges[-1]]
print("Bin Range:", bin_range)



import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Assuming you have a DataFrame 'df' with categorical variables

# Create a new DataFrame to store the encoded values
df_encoded = df.copy()

# Initialize a dictionary to store the label encoding mappings
encoding_mappings = {}

# Iterate over each column in the DataFrame
for column in df.columns:
    if df[column].dtype == 'object':  # Check if the column contains categorical values
        # Initialize a LabelEncoder for the column
        encoder = LabelEncoder()
        
        # Fit and transform the column to obtain the encoded values
        encoded_values = encoder.fit_transform(df[column])
        
        # Store the encoded values in the new DataFrame
        df_encoded[column] = encoded_values
        
        # Store the label encoding mappings for future reference
        encoding_mappings[column] = dict(zip(encoder.classes_, encoder.transform(encoder.classes_)))

# Save the encoded DataFrame and the encoding mappings to separate files
df_encoded.to_csv('encoded_data.csv', index=False)
pd.DataFrame.from_dict(encoding_mappings).to_csv('encoding_mappings.csv', index=False)





import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Assuming you have the data stored in X_train, X_test, y_train, and y_test

lgb_params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'bagging_fraction': 0.9,
    'bagging_frequency': 1,
    'cat_smooth': 70,
    'feature_fraction': 0.9,
    'learning_rate': 0.01,
    'min_child_samples': 20,
    'min_data_per_group': 100,
    'num_leaves': 18,
    'metric': 'auc',
    'unbalance': True
}

n_models = 5  # Number of models to train for bagging
oof_lgb = np.zeros(len(X_train))
pred_lgb = np.zeros(len(X_test))

scores = []

feature_importances_gain = pd.DataFrame()
feature_importances_gain['feature'] = X_train.columns

feature_importances_split = pd.DataFrame()
feature_importances_split['feature'] = X_train.columns

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for model in range(n_models):
    print("Training Model: ", model+1)
    oof_model = np.zeros(len(X_train))
    pred_model = np.zeros(len(X_test))
    
    for fold_, (train_ind, val_ind) in enumerate(folds.split(X_train, y_train)):
        print("Fold:", fold_+1)
        trn_data = lgb.Dataset(X_train.iloc[train_ind], label=y_train.iloc[train_ind], categorical_feature=cats)  # Specify categorical feature for lgb
        val_data = lgb.Dataset(X_train.iloc[val_ind], label=y_train.iloc[val_ind], categorical_feature=cats)  # Specify categorical feature for lgb
        
        lgb_params['seed'] = model * 10 + fold_  # Set a different random seed for each model-fold combination
        
        lgb_clf = lgb.train(lgb_params, trn_data, num_boost_round=1000, valid_sets=(trn_data, val_data), verbose_eval=100, early_stopping_rounds=100)
        oof_model[val_ind] = lgb_clf.predict(X_train.iloc[val_ind], num_iteration=lgb_clf.best_iteration)
        print("Fold:", fold_+1, "roc_auc =", roc_auc_score(y_train.iloc[val_ind], oof_model[val_ind]))
        scores.append(roc_auc_score(y_train.iloc[val_ind], oof_model[val_ind]))
        
        feature_importances_gain['fold{}_{}'.format(model+1, fold_ + 1)] = lgb_clf.feature_importance(importance_type='gain')
        feature_importances_split['fold{}_{}'.format(model+1, fold_ + 1)] = lgb_clf.feature_importance(importance_type='split')
        pred_model += lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration) / folds.n_splits
    
    oof_lgb += oof_model / n_models
    pred_lgb += pred_model / n_models

print(' \\\\\\\\\\\\\\\ model roc_auc ////////////// : ', np.mean(scores))

np.save('oof_lgb', oof_l



import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Assuming you have the data stored in X_train, X_test, y_train, and y_test

lgb_params = {
    'objective': 'binary',
    'boosting': 'gbdt',
    'bagging_fraction': 0.9,
    'bagging_frequency': 1,
    'cat_smooth': 70,
    'feature_fraction': 0.9,
    'learning_rate': 0.01,
    'min_child_samples': 20,
    'min_data_per_group': 100,
    'num_leaves': 18,
    'metric': 'auc',
    'unbalance': True
}

oof_lgb = np.zeros(len(X_train))
pred_lgb = np.zeros(len(X_test))

scores = []

feature_importances_gain = pd.DataFrame()
feature_importances_gain['feature'] = X_train.columns

feature_importances_split = pd.DataFrame()
feature_importances_split['feature'] = X_train.columns

folds = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

for fold_, (train_ind, val_ind) in enumerate(folds.split(X_train, y_train)):
    print("fold: ---------------------------------------", fold_)
    trn_data = lgb.Dataset(X_train.iloc[train_ind], label=y_train.iloc[train_ind], categorical_feature=cats)  # Specify categorical feature for lgb
    val_data = lgb.Dataset(X_train.iloc[val_ind], label=y_train.iloc[val_ind], categorical_feature=cats)  # Specify categorical feature for lgb

    lgb_clf = lgb.train(lgb_params, trn_data, num_boost_round=1000, valid_sets=(trn_data, val_data), verbose_eval=100, early_stopping_rounds=100)
    oof_lgb[val_ind] = lgb_clf.predict(X_train.iloc[val_ind], num_iteration=lgb_clf.best_iteration)
    print("fold:", fold_, "roc_auc =", roc_auc_score(y_train.iloc[val_ind], oof_lgb[val_ind]))
    scores.append(roc_auc_score(y_train.iloc[val_ind], oof_lgb[val_ind]))

    feature_importances_gain['fold_{}'.format(fold_ + 1)] = lgb_clf.feature_importance(importance_type='gain')
    feature_importances_split['fold_{}'.format(fold_ + 1)] = lgb_clf.feature_importance(importance_type='split')
    pred_lgb += lgb_clf.predict(X_test, num_iteration=lgb_clf.best_iteration) / folds.n_splits

print(' \\\\\\\\\\\\\\\ model roc_auc ////////////// : ', np.mean(scores))

np.save('oof_lgb', oof_lgb)
np.save('pred_lgb', pred_lgb)





from sklearn.linear_model import LassoCV
import numpy as np
import pandas as pd

# Assuming you have a preprocessed and label-encoded DataFrame called 'df' with attributes and target columns

# Separate the features and target variable
X = df.drop('target', axis=1)  # Replace 'target' with the name of your target variable column
y = df['target']

# Create a range of alpha values to be tested
alphas = np.logspace(-4, 0, 50)

# Create and fit the LassoCV model
lasso_cv = LassoCV(alphas=alphas, cv=5)
lasso_cv.fit(X, y)

# Get the optimal alpha value
optimal_alpha = lasso_cv.alpha_

# Print the optimal alpha value
print("Optimal Alpha:", optimal_alpha)



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
