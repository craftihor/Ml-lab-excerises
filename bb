import dash
from interpret.ext.dash import DashInterpretation
from interpret_community import ExplanationDashboard
import requests

# Model predictions
preds = model.predict(X_test)

# Create interpretations
interpretation = interpret_ml.explain_prediction(model, X_test[0], preds[0])

# Dash app
app = dash.Dash(__name__)
app.layout = dash_html_components.Div([
    ExplanationDashboard(interpretation, persist_state=True)
]) 

# Run app
if __name__ == '__main__':
    app.run_server(port=8050, debug=True)

# Download page
response = requests.get('http://127.0.0.1:8050/')
with open('dashboard.html', 'w') as f:
    f.write(response.text)



import os
import subprocess

def install_whl_files(folder_path):
    # List all files in the folder
    files = os.listdir(folder_path)

    # Filter only .whl files
    whl_files = [file for file in files if file.endswith(".whl")]

    # Install each .whl file without dependencies
    for whl_file in whl_files:
        whl_path = os.path.join(folder_path, whl_file)
        try:
            subprocess.run(["conda", "install", "--use-local", "--no-deps", whl_path], check=True)
            print(f"Successfully installed: {whl_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error installing {whl_file}: {e}")
            continue

if __name__ == "__main__":
    folder_path = "/path/to/your/folder"  # Replace with the actual folder path
    install_whl_files(folder_path)



Sub CreatePresentation()
    Dim pptApp As PowerPoint.Application
    Dim pptPres As PowerPoint.Presentation
    Dim pptSlide As PowerPoint.Slide
    Dim slideIndex As Integer
    
    ' Create PowerPoint application
    Set pptApp = New PowerPoint.Application
    pptApp.Visible = True ' You can set it to False if you don't want to display PowerPoint during execution
    
    ' Create a new presentation
    Set pptPres = pptApp.Presentations.Add
    
    ' Slide 1 - Title Slide
    slideIndex = 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Presentation Title"
    ' Customize the title slide with your presentation title and any other details as needed
    
    ' Slide 2 - Agenda
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Agenda"
    ' Add agenda content
    
    ' Slide 3 - Problem Statement
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Problem Statement"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Inaccurate trade classification is leading to significant losses annually. The company faces over $1M in losses each year due to misclassified trades. Additionally, analysts spend 15 hours per week manually evaluating trades using subjective methods and outdated statistical models. There is an urgent need to improve trade classification."
    
    ' Slide 4 - Current Trade Classification Challenges
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Current Trade Classification Challenges"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "The current trade classification process has an accuracy rate of only 65%. The overreliance on manual evaluation leads to subjectivity and human biases. Existing statistical models fail to account for real-time market changes. There is no automated monitoring system to detect misclassifications."
    
    ' Slide 5 - Proposed Machine Learning Solution
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Proposed Machine Learning Solution"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "To address these challenges, I propose building a machine learning model for automated trade classification. The benefits of this approach include improved accuracy through data-driven insights, continual learning and adaptation to market changes, and enabling proactive trade strategies through ongoing model predictions."
    
    ' Slide 6 - Bagging Classifier Model
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Bagging Classifier Model"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "After evaluating various algorithms, I selected the Bagging Classifier model for this problem. It is an ensembling technique that combines predictions from multiple base models to improve overall performance. Bagging reduces variance and overfitting."
    ' Add diagram of model architecture
    
    ' Slide 7 - Data Collection and Preprocessing
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Data Collection and Preprocessing"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "To train the model, I sourced trade data from company SQL databases. I cleaned the data by handling missing values and removing outliers. Categorical variables were encoded for modeling."
    
    ' Slide 8 - Feature Engineering
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Feature Engineering"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Based on correlation analysis, I identified 10 key features related to trade performance. Additional features were derived, such as technical indicators, to capture important trade patterns. Feature values were normalized."
    
    ' Slide 9 - Model Training and Evaluation
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Model Training and Evaluation"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "The data was split 70/30 into train and test sets. The model was trained on 70% data and evaluated on the unseen 30% test data."
    
    ' Slide 10 - Model Performance Metrics
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Model Performance Metrics"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "On the test set, the model achieved 95% accuracy, 90% precision, 85% recall and 0.9 F1 score."
    ' Add confusion matrix and ROC curve
    
    ' Slide 11 - Cost Matrix
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Cost Matrix"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "A cost matrix was incorporated to minimize costly misclassifications and find the optimal balance between precision and recall for the business context."
    
    ' Slide 12 - Business Impact
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Business Impact"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "Applying this model is projected to avoid $1.5M in losses annually by accurately flagging trades expected to fail. It would save analysts 15 hours per week for more high-value tasks."
    
    ' Slide 13 - Cost-Benefit Analysis
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Cost-Benefit Analysis"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "With $100K in model development costs, it is projected to deliver $1.5M in savings in the first year alone, representing a 12 month ROI."
    
    ' Slide 14 - Implementation Plan
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Implementation Plan"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "The model would be integrated with existing trade platforms via APIs. It would be rolled out sequentially across divisions. Ongoing monitoring and maintenance would ensure continual effectiveness."
    
    ' Slide 15 - Top Priorities for Improvement
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Top Priorities for Improvement"
    pptSlide.Shapes(2).TextFrame.TextRange.Text = "1. Incorporate additional data sources" & vbCrLf & "2. Online learning for real-time model adaptation" & vbCrLf & "3. Build mobile interface for alerts"
    
    ' Slide 16 - Key Learnings
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Key Learnings"
    ' Add key learnings content
    
    ' Slide 17 - Q&A
    slideIndex = slideIndex + 1
    Set pptSlide = pptPres.Slides.Add(slideIndex, ppLayoutTitleOnly)
    pptSlide.Shapes(1).TextFrame.TextRange.Text = "Q&A"
    ' Add Q&A content
    
    ' Save the presentation
    Dim outputPath As String
    outputPath = "C:\path\to\your\output\presentation.pptx" ' Specify the desired output path and filename
    pptPres.SaveAs outputPath
    
    ' Clean up and release objects
    pptPres.Close
    pptApp.Quit
    Set pptSlide = Nothing
    Set pptPres = Nothing
    Set pptApp = Nothing
End Sub











import os
import subprocess

whl_directory = '/path/to/whl_directory'

# Iterate over the files in the directory
for filename in os.listdir(whl_directory):
    if filename.endswith('.whl'):
        whl_file = os.path.join(whl_directory, filename)
        try:
            subprocess.run(['pip', 'install', whl_file], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error occurred during installation of {filename}: {e}")
            print("Attempting to fetch dependencies using Conda...")
            try:
                subprocess.run(['conda', 'install', '<dependency-name>'], check=True)
                print("Conda installation successful.")
            except subprocess.CalledProcessError as e:
                print(f"Error occurred during Conda installation of <dependency-name>: {e}")
                continue


import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime import lime_tabular

# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names
class_names = data.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a classification model (Random Forest in this example)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Create the Lime explainer
explainer = lime_tabular.LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names)

# Select a sample from the test set for explanation
sample_idx = 0
X_sample = X_test[sample_idx]
y_sample = y_test[sample_idx]

# Generate an explanation for the sample
explanation = explainer.explain_instance(X_sample, model.predict_proba, num_features=len(feature_names))

# Display the explanation
explanation.show_in_notebook()






import torch
from torch import nn
from skorch import NeuralNetClassifier

class NDF(nn.Module):

  def __init__(self):
    super().__init__()
    self.trees = nn.ModuleList([self.make_tree() for i in range(10)])
  
  def make_tree(self):
    return NeuralNetClassifier(
      # Define model architecture
      ...
    )

  def forward(self, x):
    tree_preds = [tree(x) for tree in self.trees]
    forest_pred = torch.mean(torch.stack(tree_preds), dim=0)
    return forest_pred

# Training loop
ndf = NDF()
optimizer = torch.optim.Adam(ndf.parameters())  

for epoch in range(100):

  # Boosting
  weights = np.ones(len(X)) / len(X)
  
  for xb, yb, wb in loader:
    pred = ndf(xb)
    loss = clf_loss(pred, yb) * wb
    
    # Update weights
    weights[loss > 0] *= np.exp(loss[loss > 0])

    # Auxiliary reconstruction loss   
    recon_loss = recon_criterion(encoder(xb), xb) 
    loss = loss + recon_loss
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

  # Normalize weights    
  weights = weights / np.sum(weights)

# OOB evaluation
oob_scores = [tree.score(X[oob_idx], y[oob_idx]) for tree, oob_idx in zip(ndf.trees, oob_idx)]  
print("OOB Score:", np.mean(oob_scores))


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from skorch import NeuralNetClassifier
import torch
from torch.nn import Linear, ReLU, Dropout

# Prepare data
X = pd.DataFrame(data)
y = pd.Series(target) 

X_train, X_test, y_train, y_test = train_test_split(X, y)

# Neural Decision Forest model 
class NDF(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.trees = []
        for _ in range(10): 
            tree = RandomForestClassifier()
            self.trees.append(tree)
    
    def forward(self, x):
        x = torch.tensor(x)
        preds = []
        for tree in self.trees:
            p = NeuralNetClassifier(
                module=torch.nn.Sequential(
                    Linear(X.shape[1], 64),
                    ReLU(),
                    Dropout(0.5),
                    Linear(64, 1),
                    ReLU()
                ),
                module__input_dims=[X.shape[1]],
                max_epochs=100,
                lr=0.01
            )
            p.fit(X_train, y_train)
            preds.append(p.predict(x))
        return torch.mean(torch.stack(preds), axis=0) 
        
# Create NDF model
model = NDF()

# Train model
train_loader = torch.utils.data.DataLoader(X_train, y_train) 
optim = torch.optim.Adam(model.parameters(), lr=0.01)
EPOCHS = 100
for i in range(EPOCHS):
    for xb, yb in train_loader:
        loss = model.training_step(xb, yb)
        loss.backward()
        optim.step()
        optim.zero_grad()

# Evaluate    
acc = model.evaluation(X_test, y_test)
print("Test accuracy:", acc)




import pandas as pd
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier

# Prepare data
X = pd.DataFrame(# features)  
y = pd.Series(# target)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define TabNet model
tabnet = TabNetClassifier(optimizer_fn=torch.optim.Adam,
                          optimizer_params=dict(lr=2e-2)) 

# Train model                     
tabnet.fit(
    X_train=X_train, y_train=y_train,
    eval_set=[(X_train, y_train), (X_test, y_test)],
    eval_name=['train', 'valid'],
    eval_metric=['accuracy'],
    max_epochs=100, patience=20)

# Make predictions
preds = tabnet.predict(X_test)

# Evaluate model
accuracy = tabnet.score_func(y_test, preds)
print("Test Accuracy:", accuracy)


from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier

# Define hyperparameter ranges
rf_params = {'n_estimators': [100, 200, 500], 
             'max_depth': [5, 8, 15, 25],
             'min_samples_split': [2, 5, 10],
             'min_samples_leaf': [1, 2, 4]}

# Create randomized search with 3-fold cross validation
rf_search = RandomizedSearchCV(RandomForestClassifier(), 
                               param_distributions=rf_params,
                               n_iter=20, 
                               cv=3,
                               n_jobs=-1) 

# Fit randomized search
rf_search.fit(X_train, y_train)

# Get best model
best_rf = rf_search.best_estimator_

# Evaluate on test set
accuracy = best_rf.score(X_test, y_test)
print("Accuracy:", accuracy)




from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from imblearn.over_sampling import RandomOverSampler
import random

# Generate a synthetic imbalanced dataset
X, y = make_classification(n_samples=1000, n_features=10, weights=[0.9, 0.1], random_state=42)

# Split the dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Inject noise into the dataset
noise_ratio = 0.05  # Percentage of instances to mislabel
num_noise = int(noise_ratio * len(y_train))
indices = random.sample(range(len(y_train)), num_noise)
for index in indices:
    y_train[index] = 1 - y_train[index]  # Flipping the label

# Perform oversampling using RandomOverSampler
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train, y_train)

# Shuffle the resampled data
X_train_resampled, y_train_resampled = shuffle(X_train_resampled, y_train_resampled, random_state=42)

# Train your classification model on the resampled data
# ...

# Evaluate the model on the test set
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))




import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.*;
String filePath = "path/to/your/file.xlsx";
Workbook workbook = new XSSFWorkbook(new FileInputStream(filePath));
Sheet firstSheet = workbook.getSheetAt(0); // Assuming the first sheet is at index 0
Sheet secondSheet = workbook.createSheet("SecondSheet");
int[] columnsToCopy = {0, 2, 4, 6}; // Assuming you want to copy columns A, C, E, and G
int rowCount = 0;
for (Row row : firstSheet) {
    Row newRow = secondSheet.createRow(rowCount);
    int cellCount = 0;
    for (int colIndex : columnsToCopy) {
        Cell cell = row.getCell(colIndex, MissingCellPolicy.RETURN_BLANK_AS_NULL);
        if (cell != null) {
            Cell newCell = newRow.createCell(cellCount);
            newCell.setCellValue(cell.getStringCellValue()); // Assuming the cells contain string values
        }
        cellCount++;
    }
    rowCount++;
}
workbook.removeSheetAt(0); // Remove the first sheet at index 0
String newFilePath = "path/to/new/file.xlsx";
FileOutputStream outputStream = new FileOutputStream(newFilePath);
workbook.write(outputStream);
workbook.close();
outputStream.close();



import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.poi.ss.usermodel.*;
import org.apache.poi.xssf.usermodel.XSSFWorkbook;
import java.io.*;
import java.util.*;

public class CSVToXLSXConverter {

    public static void main(String[] args) {
        String csvFilePath = "path/to/input.csv";
        String xlsxFilePath = "path/to/output.xlsx";

        List<String[]> csvData = readCSVFile(csvFilePath);
        writeXLSXFile(xlsxFilePath, csvData);
        System.out.println("Conversion completed successfully.");
    }

    public static List<String[]> readCSVFile(String csvFilePath) {
        List<String[]> csvData = new ArrayList<>();

        try (Reader reader = new FileReader(csvFilePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT)) {
            for (CSVRecord record : csvParser) {
                String[] fields = new String[record.size()];
                for (int i = 0; i < record.size(); i++) {
                    fields[i] = record.get(i);
                }
                csvData.add(fields);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }

        return csvData;
    }

    public static void writeXLSXFile(String xlsxFilePath, List<String[]> data) {
        try (Workbook workbook = new XSSFWorkbook()) {
            Sheet sheet = workbook.createSheet("Sheet1");
            int rowCount = 0;
            for (String[] rowData : data) {
                Row row = sheet.createRow(rowCount++);
                int columnCount = 0;
                for (String cellData : rowData) {
                    Cell cell = row.createCell(columnCount++);
                    cell.setCellValue(cellData);
                }
            }
            try (FileOutputStream outputStream = new FileOutputStream(xlsxFilePath)) {
                workbook.write(outputStream);
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}





import dask
from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import cross_val_score
from dask.distributed import Client

# Start a Dask client
client = Client()

# Define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

# Define models and parameters
model = BaggingClassifier()
n_estimators = [10, 100, 1000]
param_grid = {'n_estimators': n_estimators}

# Create a Dask delayed object for parallel execution
@dask.delayed
def fit_score_model(model, X, y, params):
    model.set_params(**params)
    return cross_val_score(model, X, y, cv=3, scoring='accuracy').mean()

# Create a list of delayed computations for grid search
delayed_results = []
for params in ParameterGrid(param_grid):
    delayed_result = fit_score_model(model, X, y, params)
    delayed_results.append(delayed_result)

# Compute the results in parallel
results = dask.compute(*delayed_results)

# Summarize results
best_score = max(results)
best_params = list(ParameterGrid(param_grid))[results.index(best_score)]
print("Best: %f using %s" % (best_score, best_params))
for params, score in zip(ParameterGrid(param_grid), results):
    print("%f with: %r" % (score, params))


import dask_ml.model_selection as dcv
from sklearn.datasets import make_blobs
from sklearn.ensemble import BaggingClassifier
from dask.distributed import Client

# Start a Dask client
client = Client()

# Define dataset
X, y = make_blobs(n_samples=1000, centers=2, n_features=100, cluster_std=20)

# Define models and parameters
model = BaggingClassifier()
n_estimators = [10, 100, 1000]

# Define parameter grid
param_grid = {'n_estimators': n_estimators}

# Perform grid search with Dask
grid_search = dcv.GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='accuracy')
with client:
    grid_search.fit(X, y)

# Summarize results
print("Best: %f using %s" % (grid_search.best_score_, grid_search.best_params_))
means = grid_search.cv_results_['mean_test_score']
stds = grid_search.cv_results_['std_test_score']
params = grid_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))





import eli5
from eli5.sklearn import InverseTransformWrapper
from lime.lime_tabular import LimeTabularExplainer
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
data = load_iris()
X, y = data.data, data.target
feature_names = data.feature_names
class_names = data.target_names

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Wrap the classifier for inverse transform with eli5
wrapped_classifier = InverseTransformWrapper(rf_classifier)

# Create a LIME explainer
explainer = LimeTabularExplainer(X_train, feature_names=feature_names, class_names=class_names)

# Choose a random test instance for explanation
instance_idx = 10
instance = X_test[instance_idx]

# Generate explanation with LIME
lime_explanation = explainer.explain_instance(instance, wrapped_classifier.predict_proba)

# Print LIME explanation
print("LIME Explanation:")
print(lime_explanation.as_list())

# Generate explanation with eli5
eli5_explanation = eli5.explain_weights(rf_classifier, feature_names=feature_names)

# Print eli5 explanation
print("eli5 Explanation:")
print(eli5_explanation)




import dask
import dask.dataframe as dd
from dask.distributed import Client
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Start a Dask client
client = Client()

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Convert data to Dask DataFrame
ddf = dd.from_pandas(pd.DataFrame(X), npartitions=4)
ddf['target'] = y

# Create a random forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100)

# Perform cross-validation using Dask and cross_val_score
scores = dask.compute(cross_val_score(rf_classifier, ddf.drop('target', axis=1).values.compute(), ddf['target'].values.compute(), cv=5)) 

# Print the scores
print("Cross-validation scores:", scores)





import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Create a GPU-accelerated random forest classifier using xgboost
params = {
    'tree_method': 'gpu_hist',  # Specify the GPU training method
    'n_estimators': 100
}
rf_classifier = xgb.XGBClassifier(**params)

# Perform cross-validation using cross_val_score
scores = cross_val_score(rf_classifier, X, y, cv=5)  # Use 5-fold cross-validation

# Print the scores
print("Cross-validation scores:", scores)




from sklearn.kernel_approximation import Nystroem
from sklearn.linear_model import SGDClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Step 1: Generate synthetic classification data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# Step 2: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Nystroem approximation
nystroem = Nystroem(n_components=100, random_state=42)
X_train_transformed = nystroem.fit_transform(X_train)
X_test_transformed = nystroem.transform(X_test)

# Step 4: SGDClassifier
sgd_classifier = SGDClassifier(loss='hinge', alpha=0.001, random_state=42)
sgd_classifier.fit(X_train_transformed, y_train)

# Step 5: Predict using the trained model
y_pred = sgd_classifier.predict(X_test_transformed)

# Step 6: Evaluate the model performance
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)




from tensorflow import keras
from tensorflow.keras import regularizers

model = keras.Sequential()
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01), input_shape=(input_dim,)))
model.add(keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(keras.layers.Dense(1, activation='sigmoid'))



import numpy as np

def get_optimal_bins(prices):
    # Set the desired range for the bin edges
    bin_range = 20

    # Calculate the number of bins based on the desired range
    num_bins = int(np.ceil((np.max(prices) - np.min(prices)) / bin_range))

    # Create the bin edges
    bin_edges = np.linspace(np.min(prices), np.max(prices), num_bins + 1)

    # Replace the bin edges with bin labels
    bin_labels = [f'Bin{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(num_bins)]

    # Substitute the bin labels in the prices list
    binned_prices = []
    for price in prices:
        bin_idx = np.searchsorted(bin_edges, price, side='right')
        if bin_idx == len(bin_edges):
            binned_prices.append(bin_labels[bin_idx - 2])
        elif bin_idx == 0:
            binned_prices.append(bin_labels[bin_idx - 1])
        else:
            binned_prices.append(bin_labels[bin_idx - 1])

    # Return the binned prices, optimal number of bins, and bin edges
    return binned_prices, num_bins, bin_edges


num_bins = int(math.ceil(math.log2(len(prices)) + 1))

    # Create the bin edges
    bin_edges = np.linspace(np.min(prices), np.max(prices), num_bins + 1)

    # Replace



import math
import numpy as np

def get_optimal_bins(prices):
    # Calculate the optimal number of bins using the Freeman-Diaconis rule
    IQR = np.percentile(prices, 75) - np.percentile(prices, 25)
    num_bins = int(round((2 * IQR) / math.pow(len(prices), 1/3)))

    # Create the optimal bins
    counts, bin_edges = np.histogram(prices, bins=num_bins)

    # Replace the bin edges with bin labels
    bin_labels = [f'Bin{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}' for i in range(len(bin_edges) - 1)]

    # Substitute the bin labels in the prices list
    binned_prices = []
    for price in prices:
        bin_idx = np.searchsorted(bin_edges, price, side='right')
        if bin_idx == len(bin_edges):
            binned_prices.append(bin_labels[bin_idx - 2])
        elif bin_idx == 0:
            binned_prices.append(bin_labels[bin_idx - 1])
        else:
            binned_prices.append(bin_labels[bin_idx - 1])

    # Return the binned prices, optimal number of bins, and bin edges
    return binned_prices, num_bins, bin_edges




from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Undersampling
undersampler = RandomUnderSampler(random_state=42)
X_train_resampled, y_train_resampled = undersampler.fit_resample(X_train, y_train)

# Oversampling
oversampler = RandomOverSampler(random_state=42)
X_train_resampled, y_train_resampled = oversampler.fit_resample(X_train_resampled, y_train_resampled)

# Train your model on the resampled data
model.fit(X_train_resampled, y_train_resampled)

# Make predictions on the test data
y_pred = model.predict(X_test)

# Evaluate the model's performance
print(classification_report(y_test, y_pred))



import pandas as pd

# Create a sample DataFrame
data = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

# Iterate through rows and make changes
for index, row in data.iterrows():
    # Access the values of each column in the row
    value_A = row['A']
    value_B = row['B']
    
    # Perform some operations on the row
    new_value = value_A + value_B
    
    # Add a new column to the row
    row['C'] = new_value

# Print the updated DataFrame
print(data)





import pandas as pd

# Create an empty DataFrame
df = pd.DataFrame(columns=['Column'])

# Assuming you have the string stored in a variable called 'data_string'
data_string = 'gdhdf|"dfs"|"ghdf'

# Split the string by the separator "|"
entries = data_string.split('|')

# Remove the surrounding double quotes from each entry
entries = [entry.strip('"') for entry in entries]

# Insert the entries into the DataFrame
df['Column'] = entries

# Print the DataFrame
print(df)




import pandas as pd
import numpy as np

# Assuming you have a DataFrame called 'data' with a numerical column named 'value'

# Automatically calculate the number of bins using the Freedman-Diaconis rule
q75, q25 = np.percentile(data['value'], [75 ,25])
iqr = q75 - q25
bin_width = 2 * iqr / (len(data) ** (1/3))
num_bins = int((data['value'].max() - data['value'].min()) / bin_width)

# Perform automatic binning using pandas cut() function
bin_labels = []  # List to store bin labels

# Use the cut() function to automatically bin the 'value' column
data['bin'], bins = pd.cut(data['value'], bins=num_bins, retbins=True, labels=False)

# Generate bin range labels using the bin ranges
for i in range(len(bins) - 1):
    bin_labels.append(f'{bins[i]:.2f}-{bins[i+1]:.2f}')

# Replace the bin numbers with the bin range labels in the 'bin' column
data['bin'] = pd.Series(bin_labels)[pd.cut(data['value'], bins=num_bins, labels=False)]

# Print the DataFrame with bin range labels
print(data)




import numpy as np 

import pandas as pd 

data = pd.DataFrame(data=pd.read_csv('trainingexamples.csv')) 

concepts = np.array(data.iloc[:,0:-1])

print(concepts) 

target = np.array(data.iloc[:,-1])  

print(target)

def learn(concepts, target): 

    specific_h = concepts[0].copy()     

    print("initialization of specific_h and general_h")     

    print(specific_h)  

    general_h = [["?" for i in range(len(specific_h))] for i in range(len(specific_h))]     

    print(general_h)  

    for i, h in enumerate(concepts): 

        if target[i] == "Y": 

            for x in range(len(specific_h)): 

                if h[x]!= specific_h[x]:                    

                    specific_h[x] ='?'                     

                    general_h[x][x] ='?'

                print(specific_h)

        print(specific_h)

        if target[i] == "N":            

            for x in range(len(specific_h)): 

                if h[x]!= specific_h[x]:                    

                    general_h[x][x] = specific_h[x]                

                else:                    

                    general_h[x][x] = '?'        

        print(" steps of Candidate Elimination Algorithm",i+1)        

        print(specific_h)         

        print(general_h)  

    indices = [i for i, val in enumerate(general_h) if val == ['?', '?', '?', '?', '?', '?']]    

    for i in indices:   

        general_h.remove(['?', '?', '?', '?', '?', '?']) 

    return specific_h, general_h 

s_final, g_final = learn(concepts, target)

print("Final Specific_h:", s_final, sep="\n")

print("Final General_h:", g_final, sep="\n") 

#data.head()  






Sure! Here's a brief overview of Lasso regression and its attribute importances:

Lasso Regression:
- Lasso stands for "Least Absolute Shrinkage and Selection Operator."
- It is a linear regression model that performs both regularization and feature selection.
- Lasso regression adds a regularization term to the ordinary least squares (OLS) cost function, which penalizes the magnitude of the coefficients.
- The regularization term encourages the model to minimize the sum of the absolute values of the coefficients, resulting in sparse solutions where less important features have coefficients reduced to zero.
- Lasso regression is useful for feature selection, as it automatically identifies and selects the most relevant features for prediction.

Attribute Importances in Lasso:
- In Lasso regression, the importance of attributes is measured by the magnitude of their corresponding coefficients.
- The higher the absolute value of the coefficient, the more important the attribute is in determining the outcome.
- Attributes with non-zero coefficients are considered important, as they contribute significantly to the prediction.
- Attributes with zero coefficients are considered unimportant and are effectively excluded from the model.
- Lasso provides a way to rank attributes based on their importances, where larger coefficients indicate higher importances and smaller coefficients indicate lower importances.

It's important to note that Lasso regression assumes a linear relationship between the attributes and the target variable. If the relationship is non-linear, other regression models or techniques might be more suitable. Additionally, Lasso assumes that the attributes are linearly independent and free from multicollinearity.

Sure! Here's a brief overview of Random Forest and its attribute importances:

Random Forest:
- Random Forest is an ensemble learning algorithm that combines multiple decision trees to make predictions.
- It is a versatile and powerful algorithm that can be used for both regression and classification tasks.
- Random Forest builds a collection of decision trees, where each tree is trained on a random subset of the data and features.
- The predictions from individual trees are combined using voting (classification) or averaging (regression) to make the final prediction.
- Random Forest is known for its ability to handle complex relationships, handle high-dimensional data, and mitigate overfitting.

Attribute Importances in Random Forest:
- Random Forest provides a measure of attribute importance based on the decrease in impurity (e.g., Gini impurity) that is achieved by splitting on a particular attribute.
- The attribute importance is calculated by averaging the impurity decrease across all the trees in the forest.
- The higher the impurity decrease, the more important the attribute is considered in determining the outcome.
- Random Forest attribute importances provide insights into which features have the most predictive power for the given task.
- Importances are often normalized to sum up to 1 or scaled to a specific range for easier interpretation and comparison.

It's worth noting that Random Forest can handle both linear and non-linear relationships between the attributes and the target variable. It is also robust to outliers and can handle missing values in the data. Additionally, Random Forest provides measures of attribute importance that can help identify the most influential features for making predictions.

Sure! Here's a brief overview of Gradient Boosting Regression (GBR) and its attribute importances:

Gradient Boosting Regression (GBR):
- GBR is a machine learning algorithm that builds an ensemble of weak prediction models, typically decision trees, in a sequential manner.
- It belongs to the family of boosting algorithms, where each subsequent model is trained to correct the mistakes made by the previous models.
- GBR works by iteratively fitting new models to the negative gradient of the loss function, thereby minimizing the overall loss.
- The final prediction is obtained by summing the predictions from all the individual models.

Attribute Importances in GBR:
- GBR provides a measure of attribute importance based on how much each attribute contributes to the reduction of the loss function during the training process.
- The importance of an attribute is calculated as the total reduction in the loss function attributed to splits on that attribute over all the trees in the ensemble.
- Attributes that consistently lead to greater reductions in the loss function are considered more important.
- GBR attribute importances can help identify the most influential features for making predictions and provide insights into the relationships between the attributes and the target variable.

It's important to note that GBR is a powerful algorithm capable of handling complex relationships and achieving high predictive accuracy. However, it can be prone to overfitting if not properly tuned or if the dataset is too small. Regularization techniques such as controlling the learning rate, tree depth, and the number of estimators can help mitigate overfitting and improve generalization.

Sure! Here's a brief overview of Multinomial Naive Bayes (MultinomialNB) and its attribute importances:

Multinomial Naive Bayes (MultinomialNB):
- MultinomialNB is a probabilistic classifier that is specifically designed for classification tasks with discrete features.
- It is commonly used for text classification tasks where the features represent word counts or frequency distributions.
- MultinomialNB assumes that the features follow a multinomial distribution, hence the name.
- It works based on Bayes' theorem and calculates the probability of a sample belonging to each class given its feature values.
- MultinomialNB assigns the class label with the highest probability as the predicted class.

Attribute Importances in MultinomialNB:
- MultinomialNB does not inherently provide attribute importances as it is based on probabilistic calculations rather than feature importance rankings.
- However, you can still gain insights into attribute contributions by examining the estimated probabilities and their corresponding feature values.
- The coefficients (log probabilities) learned by MultinomialNB can give an indication of the relative importance of different features.
- Positive coefficients indicate that an increase in the feature value is more likely to be associated with the corresponding class, while negative coefficients indicate the opposite.
- Higher absolute values of the coefficients generally suggest greater importance or influence of the corresponding feature.

It's important to note that MultinomialNB is suitable for classification tasks with discrete features, particularly in cases where the assumption of feature independence holds reasonably well. However, it may not be the best choice for datasets with continuous or highly correlated features.

Sure! Here's a brief overview of Decision Tree Classifier and its attribute importances:

Decision Tree Classifier:
- Decision Tree Classifier is a supervised machine learning algorithm used for classification tasks.
- It creates a flowchart-like structure called a decision tree, where each internal node represents a feature, each branch represents a decision based on that feature, and each leaf node represents a class label or a class distribution.
- The decision tree is constructed by recursively partitioning the data based on different features and their thresholds, aiming to maximize the separation of the classes.
- During prediction, the input features are traversed down the decision tree until a leaf node is reached, and the corresponding class label is assigned to the sample.

Attribute Importances in Decision Tree Classifier:
- Decision Tree Classifier provides a measure of attribute importances based on how much each attribute contributes to the purity of the classes within the nodes of the tree.
- The attribute importance is typically calculated as the total reduction in impurity (e.g., Gini impurity or entropy) achieved by splits on that attribute throughout the tree.
- Features that lead to significant reductions in impurity are considered more important in determining the class labels.
- Decision Tree attribute importances can help identify the most influential features for making predictions and provide insights into the relationships between the attributes and the target variable.

It's worth noting that Decision Tree Classifier can capture non-linear relationships and interactions between features. However, it is prone to overfitting, especially when the tree becomes deep and complex. Techniques like pruning, setting a maximum depth, or using ensemble methods like Random Forest can help mitigate overfitting and improve the generalization capability of the model.






from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Assuming you have already preprocessed your data and stored it in 'X' and 'y' variables

# Create and fit the LDA model
lda = LinearDiscriminantAnalysis()
lda.fit(X, y)

# Get the coefficients or loadings of the LDA components
loadings = lda.coef_

# Access the loadings for each component
for i, component in enumerate(loadings):
    original_columns = X.columns  # Replace 'X' with your original data
    component_loadings = dict(zip(original_columns, component))

    # Sort the loadings in descending order
    sorted_loadings = sorted(component_loadings.items(), key=lambda x: abs(x[1]), reverse=True)

    # Select the top 10 contributing columns
    top_10_loadings = sorted_loadings[:10]

    print(f"Top 10 Loadings for Component {i+1}:")
    for feature, loading in top_10_loadings:
        print(f"{feature}: {loading}")
    print()



from sklearn.decomposition import PCA

# Assuming you have already performed PCA on your data and stored it in 'pca' variable

# Get the loadings or coefficients of the PCA components
loadings = pca.components_

# Access the loadings for each component
for i, component in enumerate(loadings):
    original_columns = X.columns  # Replace 'X' with your original data
    component_loadings = dict(zip(original_columns, component))

    # Sort the loadings in descending order
    sorted_loadings = sorted(component_loadings.items(), key=lambda x: abs(x[1]), reverse=True)

    # Select the top 10 contributing columns
    top_10_loadings = sorted_loadings[:10]

    print(f"Top 10 Loadings for Component {i+1}:")
    for feature, loading in top_10_loadings:
        print(f"{feature}: {loading}")
    print()



from sklearn.decomposition import PCA

# Assuming you have already performed PCA on your data and stored it in 'pca' variable

# Get the loadings or coefficients of the PCA components
loadings = pca.components_

# Access the loadings for each component
for i, component in enumerate(loadings):
    original_columns = X.columns  # Replace 'X' with your original data
    component_loadings = dict(zip(original_columns, component))
    print(f"Loadings for Component {i+1}:")
    for feature, loading in component_loadings.items():
        print(f"{feature}: {loading}")
    print()




import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam, RMSprop
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
model.add(Dropout(0.2))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(y_train.shape[1], activation='softmax'))

# Compile the model
optimizer = Adam(learning_rate=0.001)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(X_test, y_test, verbose=1)
print("Test Loss:", loss)
print("Test Accuracy:", accuracy)




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
