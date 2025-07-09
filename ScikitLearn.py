# Introduction

# provides a wide range of tools for tasks like classification, regression, clustering, 
# dimensionality reduction, model evaluation, and data preprocessing
# It is built on NumPy, SciPy, and Matplotlib, making it efficient and easy to use.
# simplifies the entire machine learning workflow — from cleaning and splitting data, 
# to training models and tuning them for best performance.
# used for traditional algorithmic modeling, not deep learning

# Importing necessary material
from sklearn import datasets                                # contains various datasets for practice
from sklearn.model_selection import train_test_split        # splits data for training/testing

# getting started 
iris = datasets.load_iris()     # Load the iris dataset
print(iris.feature_names)       # sepal and petal dimensions
print(iris.target_names)        # species of iris

x = iris.data               # features like measurements of sepal and petal -> input
y = iris.target             # target variable (species) -> output

# data preprocessing
from sklearn.preprocessing import StandardScaler            # scales features to zero mean and unit variance.

X_train, X_test, y_train,y_test = train_test_split(x,y,test_size=0.2)
scaler = StandardScaler()                           # Initialize the scaler
X_train_scaled = scaler.fit_transform(X_train)      # Fit and transform the training data
X_test_scaled = scaler.transform(X_test)            # Transform the test data

# regression model
from sklearn.linear_model import LinearRegression           # fits a straight line to continuous data
from sklearn.metrics import mean_squared_error              # evaluates how close predictions are to actual values

model = LinearRegression()                                  # Initialize the model
model.fit(X_train, y_train)                                 # Fit the model to the training data
predictions = model.predict(X_test)                         # Make predictions on the test data
print("MSE:", mean_squared_error(y_test,predictions))       # Evaluate the model using Mean Squared Error, 
# how far off the predictions are?  lower is better

# classification model
from sklearn.metrics import accuracy_score                  # evaluates how many predictions are correct
from sklearn.linear_model import LogisticRegression         # used for binary classification tasks

model = LogisticRegression()            # Initialize the model
model.fit(X_train,y_train)              # Fit the model to the training data
predictions = model.predict(X_test)     # Make predictions on the test data
print("Accuracy:", accuracy_score(y_test,predictions))    # Evaluate the model using accuracy score, what ratio of predictions are correct?

# model evaluation
from sklearn.metrics import classification_report, confusion_matrix     # provides detailed metrics for classification tasks
print(confusion_matrix(y_test, predictions))                            # Confusion matrix shows true vs predicted classifications 
print(classification_report(y_test, predictions))                       # Detailed report of precision, recall, f1-score for each class   

# model selection and tuning
from sklearn.model_selection import cross_val_score, GridSearchCV       # used for model selection and hyperparameter tuning
from sklearn.neighbors import KNeighborsClassifier

scores = cross_val_score(model, X_train, y_train, cv=5)                 # Splits data into 5 parts, trains and validates on each
print("Cross-validation scores:", scores)                               # to reduce overfitting and give a better generalization estimate.

parameters = {'n_neighbors': [3,5,7]}                                   # dictionary of hyperparameter values to try
grid = GridSearchCV(KNeighborsClassifier(),parameters,cv=3)             # train KNeighborsClassifier() with those parameters 3 time each 
grid.fit(X_train,y_train)                                               # Trains and evaluates all model+parameter combinations
print("best parameters: ", grid.best_params_)

# pipelines (just the optimization)
from sklearn.pipeline import Pipeline           # allows combining preprocessing + model into 1 reusable object
                                                # no need to write seperate scaaling and model steps

pipe = Pipeline([
    ('scaler', StandardScaler()),               # scale input
    ('model', LogisticRegression())             # train model
])

pipe.fit(X_train,y_train)                       
print(pipe.score(X_test,y_test))                # gives accuracy


# testing the model
new_flower = [[6.5, 2, 1.5, 0.2]]       # test dataset
pred = pipe.predict(new_flower)         # returns array with predicted flower as first index
print("predicted specie is:", iris.target_names[pred[0]])       # fetches first index of pred array as predcited flower is on first index
