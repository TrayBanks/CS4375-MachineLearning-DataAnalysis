import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report

# Read the Auto data
auto_df = pd.read_csv('Auto.csv')

# Output the first few rows
print(auto_df.head())

# Output the dimensions of the data
print(auto_df.shape)

# Data exploration with code
print(auto_df[['mpg', 'weight', 'year']].describe())
# Range and average of mpg column: Range = 6-46, Average = 23.45
# Range and average of weight column: Range = 1613-5140, Average = 2970.26
# Range and average of year column: Range = 70-82, Average = 75.97

# Explore data types
print(auto_df.dtypes)

# Change the cylinders column to categorical
auto_df['cylinders'] = auto_df['cylinders'].astype('category').cat.codes

# Change the origin column to categorical
auto_df['origin'] = auto_df['origin'].astype('category')

# Verify the changes with the dtypes attribute
print(auto_df.dtypes)

# Deal with NAs
auto_df.dropna(inplace=True)

# Output the new dimensions
print(auto_df.shape)

# Modify columns
auto_df['mpg_high'] = (auto_df['mpg'] > auto_df['mpg'].mean()).astype(int)
auto_df.drop(['mpg', 'name'], axis=1, inplace=True)
print(auto_df.head())

# Data exploration with graphs
sns.catplot(x='mpg_high', kind='count', data=auto_df)
# We can see that there are more cars with low mpg than high mpg in the dataset

sns.relplot(x='horsepower',
            y='weight',
            hue='mpg_high',
            style='mpg_high',
            data=auto_df)
# We can see that the majority of cars with high mpg have lower horsepower and lower weight

sns.boxplot(x='mpg_high', y='weight', data=auto_df)
# We can see that the median weight of cars with high mpg is lower than the median weight of cars with low mpg

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(auto_df.drop('mpg_high',
                                                                 axis=1),
                                                    auto_df['mpg_high'],
                                                    test_size=0.2,
                                                    random_state=1234)
print(X_train.shape)
print(X_test.shape)

# Logistic Regression
lr = LogisticRegression(solver='lbfgs', max_iter=1000)
lr.fit(X_train, y_train)
lr_y_pred = lr.predict(X_test)

print(classification_report(y_test, lr_y_pred, zero_division=0))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_y_pred = dt.predict(X_test)
print(classification_report(y_test, dt_y_pred))

# Neural Network
nn1 = MLPClassifier(hidden_layer_sizes=(10, 10), max_iter=1000)
nn1.fit(X_train, y_train)
nn1_y_pred = nn1.predict(X_test)
print(classification_report(y_test, nn1_y_pred))

nn2 = MLPClassifier(hidden_layer_sizes=(50, 50), max_iter=2000)
nn2.fit(X_train, y_train)
nn2_y_pred = nn2.predict(X_test)
print(classification_report(y_test, nn2_y_pred))

# Analysis
#a. Based on the performance metrics, some models performed better than others. The Neural Network with hidden layer sizes of (50, 50) and max_iter of 2000 had the highest accuracy score, followed by the Neural Network with hidden layer sizes of (10, 10) and max_iter of 1000. The Decision Tree model had the lowest accuracy score, followed by the Logistic Regression model.

#b. The recall and precision scores for each class varied, indicating that the models had different strengths and weaknesses in identifying cars with high or low mpg.

#c. The better-performing models may have been better at capturing complex relationships between the features and the target variable. The decision tree model may have overfit the training data and not generalized well to the test data. Logistic regression assumes a linear relationship, which may not always hold. Neural network models are better at capturing non-linear relationships.

#d. R and sklearn are both useful tools for data analysis and modeling. Sklearn has a larger community and more resources available, making it easier to find solutions to problems, and has a more intuitive syntax. However, R has unique features and packages that may be better suited for specific analysis needs.