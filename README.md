# Handling Missing Data
```python
import pandas as pd
import numpy as np
df = pd.read_csv(r"dataset1.csv")
print(df.shape) #give the shape
print(df.isnull()) #returns true where null
print(df.isnull().sum()) #give total null values in each columns
print(df.isnull().sum().sum()) ##give total null values in whole dataset
print((df.isnull().sum()/df.shape[0])*100) #give perecntage of null values in each column
print((df.isnull().sum().sum()/df.shape[0])*100) #give perecntage of null values in whole dataset
print(df.notnull().sum()) #give total not null values of each columnns
print(df.notnull().sum().sum())  #give total not null values of whole dataset
df.drop(columns="PassengerId" , inplace=True) #to drop a column
df.dropna(inplace=True) #to drop all the null values rows
df.fillna(10) #to fill 10 in all null values
df["Gender"].fillna( df["Gender"].mode()[0] , inplace=True) #to fill mode in a column
df_cleaned = df[df["Age"] >= 0] #to remove negative values
df["Age] = df["Age"].apply(lambda x: np.nan if x<0 else x) #replace with NaN for negative values
print(df.select_dtypes(include='object')) #select only those columns where datatype is object
df.duplicated() #returns true where whole row is duplicate
df.drop_duplicates() #remove the duplicated row

#IF BELOW GRAPH IS BELL CURVE FILL IT WITH MEAN AND IF IT IS SKEWED FILL IT WIHT MEDIAN.
import pandas as pd
df = pd.read_csv(r'dataset1.csv')
import matplotlib.pyplot as plt
# AGE distribution plot
plt.hist(df["Age"].dropna(), bins=20, color='skyblue', edgecolor='black')
plt.title("Age Distribution")
plt.xlabel("Age")
plt.ylabel("Count")
plt.show()
# FARE distribution plot
plt.hist(df["Fare"].dropna(), bins=20, color='orange', edgecolor='black')
plt.title("Fare Distribution")
plt.xlabel("Fare")
plt.ylabel("Count")
plt.show()

#TO DETECT OUTLIERS
import seaborn as sns
import matplotlib.pyplot as plt
data = [10,20,30,40 , 50 , 60 ,70 ,15000 , 20000]
sns.boxplot(x=data)
plt.title("Detecting outliers")
plt.show()

#TO REMOVE OR CAP OUTLIERS WITH IQR METHOD (FOR SKEWED DATA)
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# detect boolean mask
outliers_mask = (df['value'] < lower_bound) | (df['value'] > upper_bound)
# cap (clip) values to bounds
df['value_iqr_capped'] = df['value'].clip(lower=lower_bound, upper=upper_bound)

#Z-Score (Bell curve)
import pandas as pd
from scipy.stats import zscore

# Dataframe with proper column name
data = [10, 12, 13, 15, 14, 100, 105]
df = pd.DataFrame(data, columns=['data'])  # give column name

# Calculate Z-score
df['z_score'] = zscore(df['data'])

# Threshold
threshold = 3

# Boolean mask for outliers
outliers_mask = df['z_score'].abs() > threshold

# Select only outliers
outliers = df[outliers_mask]
print("Outliers detected:")
print(outliers)

lower_bound = df['value'].mean() - (3*df['value].std())
upper_bound = df['value'].mean() + (3*df['value].std())
```

# One-Hot Encoding

## 1. using scikit learn
```python
import pandas as pd
from sklearn.preprocessing import OneHotEncoder

data = {
    "City":["Delhi" , "Mumbai" , "Pune" , "Chennai"]
}

df = pd.DataFrame(data)
ohe = OneHotEncoder(sparse_output=False)
encoded = ohe.fit_transform(df)
print(pd.DataFrame(encoded , columns=ohe.get_feature_names_out(["City"]) ))
```
**Output:**
```csv
 City_Chennai  City_Delhi  City_Mumbai  City_Pune
0           0.0         1.0          0.0        0.0
1           0.0         0.0          1.0        0.0
2           0.0         0.0          0.0        1.0
3           1.0         0.0          0.0        0.0
```
## 2. Using pandas
```python
import pandas as pd
data = {
    "City":["Delhi" , "Mumbai" , "Pune" , "Chennai"]
}
df = pd.DataFrame(data)
df_dummies = pd.get_dummies(df , columns=["City"])
df_dummies_binary = df_dummies.astype(int)
print(df_dummies_binary)
```
**Output:**
```csv
City_Chennai  City_Delhi  City_Mumbai  City_Pune
0             0           1            0          0
1             0           0            1          0
2             0           0            0          1
3             1           0            0          0
```

# Label Encoding
```python
from sklearn.preprocessing import LabelEncoder
data = {
    "City":["Delhi" , "Mumbai" , "Pune" , "Chennai"]
}
df = pd.DataFrame(data)
le = LabelEncoder()
df["City Encoded"] = le.fit_transform(df)
print(df)
```
**Output:**
```csv
City  City Encoded
0    Delhi             1
1   Mumbai             2
2     Pune             3
3  Chennai             0
```
# Ordinal Encoding
```python
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder
data = {
    "Size":["Small" , "Medium" , "Large" , "Large" , "Small" , "Medium"]
}
encoder = [["Small", "Medium" , "Large"]]
df = pd.DataFrame(data)
oe = OrdinalEncoder(categories=encoder)
df["Size Encoded"] = oe.fit_transform(df[["Size"]])
print(df)
```
**Ouptut:**
```csv
Size  Size Encoded
0   Small           0.0
1  Medium           1.0
2   Large           2.0
3   Large           2.0
4   Small           0.0
5  Medium           1.0
```
# Feature scaling

## Standardization
### Standard Scaler

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
data = {
    "Hours":[15 , 20 , 30 ,35 , 40] , 
    "Marks":[20000 , 25000 , 30000 , 35000 , 40000]
}

df = pd.DataFrame(data)
ss = StandardScaler()
ss_output = ss.fit_transform(df)
print(pd.DataFrame(ss_output, columns=["Hours" , "Marks"]))
```
**Output:**
```csv
 Hours     Marks
0 -1.401826 -1.414214
1 -0.862662 -0.707107
2  0.215666  0.000000
3  0.754829  0.707107
4  1.293993  1.414214
```

## Normalization
### Min Max scaler
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
data = {
    "Hours":[15 , 20 , 30 ,35 , 40] , 
    "Marks":[20000 , 25000 , 30000 , 35000 , 40000]
}

df = pd.DataFrame(data)
mms = MinMaxScaler()
mms_output = mms.fit_transform(df)
print(pd.DataFrame(mms_output, columns=["Hours" , "Marks"]))
```
**Output:**
```csv
 Hours  Marks
0    0.0   0.00
1    0.2   0.25
2    0.6   0.50
3    0.8   0.75
4    1.0   1.00
```
### Robust Scaler
```python
import pandas as pd
from sklearn.preprocessing import RobustScaler

data = [15 , 20 , 30 ,35 ,40]

df = pd.DataFrame(data)

rs = RobustScaler()
scaled_data = rs.fit_transform(df)
df_scaled = pd.DataFrame(scaled_data, columns=["Encoded"])
print(df_scaled)
```
**Output**
```csv
   Encoded
0      -1.000000
1      -0.666667
2       0.000000
3       0.333333
4       0.666667
```
# TRAIN_TEST_SPLIT
```python
from sklearn.model_selection import train_test_split
X = df[["Hours"]]
y = df[["Marks"]]

X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
print("Training data")
print(X_train)

print("Testing data")
print(X_test)
```
**Ouptut:**
```csv
Training data
Hours
4     40
2     30
0     15
3     35

Testing data
Hours
1     20
```
# Supervised Learning
## Regression
### Linear reression
### To check linearity in data

```python
#program to check linearity
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = {
    "House no." : [1,2,3,4,5,6,7] ,
    "No. of people lives" : [1,2,3,4,5,6,7]
}

df = pd.DataFrame(data)
sns.scatterplot(x='House no.' , y="No. of people lives" , data=df)
plt.show()
```

### Codes of Linear Regression

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = {
    "House no." : [1,2,3,4,5,6,7] ,
    "No. of people lives" : [1,2,3,4,5,6,7]
}

df = pd.DataFrame(data)
X = df[["House no."]]
y = df['No. of people lives']
X_train , X_test , y_train , y_test = train_test_split(X , y , test_size=0.2 , random_state=42)
model = LinearRegression()
model.fit(X_train , y_train)

user = int(input("Enter House no.: "))

y_pred = model.predict([[user]])

print(f"IN HOUSE NO. {user}, {y_pred[0]} no. of peoples live")
```

### Multiple Linear Regression

```python
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Step 1: Create sample data with multiple features
data = {
    "House no.": [1, 2, 3, 4, 5, 6, 7],
    "No. of rooms": [2, 3, 2, 4, 3, 5, 4],
    "Area (sqft)": [500, 700, 600, 800, 750, 900, 850],
    "No. of people lives": [2, 3, 3, 4, 4, 5, 5]  # Target
}

df = pd.DataFrame(data)

# Step 2: Define features (X) and target (y)
X = df[["House no.", "No. of rooms", "Area (sqft)"]]  # Multiple features
y = df["No. of people lives"]

# Step 3: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 5: Take user input for multiple features
house_no = int(input("Enter House no.: "))
num_rooms = int(input("Enter number of rooms: "))
area = float(input("Enter area in sqft: "))

# Step 6: Predict using user input
user_input = [[house_no, num_rooms, area]]
y_pred = model.predict(user_input)

print(f"Predicted number of people living: {y_pred[0]:.0f}")
```
# Polynomial Regression
```python
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Dataset
data = {
    "Rooms": [1, 2, 3, 4, 5, 6],
    "People": [3, 5, 10, 20, 25, 30]
}

df = pd.DataFrame(data)

X = df[['Rooms']]
y = df['People']

# Polynomial Features (degree=2)
pf = PolynomialFeatures(degree=2)
X_poly = pf.fit_transform(X)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

# Linear Regression on Polynomial Features
lr = LinearRegression()
lr.fit(X_train, y_train)

# Model Score
print("Model R^2 Score:", lr.score(X_test, y_test))

# --- User Input Prediction ---
while True:
    try:
        user_input = float(input("Enter a value to predict (or type 'exit' to quit): "))
    except ValueError:
        print("Exiting...")
        break
    
    # Transform the input to polynomial features
    user_input_poly = pf.transform([[user_input]])
    
    # Predict
    prediction = lr.predict(user_input_poly)
    print(f"Predicted Output: {prediction[0]}")
```

# Ridge , Lasso And ElasticNet Regression

```python
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Sample Dataset
X = np.array([[1], [2], [3], [4], [5], [6]])  # Single input
y = np.array([3, 5, 10, 20, 25, 30])

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 1. Linear Regression ---
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("Linear Regression:")
print("Coefficients:", lr.coef_)
print("Intercept:", lr.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_lr))
print()

# --- 2. Ridge Regression ---
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("Ridge Regression:")
print("Coefficients:", ridge.coef_)
print("Intercept:", ridge.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_ridge))
print()

# --- 3. Lasso Regression ---
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("Lasso Regression:")
print("Coefficients:", lasso.coef_)
print("Intercept:", lasso.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_lasso))
print()

# --- 4. ElasticNet Regression ---
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)
elastic.fit(X_train, y_train)
y_pred_elastic = elastic.predict(X_test)
print("ElasticNet Regression:")
print("Coefficients:", elastic.coef_)
print("Intercept:", elastic.intercept_)
print("MSE:", mean_squared_error(y_test, y_pred_elastic))
```
