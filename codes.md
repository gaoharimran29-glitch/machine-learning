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

## using scikit learn
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
## using pandas
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
