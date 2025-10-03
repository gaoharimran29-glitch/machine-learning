# Handling Missing Data
```python
import pandas as pd
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
df["Gender"].fillna( df["Gender"].mode , inplace=True) 
print(df.select_dtypes(include='object')) #select only those columns where datatype is object
df.duplicated() #returns true where whole row is duplicate
df.drop_duplicates() #remove the duplicated row
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
