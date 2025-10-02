## Artificial Intelligence (AI)
- **Definition**: The broad field of creating machines that can think, reason, and act like humans.
- AI focuses on building systems that can perform tasks that normally require human intelligence.

**Examples:**
- Siri or Alexa → Understand and respond to voice commands.
- Google Translate → Convert text from one language to another.
- Self-driving cars → Make driving decisions automatically.

**AI is the umbrella term**. Machine Learning (ML) and Deep Learning (DL) are subsets of AI.

---

## Machine Learning (ML)
- **Definition**: A subset of AI where systems learn patterns from **data** without being explicitly programmed.
- The model improves its performance over time as it is exposed to more data.

**Examples:**
- Spam email detection → Classifies emails as spam or not spam.
- Netflix/YouTube recommendations → Suggests movies/videos based on your past viewing.
- Predicting house prices → Based on size, location, and features.

---

## Deep Learning (DL)
- **Definition**: A specialized subset of ML that uses **Artificial Neural Networks** (ANNs), inspired by the human brain.
- Capable of handling **large and complex data** such as images, audio, video, and text.

**Examples:**
- Facebook photo tagging → Recognizing faces automatically.
- Tesla Autopilot → Processing real-time camera and sensor data.
- ChatGPT / Google Bard → Large language models based on Transformers.

**Difference**:
- ML → Needs feature engineering (manual data preparation).
- DL → Learns features automatically from raw data.

---

## Types of Machine Learning

### A. Supervised Learning
- **Definition**: Training the model with both **input data (X)** and the correct **output labels (Y)**.
- The model learns the mapping from input → output.

**Examples:**
- Predicting house price (input = size, location, output = price).
- Classifying email (input = text, output = spam / not spam).

**Algorithms**: Linear Regression, Logistic Regression, Decision Trees, Random Forest, SVM.

---

### B. Unsupervised Learning
- **Definition**: Training the model only with **input data** (no labels).
- The model tries to discover hidden patterns or groupings in the data.

**Examples:**
- Customer segmentation (grouping similar customers by shopping behavior).
- Market basket analysis (finding items often bought together).

**Algorithms**: K-Means Clustering, Hierarchical Clustering, PCA.

---

### C. Reinforcement Learning
- **Definition**: An **agent** interacts with an **environment**, takes actions, and learns by receiving **rewards** or **penalties**.
- The goal is to maximize total reward over time.

**Examples:**
- Google DeepMind’s AlphaGo → Learned to play Go better than humans.
- Self-driving car → Reward for staying in lane, penalty for crashing.
- Robotics → Reward when walking correctly, penalty when falling.

**Algorithms**: Q-Learning, Deep Q-Networks (DQN), Policy Gradient.
---
# Data Preprocessing
1. Handling Missing Data
2. Label Encoding
3. Feature Scaling

# Handling Missing Data
- Handle the missing data and fill or delete it using pandas functions like fillna() , dropna()

# Categorical Encoding in Machine Learning  

## Why Encoding is Needed?  
- Machine Learning models can only understand **numerical data**.  
- But real-world datasets often contain **categorical features** such as `City`, `Gender`, `Education`.  
- To use these features in ML algorithms, we must **encode** them into numbers.  

---

## Types of Categorical Variables  

1. **Ordinal Variables**  
   - Categories have a **natural order or ranking**.  
   - Example:  
     - Education: `Primary < Secondary < Graduate < Post-Graduate`  
     - Size: `Small < Medium < Large`  

2. **Nominal Variables**  
   - Categories **do not have any order**.  
   - Example:  
     - Cities: `Delhi, Mumbai, Bangalore`  
     - Colors: `Red, Blue, Green`  

---

## Encoding Methods  

### 1. **Label Encoding**  
Each unique category is assigned an integer value.  

**Example:**  

| City      | Encoded |
|-----------|---------|
| Bangalore | 0       |
| Delhi     | 1       |
| Mumbai    | 2       |

**Code:**  

```python
from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = {"City": ["Delhi", "Mumbai", "Bangalore", "Delhi"]}
df = pd.DataFrame(data)

le = LabelEncoder()
df["City_Encoded"] = le.fit_transform(df["City"])
print(df)
```
**Output**
```csv
        City  City_Encoded
0      Delhi             1
1     Mumbai             2
2  Bangalore             0
3      Delhi             1
```

### ✅ Advantages:

- Very simple and fast.
-Creates only one column.

### ❌ Disadvantages:
- Introduces a false sense of order (e.g., Mumbai > Delhi > Bangalore).
- Not suitable for nominal data.

# One-Hot Encoding in Machine Learning

## What is One-Hot Encoding?
One-Hot Encoding is a technique to convert **categorical variables** into a form that can be provided to **machine learning algorithms** to improve predictions.

Instead of assigning numbers directly (like Label Encoding), it creates **binary columns (0/1)** for each category.

---

## 🎯 Example Dataset

Let’s take a simple dataset:

```csv
City
Delhi
Mumbai
Kolkata
Delhi
Mumbai
```
**Code:**
```python
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

# Sample DataFrame
data = {"City": ["Delhi", "Mumbai", "Kolkata", "Delhi", "Mumbai"]}
df = pd.DataFrame(data)

# Create OneHotEncoder object
ohe = OneHotEncoder(sparse_output=False)

# Fit and transform the City column
ar = ohe.fit_transform(df[["City"]])

# Convert to DataFrame with column names
df_ohe = pd.DataFrame(ar, columns=ohe.get_feature_names_out(["City"]))

print("Original Data:")
print(df)
print("\nOne-Hot Encoded Data:")
print(df_ohe)
```
**Output:**
```csv
   City_Delhi  City_Kolkata  City_Mumbai
0         1.0           0.0          0.0
1         0.0           0.0          1.0
2         0.0           1.0          0.0
3         1.0           0.0          0.0
4         0.0           0.0          1.0
```

### ✅ Advantages of One-Hot Encoding

- Preserves information without giving false order/priority.
- Works best for Nominal data (categories without order, e.g., City, Color, Gender).
- Prevents algorithms from misunderstanding numerical relationships.

### ⚠️ Disadvantages

- Increases the number of features (dimensionality issue).
- Not efficient for high-cardinality categorical features (e.g., thousands of unique IDs).

### When to Use?
Use One-Hot Encoding when the categorical feature is Nominal (no order).
Example: City, Gender, Country, Department.

# Whole Summarized Code for boths
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Sample Data
data = {"City": ["Delhi", "Mumbai", "Kolkata", "Delhi", "Mumbai"]}
df = pd.DataFrame(data)
print("Original DataFrame:")
print(df)

# -------------------------------
# 1️⃣ Label Encoding
# -------------------------------
le = LabelEncoder()
df["City_LabelEncoded"] = le.fit_transform(df["City"])
print("\nAfter Label Encoding:")
print(df)

# -------------------------------
# 2️⃣ One-Hot Encoding (Scikit-learn)
# -------------------------------
ohe = OneHotEncoder(sparse_output=False)
ohe_array = ohe.fit_transform(df[["City"]])
df_ohe = pd.DataFrame(ohe_array, columns=ohe.get_feature_names_out(["City"]))
print("\nAfter One-Hot Encoding (sklearn):")
print(df_ohe)

# -------------------------------
# 3️⃣ One-Hot Encoding (pandas get_dummies)
# -------------------------------
df_dummies = pd.get_dummies(df[["City"]], columns=["City"])
print("\nAfter One-Hot Encoding (pandas get_dummies):")
print(df_dummies)

# If boolean appears, convert to int (ensure binary 0/1)
df_dummies_binary = df_dummies.astype(int)
print("\nPandas get_dummies (Forced Binary 0/1):")
print(df_dummies_binary)
```
