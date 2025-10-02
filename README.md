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
4. Split Data

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

### Advantages:

- Very simple and fast.
-Creates only one column.

### Disadvantages:
- Introduces a false sense of order (e.g., Mumbai > Delhi > Bangalore).
- Not suitable for nominal data.

# One-Hot Encoding in Machine Learning

## What is One-Hot Encoding?
One-Hot Encoding is a technique to convert **categorical variables** into a form that can be provided to **machine learning algorithms** to improve predictions.

Instead of assigning numbers directly (like Label Encoding), it creates **binary columns (0/1)** for each category.

---

## Example Dataset

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

### Advantages of One-Hot Encoding

- Preserves information without giving false order/priority.
- Works best for Nominal data (categories without order, e.g., City, Color, Gender).
- Prevents algorithms from misunderstanding numerical relationships.

### Disadvantages

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

# Feature Scaling in Machine Learning

## Introduction
In Machine Learning, different features can have **different scales**.  
For example:
- Age might be in **years** (18, 25, 40)
- Salary might be in **thousands** (20,000 – 100,000)

If we directly feed such data to ML models, the feature with **larger range** (like salary) will dominate over smaller scale features (like age).  

** Feature Scaling helps to bring all features to the same scale.**

---

## Types of Feature Scaling

### 1. **Standardization (Z-score Normalization)**
- Formula:  
  \[
  z = \frac{x - \mu}{\sigma}
  \]  
  where \( \mu \) = mean, \( \sigma \) = standard deviation  

- Transforms data to have:
  - Mean = 0
  - Standard Deviation = 1

- Best when data follows **Gaussian distribution** (normal distribution).  
- Commonly used in **Logistic Regression, SVM, PCA, Neural Networks**.

---

### 2. **Normalization (Min-Max Scaling)**
- Formula:  
  \[
  x' = \frac{x - x_{min}}{x_{max} - x_{min}}
  \]

- Transforms data to a **fixed range**, usually **[0,1]**.
- Best when you want to **preserve relationships** but scale data.  
- Common in **Neural Networks** where inputs are preferred in 0–1.

---

## Python Implementation

```python
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Sample Data
data = {
    "Age": [18, 25, 40, 60, 80],
    "Salary": [20000, 50000, 100000, 30000, 80000]
}
df = pd.DataFrame(data)
print("Original Data:")
print(df)

# -------------------------------
# 1️⃣ Standardization
# -------------------------------
scaler = StandardScaler()
standardized = scaler.fit_transform(df)
df_standardized = pd.DataFrame(standardized, columns=df.columns)
print("\nAfter Standardization (Z-score):")
print(df_standardized)

# -------------------------------
# 2️⃣ Normalization (Min-Max)
# -------------------------------
minmax = MinMaxScaler()
normalized = minmax.fit_transform(df)
df_normalized = pd.DataFrame(normalized, columns=df.columns)
print("\nAfter Normalization (Min-Max):")
print(df_normalized)
```
# When to Use Standardization vs Normalization

Feature Scaling is critical in Machine Learning, but choosing the right method depends on the **algorithm** and **data distribution**.

---

## 1. Standardization (Z-score Scaling)
- **Formula:**
  \[
  z = \frac{x - \mu}{\sigma}
  \]

- **When to Use?**
  - Data is **normally distributed** (bell curve).
  - Algorithms assume data is **centered around 0**.
  - Less sensitive to **outliers** compared to Min-Max.

- **Example:**
  Predicting **Diabetes using Logistic Regression**:
  - Feature 1: Age (18–80)  
  - Feature 2: Glucose Level (50–250)  

  Logistic Regression assumes mean-centered features, so **Standardization** is preferred.

---

## 2. Normalization (Min-Max Scaling)
- **Formula:**
  \[
  x' = \frac{x - x_{min}}{x_{max} - x_{min}}
  \]

- **When to Use?**
  - Data doesn’t follow normal distribution.
  - Need features in a fixed **[0,1] range**.
  - Algorithms sensitive to **absolute distances**.

- **Example:**
  Movie recommendation using **KNN (K-Nearest Neighbors)**:
  - Feature 1: Age (18–80)  
  - Feature 2: Rating (1–5)  

  Without scaling, **Age dominates** due to larger range.  
  ✔️ Normalization ensures equal contribution.

---

## Side-by-Side Comparison

| Algorithm | Preferred Scaling | Why |
|-----------|------------------|-----|
| **Logistic Regression** | Standardization | Works better with mean-centered data |
| **SVM (Support Vector Machine)** | Standardization | Kernel functions depend on variance |
| **KNN (K-Nearest Neighbors)** | Normalization | Distance calculation (Euclidean) is scale-sensitive |
| **Neural Networks (Deep Learning)** | Normalization | Inputs in [0,1] or [-1,1] speed up training |
| **PCA (Principal Component Analysis)** | Standardization | Depends on covariance and variance |
| **Gradient Descent (Linear Regression, etc.)** | Both | Either improves convergence |

--- Rule of Thumb

1. If algorithm depends on distances (KNN, K-Means, Neural Nets) → Use Normalization.

2. If algorithm depends on distributions/variance (SVM, Logistic, PCA) → Use Standardization.

# Train-Test Split in Machine Learning

## Introduction
Before training a machine learning model, it’s crucial to evaluate how well it performs on **unseen data**.  
We do this by splitting the dataset into:

- **Training set** → Used to **train the model**  
- **Testing set** → Used to **evaluate the model** on unseen data

> This prevents **data leakage** and ensures realistic model performance.

---

## Why Split Data?
1. To avoid overfitting: Model learns training data too well but fails on new data.  
2. To measure generalization: Gives a realistic estimate of how your model will perform in production.  
3. To safely apply preprocessing: Scaling, encoding, or transformations must only be fitted on training data.

---

## Typical Split Ratios
| Purpose | Common Ratio |
|---------|-------------|
| Training | 70–80% |
| Testing | 20–30% |

> For classification tasks, often use **stratified splitting** to preserve class distribution.

---

## Python Example

```python
import pandas as pd
from sklearn.model_selection import train_test_split

# Sample Data
data = {
    "Age": [18, 25, 40, 60, 80],
    "Salary": [20000, 50000, 100000, 30000, 80000],
    "Purchased": [0, 1, 0, 0, 1]  # Target variable
}
df = pd.DataFrame(data)

# Features and Target
X = df[["Age", "Salary"]]   # Independent variables
y = df["Purchased"]         # Target variable

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("X_train:\n", X_train)
print("\nX_test:\n", X_test)
print("\ny_train:", y_train.tolist())
print("\ny_test:", y_test.tolist())
```
### Sample Output
```csv
X_train:
   Age  Salary
4   80   80000
2   40  100000
0   18   20000
3   60   30000

X_test:
   Age  Salary
1   25   50000

y_train: [1, 0, 0, 0]
y_test: [1]
```

### Key Parameters

- test_size: Proportion of dataset to use as test set (e.g., 0.2 → 20%)
- random_state: Ensures reproducibility of the split
- stratify: Ensures class distribution is maintained in train/test (important for classification)
---

### Best Practices

1. Split first, then preprocess: Fit scalers/encoders only on X_train

2. Use stratification for classification tasks with imbalanced classes

3. Set random_state for reproducibility

4. Do not touch X_test before final evaluation

---

Next Step

After splitting, the workflow typically continues as:

1. Fit encoders / scalers on X_train
2. Transform X_train and X_test using the fitted preprocessors
3. Train the model on X_train
4. Evaluate performance on X_test

Note:- Combining Train-Test Split + Scaling + Encoding is the foundation of a safe and robust ML pipeline.