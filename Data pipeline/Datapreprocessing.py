# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# %%
df=pd.read_csv('fetal_health.csv')

# %%
df.drop_duplicates(inplace=True)

# %%
df.drop('histogram_tendency',axis=1,inplace=True)

# %%
df

# %%
df.shape

# %%
df.columns

# %%
X=df.drop('fetal_health',axis=1).values
y=df['fetal_health'].values

# %%
print('features shape' ,X.shape)
print('Target shape' ,y.shape)

# %%
y

# %%

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,   # 20% test, 80% train (change as you like)
    random_state=42, # for reproducibility
    stratify=y       # if classification and want balanced splits, else omit
)


# %%
from sklearn.preprocessing import StandardScaler
import pandas as pd

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)


# %%
X_train

# %%
from sklearn.preprocessing import PowerTransformer
pt=PowerTransformer(method='yeo-johnson')
X_train=pt.fit_transform(X_train)

# %%
X_train

# %%
X_test=pt.transform(X_test)

# %%
pt.lambdas_

# %%
X_train

# %%
X_test

# %%
print("Means:", scaler.mean_)
print("Standard deviations:", scaler.scale_)
print('train  shape' ,X_train.shape)
print('Train Target shape' ,y_train.shape)
print('Test  shape' ,X_test.shape)
print('Test target shape' ,y_test.shape)


# %%
import pandas as pd

# Your column names (features + target)
columns = ['baseline value', 'accelerations', 'fetal_movement',
           'uterine_contractions', 'light_decelerations', 'severe_decelerations',
           'prolongued_decelerations', 'abnormal_short_term_variability',
           'mean_value_of_short_term_variability',
           'percentage_of_time_with_abnormal_long_term_variability',
           'mean_value_of_long_term_variability', 'histogram_width',
           'histogram_min', 'histogram_max', 'histogram_number_of_peaks',
           'histogram_number_of_zeroes', 'histogram_mode', 'histogram_mean',
           'histogram_median', 'histogram_variance', 'fetal_health']

feature_names = columns[:-1]
target_name = columns[-1]

# Convert to DataFrames / Series
X_train_df = pd.DataFrame(X_train, columns=feature_names)
X_test_df = pd.DataFrame(X_test, columns=feature_names)
y_train_df = pd.Series(y_train, name=target_name)
y_test_df = pd.Series(y_test, name=target_name)



# %%
feature_names

# %%
target_name

# %%
X_train_df

# %%
X_test_df

# %%
plt.figure(figsize=(18,6))
X_test_df.boxplot()
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(18,6))
X_train_df.boxplot()
plt.xticks(rotation=90)
plt.show()

# %%
drop_columns = ['fetal_movement', 'severe_decelerations',
           'prolongued_decelerations', 'histogram_number_of_zeroes']

# %%
X_test_df.drop(columns=drop_columns,axis=1,inplace=True)

# %%
X_train_df.drop(columns=drop_columns,axis=1,inplace=True)

# %%
plt.figure(figsize=(18,6))
X_train_df.boxplot()
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(18,6))
X_test_df.boxplot()
plt.xticks(rotation=90)
plt.show()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Select only numerical columns
num_cols = X_train_df.select_dtypes(include=['int64', 'float64']).columns

# Grid settings
cols_per_row = 3
rows = math.ceil(len(num_cols) / cols_per_row)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
axes = axes.flatten()  # Flatten to make indexing easy

# Plot each numeric column
for i, col in enumerate(num_cols):
    sns.histplot(X_train_df[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Hide any empty subplot slots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# %%
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Select only numerical columns
num_cols = X_test_df.select_dtypes(include=['int64', 'float64']).columns

# Grid settings
cols_per_row = 3
rows = math.ceil(len(num_cols) / cols_per_row)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
axes = axes.flatten()  # Flatten to make indexing easy

# Plot each numeric column
for i, col in enumerate(num_cols):
    sns.histplot(X_test_df[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Hide any empty subplot slots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# %%


# %%
# Save to CSV
X_train_df.to_csv('X_train.csv', index=False)
X_test_df.to_csv('X_test.csv', index=False)
y_train_df.to_csv('y_train.csv', index=False)
y_test_df.to_csv('y_test.csv', index=False)



