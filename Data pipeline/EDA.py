# %%
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

# %%
df=pd.read_csv('fetal_health.csv')

# %%
df.head()

# %%
df.info()

# %%
df.shape

# %%
df.corr()['fetal_health']

# %%
df.isna().sum()

# %%
df.duplicated().sum()

# %%
import seaborn as sns
import matplotlib.pyplot as plt
import math

# Select only numerical columns
num_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Grid settings
cols_per_row = 3
rows = math.ceil(len(num_cols) / cols_per_row)

# Create subplots
fig, axes = plt.subplots(nrows=rows, ncols=cols_per_row, figsize=(5 * cols_per_row, 4 * rows))
axes = axes.flatten()  # Flatten to make indexing easy

# Plot each numeric column
for i, col in enumerate(num_cols):
    sns.histplot(df[col], kde=True, bins=30, ax=axes[i])
    axes[i].set_title(f"Distribution of {col}")
    axes[i].set_xlabel(col)
    axes[i].set_ylabel("Frequency")

# Hide any empty subplot slots
for j in range(len(num_cols), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# %%
plt.figure(figsize=(18,6))
df.boxplot()
plt.xticks(rotation=90)
plt.show()

# %%
plt.figure(figsize=(14,4))
sns.countplot(y=df['fetal_health'],palette='rocket')


