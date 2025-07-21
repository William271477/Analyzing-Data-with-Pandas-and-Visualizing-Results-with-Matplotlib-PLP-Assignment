import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris

# Enable plotting inline if using Jupyter
# %matplotlib inline

# Load and explore the dataset
try:
    iris = load_iris()
    data = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    data['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    
    print("✅ Dataset loaded successfully.\n")
    print("📌 First five rows:\n", data.head())
    print("\n📌 Dataset info:")
    print(data.info())
    print("\n📌 Missing values:\n", data.isnull().sum())

    # No missing values in this dataset; otherwise handle them:
    # data = data.dropna() or data.fillna(method='ffill')

except FileNotFoundError:
    print("❌ Dataset not found.")
except Exception as e:
    print(f"❌ Error loading dataset: {e}")

# Basic Data Analysis
print("\n📊 Statistical Summary:\n", data.describe())

grouped = data.groupby('species').mean(numeric_only=True)
print("\n📌 Mean of features per species:\n", grouped)

# Pattern Observation Example:
print("\n🔍 Observation: Iris-virginica tends to have the largest average measurements.")

# Data Visualization
sns.set(style="whitegrid")  # Use seaborn style

# Line chart (just simulating trends per sample for sepal length)
plt.figure(figsize=(10, 5))
plt.plot(data.index, data['sepal length (cm)'], label='Sepal Length', color='blue')
plt.title('Sepal Length Trend Over Sample Index')
plt.xlabel('Sample Index')
plt.ylabel('Sepal Length (cm)')
plt.legend()
plt.tight_layout()
plt.savefig('line_chart_sepal_length.png')
plt.show()

# Bar chart: average petal length per species
plt.figure(figsize=(8, 6))
sns.barplot(x='species', y='petal length (cm)', data=data, estimator='mean', palette='Set2')
plt.title('Average Petal Length per Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig('bar_chart_petal_length.png')
plt.show()

# Histogram of sepal width
plt.figure(figsize=(8, 6))
plt.hist(data['sepal width (cm)'], bins=15, color='teal', edgecolor='black')
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('histogram_sepal_width.png')
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(8, 6))
sns.scatterplot(x='sepal length (cm)', y='petal length (cm)', hue='species', data=data)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.tight_layout()
plt.savefig('scatter_sepal_vs_petal.png')
plt.show()
