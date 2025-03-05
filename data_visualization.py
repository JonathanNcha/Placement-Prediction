import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load the preprocessed data
df = pd.read_csv("data/processed_data.csv")


# Function to create visualizations
def create_visualizations(df):
    # Placement status count plot
    plt.figure(figsize=(8, 6))
    sns.countplot(x="PlacementStatus", data=df)
    plt.title("Placement Status Count")
    plt.savefig("visualizations/placement_status_count.png")
    plt.close()

    # Distribution of CGPA
    plt.figure(figsize=(8, 6))
    sns.histplot(df["CGPA"], kde=True, bins=30)
    plt.title("Distribution of CGPA")
    plt.savefig("visualizations/cgpa_distribution.png")
    plt.close()

    # Correlation heatmap
    plt.figure(figsize=(12, 10))
    corr = df.corr()
    sns.heatmap(corr, annot=False, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.savefig("visualizations/correlation_heatmap.png")
    plt.close()

    # Placement Status vs Internships
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="PlacementStatus", y="Internships", data=df)
    plt.title("Placement Status vs Internships")
    plt.savefig("visualizations/placement_vs_internships.png")
    plt.close()

    # Placement Status vs CGPA
    plt.figure(figsize=(8, 6))
    sns.boxplot(x="PlacementStatus", y="CGPA", data=df)
    plt.title("Placement Status vs CGPA")
    plt.savefig("visualizations/placement_vs_cgpa.png")
    plt.close()


# Create visualizations directory if it doesn't exist
if not os.path.exists("visualizations"):
    os.makedirs("visualizations")

# Generate visualizations
create_visualizations(df)
