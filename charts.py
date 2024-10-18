import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import os

# Load the data
covid_data = pd.read_csv("/content/Covid Dataset.csv")

# Create a static folder if it doesn't exist
if not os.path.exists("static"):
    os.makedirs("static")

# Plot heatmap and save
plt.figure(figsize=(25, 20))
sns.heatmap(covid_data.corr(), annot=True, cmap="PuRd")
plt.title("Correlation Heatmap")
plt.savefig("static/imgcorrelation_heatmap.png")  # Save the heatmap
plt.close()  # Close the figure

# Plot pie chart and save
covid_data["COVID-19"].value_counts().plot.pie(explode=[0.1, 0.1], autopct='%1.1f%%', shadow=True)
plt.title('Covid Positive')
plt.savefig("static/img/covid_positive_pie_chart.png")  # Save the pie chart
plt.close()  # Close the figure

# Set up the figure and axes
fig, axes = plt.subplots(3, 3, figsize=(15, 10))
axes = axes.flatten()  # Flatten the 2D array of axes to 1D

# List of features to plot
features = [
    'COVID-19', 
    'Breathing Problem', 
    'Fever', 
    'Dry Cough', 
    'Sore throat', 
    'Abroad travel', 
    'Contact with COVID Patient', 
    'Attended Large Gathering'
]

# List of hues (if applicable)
hues = [None, None, 'COVID-19', 'COVID-19', 'COVID-19', 'COVID-19', 'COVID-19', 'COVID-19']

# List of palettes
palettes = [None, None, "Set2", "PuRd", "Set1", None, "PuRd", "Set1"]

# Create the plots and save
for ax, feature, hue, palette in zip(axes, features, hues, palettes):
    sns.countplot(x=feature, hue=hue, data=covid_data, palette=palette, ax=ax)
    for p in ax.patches:
        ax.annotate(f'\n{p.get_height()}', 
                    (p.get_x() + 0.2, p.get_height() + 100), 
                    ha='center', va='top', 
                    color='white', size=10)
    ax.set_title(feature)
    plt.savefig(f"static/img/{feature}_count_plot.png")  # Save each count plot
    ax.clear()  # Clear the axes for the next plot

# Adjust layout and show
plt.tight_layout()
plt.close()  # Close the figure
