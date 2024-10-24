import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from math import pi

# Load data
df = pd.read_csv('Covid_Dataset.csv')

# Convert 'Yes'/'No' to 1/0 and ensure COVID-19 column is numeric
df_numeric = df.replace({'Yes': 1, 'No': 0})
df_numeric['COVID-19'] = df_numeric['COVID-19'].astype(int)

# Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df_numeric.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title('Heatmap of Symptoms and COVID-19 Status')
plt.savefig('static/images/heatmap.png')  # Save plot
plt.close()

# Pie chart
plt.figure(figsize=(8, 6))
df['COVID-19'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90, colors=['#ff9999','#66b3ff'])
plt.title('Distribution of COVID-19 Cases')
plt.ylabel('')
plt.savefig('static/images/pie_chart.png')  # Save plot
plt.close()

# Bar chart of symptoms count in positive cases
positive_cases = df[df['COVID-19'] == 'Yes']
symptom_columns = df.columns.drop('COVID-19')
positive_cases[symptom_columns] = positive_cases[symptom_columns].replace({'Yes': 1, 'No': 0})
symptoms_count = positive_cases[symptom_columns].sum()

plt.figure(figsize=(10, 6))
symptoms_count.plot(kind='bar', color='orange')
plt.title('Count of Symptoms in COVID-19 Positive Cases')
plt.ylabel('Count')
plt.xlabel('Symptoms')
plt.xticks(rotation=45)
plt.savefig('static/images/symptom_count_bar.png')  # Save plot
plt.close()

# Count plot of symptoms in positive cases
plt.figure(figsize=(10, 6))
sns.countplot(data=positive_cases.melt(value_vars=positive_cases.columns[:-1]), x='variable', hue='value')
plt.title('Symptoms Count in COVID-19 Positive Cases')
plt.xlabel('Symptoms')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.legend(title='Presence', loc='upper right')
plt.savefig('static/images/symptom_count_countplot.png')  # Save plot
plt.close()

# Radar chart
df_numeric = df.replace({'Yes': 1, 'No': 0})

# Calculate the mean of each symptom for COVID-19 positive and negative cases
positive_means = df_numeric[df_numeric['COVID-19'] == 1].mean()[:-1]  # Exclude 'COVID-19' from the means
negative_means = df_numeric[df_numeric['COVID-19'] == 0].mean()[:-1]

# Number of variables (symptoms)
categories = list(positive_means.index)
N = len(categories)

# Angles for radar chart
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Radar chart for positive cases
plt.figure(figsize=(8, 8))
ax = plt.subplot(111, polar=True)

# Positive and negative values
positive_values = positive_means.tolist()
positive_values += positive_values[:1]

negative_values = negative_means.tolist()
negative_values += negative_values[:1]

# Plot radar chart
ax.plot(angles, positive_values, linewidth=2, linestyle='solid', label='COVID-19 Positive')
ax.fill(angles, positive_values, 'b', alpha=0.1)

ax.plot(angles, negative_values, linewidth=2, linestyle='solid', label='COVID-19 Negative')
ax.fill(angles, negative_values, 'r', alpha=0.1)

# Labels and title
plt.xticks(angles[:-1], categories)
plt.title('Symptom Comparison: COVID-19 Positive vs Negative Cases', size=15, color='darkblue', y=1.1)
plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
plt.savefig('static/images/radar_chart.png')  # Save plot
plt.close()
