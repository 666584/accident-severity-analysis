import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set the plot style
sns.set(style="whitegrid")

# Part 1 Set path and read data
# Use user_uploaded file path
file_path = 'accidents_with_location.csv'

# Read CSV file
accidents_with_location = pd.read_csv(file_path)

# Part 2 Creat folder to save images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Part 3 Plotting Section
# 1. Relationship between speed zone category and accident severity
plt.figure(figsize=(8,5))
sns.boxplot(x='SPEED_ZONE_CAT', y='SEVERITY', data=accidents_with_location, palette='Set3')
plt.title('Speed Zone Category vs. Accident Severity')
plt.xlabel('Speed Zone Category')
plt.ylabel('Accident Severity')
plt.savefig(os.path.join(output_dir, 'speed_zone_vs_severity.png'))
plt.close()

# 2. Distribution of accident severity by road type category
plt.figure(figsize=(8, 5))
sns.countplot(x='ROAD_TYPE_CAT', hue='SEVERITY', data=accidents_with_location, palette='Set3')
plt.title('Road Type Category vs. Accident Severity')
plt.xlabel('Road Type Category')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity')
plt.savefig(os.path.join(output_dir, 'road_type_vs_severity.png'))
plt.close()

# 3. Heatmap of road geometry vs accident severity
plt.figure(figsize=(12, 8))
heatmap_data = pd.crosstab(accidents_with_location['ROAD_GEOMETRY_DESC'], accidents_with_location['SEVERITY'])
sns.heatmap(heatmap_data, annot=True, fmt='d', cmap='YlGnBu')
plt.title('Road Geometry vs. Accident Severity')
plt.xlabel('Accident Severity')
plt.ylabel('Road Geometry Description')
plt.savefig(os.path.join(output_dir, 'road_geometry_vs_severity.png'))
plt.close()

# 4. Heatmap of correlation matrix of features
plt.figure(figsize=(16, 10))
corr_matrix = accidents_with_location.corr(numeric_only=True)
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'))
plt.close()

# 5. Relationship between intersection road type and accident severity
plt.figure(figsize=(8, 5))
sns.countplot(x='ROAD_TYPE_INT_CAT', hue='SEVERITY', data=accidents_with_location, palette='Set3')
plt.title('Intersection Road Type vs. Accident Severity')
plt.xlabel('Intersection Road Type Category')
plt.ylabel('Number of Accidents')
plt.legend(title='Severity')
plt.savefig(os.path.join(output_dir,'intersection_road_type_vs_severity.png'))
plt.close()

# Part 4 Complete the data output
output_images = os.listdir(output_dir)
output_images
