import os
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.cluster import KMeans
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# What is the impact of road types, intersections, and speed zones on accident risk?

accident = pd.read_csv("accidents_with_location.csv")
output_dir = 'output_images'

features = accident[['ACCIDENT_NO','ROAD_GEOMETRY', 'SPEED_ZONE_CAT', 'ROAD_TYPE_CAT']]

encoder = OneHotEncoder()
scaled_data = encoder.fit_transform(features[['ROAD_GEOMETRY', 'SPEED_ZONE_CAT', 'ROAD_TYPE_CAT']]).toarray()

sum_sq_err = []
labels = {}
k_range = range(1, 11)
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_data)
    sum_sq_err.append(kmeans.inertia_)
    labels[k] = kmeans.labels_

# Plot the elbow method
plt.plot(k_range, sum_sq_err, marker='o')
plt.title("The Elbow Method Showing Optimal K")
plt.xlabel("Number of Clusters (K)")
plt.xticks(k_range)
plt.ylabel("Sum of Squared Errors")
plt.grid(True)
plt.savefig(os.path.join(output_dir, "task4_elbow.png"))


#based on the elbow method, we can see that the optimal number of clusters is 5
optimal_k = 5


kmeans = KMeans(n_clusters=optimal_k, random_state=42)
features['Cluster'] = kmeans.fit_predict(scaled_data)
accident_with_cluster = pd.merge(
    accident[['ACCIDENT_NO', 'SEVERITY']],
    features[['ACCIDENT_NO', 'Cluster']],
    on='ACCIDENT_NO',
    how='left'
)

# Draw plot: count the number of accidents per cluster
plt.figure(figsize=(10,6))
sns.countplot(data=features, x='Cluster')
plt.title("Accident count per Cluster")
plt.xlabel("Cluster")
plt.ylabel("Accident Count")
plt.savefig(os.path.join(output_dir, "task4_cluster_count.png"))
plt.show()

#  Draw pie chart, severity distribution per cluster with percentage

severity_counts = accident_with_cluster.groupby(['Cluster', 'SEVERITY']).size().unstack(fill_value=0)
severity_levels = [1, 2, 3, 4]
colormap = {1: 'blue',  2: 'green', 3: 'orange', 4: 'red'}
severity_pct_dist = [0.5, 0.5, 0.5, 0.8]

fig, axes = plt.subplots(1, 5, figsize=(20, 6), constrained_layout=True)

for i, cluster_id in enumerate(severity_counts.index):
    ax = axes[i]
    cluster_data = severity_counts.loc[cluster_id]
    cluster_data = cluster_data.reindex(severity_levels, fill_value=0)
    
    ax.pie(
        cluster_data,
        labels=[str(s) for s in severity_levels],
        autopct='%1.1f%%',
        colors=[colormap[s] for s in severity_levels],
        startangle=90,
        pctdistance=0.65,
    )
    ax.set_title(f"Cluster {cluster_id}")

legend_color = [mpatches.Patch(color=color, label=f'Severity {s}') for s, color in colormap.items()]

plt.legend(handles=legend_color, title='Severity', loc='lower left', bbox_to_anchor=(0.5, -0.35), fontsize=10)
plt.suptitle("Severity Distribution per Cluster", fontsize=16)
plt.savefig(os.path.join(output_dir, "task4_cluster_severity_pie.png"))
plt.show()

# print out the data of the clusters, count for each feature and severity
road_type_map = {
    0: "rural_road_type",
    1: "intercity_road_type",
    2: "residential_road_type",
    3: "Other"
}

road_geometry_map = {
    1: "Cross intersection",
    2: "T-intersection",
    3: "Y-intersection",
    4: "multiple intersection",
    5: "not at intersection",
    6: "Dead end",
    7: "Road closure",
    8: "Private property",
    9: "Unknown",
}

Speed_zone_map = {
        1: "30-40",
        2: "50-60",
        3: "70-80",
        4: "90-110",
}


for cluster in sorted(features['Cluster'].unique()):
    cluster_data = features[features['Cluster'] == cluster]
    
    # Count the number of each speed zone category
    speed_counts = cluster_data['SPEED_ZONE_CAT'].map(Speed_zone_map).value_counts()
    # Count the number of each road geometry category
    road_type_counts = cluster_data['ROAD_TYPE_CAT'].map(road_type_map).value_counts()
    # Count the number of each road geometry category
    road_geometry_counts = cluster_data['ROAD_GEOMETRY'].map(road_geometry_map).value_counts()
    # count the number of severity
    severity_counts = accident_with_cluster.groupby(['Cluster', 'SEVERITY']).size().unstack(fill_value=0)
    severity_d = severity_counts.loc[cluster] 
    print(f"severity_d = {severity_d}")
    
    rows = []
    for speed_zone_range, count in speed_counts.items():
        rows.append({
            'Cluster': cluster,
            'Category': 'SPEED_ZONE',
            'data': speed_zone_range,
            'Count': count
        })
        

    for road_type, count in road_type_counts.items():
        rows.append({
            'Cluster': cluster,
            'Category': 'ROAD_TYPE',
            'data': road_type,
            'Count': count
        })
    for road_geometry, count in road_geometry_counts.items():
        rows.append({
            'Cluster': cluster,
            'Category': 'ROAD_GEOMETRY',
            'data': road_geometry,
            'Count': count
        })
    for severity, count in severity_d.items():
        rows.append({
            'Cluster': cluster,
            'Category': 'SEVERITY',
            'data': f'{int(severity)}',
            'Count': count
        })

    summary_df = pd.DataFrame(rows)
    summary_df_name = os.path.join(output_dir, f"task_4_cluster_{cluster}_data.csv")
    summary_df.to_csv(summary_df_name, index=False)
