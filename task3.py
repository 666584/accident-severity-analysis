import pandas as pd
from sklearn import tree
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
import graphviz
from sklearn.linear_model import LogisticRegression
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import os

# Part 2 Creat folder to save images
output_dir = 'output_images'
if not os.path.exists(output_dir):
    os.makedirs(output_dir, exist_ok=True)

# Read data from CSV file
accidents = pd.read_csv('accidents_with_location.csv')

# Extract columens LogisticRegression model need
lr = accidents[['ROAD_GEOMETRY', 'SEVERITY', 'ROAD_TYPE_CAT', 'ROAD_TYPE_INT_CAT', 'SPEED_ZONE_CAT']].copy()
# Remove unknown or not normal road type
lr = lr[(lr['ROAD_TYPE_CAT'] != 3) & (lr['ROAD_TYPE_INT_CAT'] != 3)]

# Splite features and label columens
lr_y = lr['SEVERITY']
lr_X = lr.drop(columns=['SEVERITY'])
# Encode these categorical features
lr_X = pd.get_dummies(lr_X, columns=['ROAD_GEOMETRY', 'ROAD_TYPE_CAT', 'ROAD_TYPE_INT_CAT'])
# Splite train and test dataset
lr_X_train, lr_X_test, lr_y_train, lr_y_test = train_test_split(
    lr_X, lr_y,
    stratify=lr_y,
    test_size=0.2,
    random_state=42
)

# Get labels
classes = lr_y_train.unique()
# Calculate weights
weights = compute_class_weight('balanced', classes=classes, y=lr_y_train)
class_weights = dict(zip(classes, weights))
#  Logistic Regression model with paremeters
lr_model = LogisticRegression(max_iter=1000,class_weight=class_weights,solver='lbfgs')
# Training
lr_model.fit(lr_X_train, lr_y_train)

# Prediction
lr_y_pred = lr_model.predict(lr_X_test)

# Evaluation and save result
lr_report = classification_report(lr_y_test, lr_y_pred, zero_division=0)
with open('output_images\LR_report.txt', 'w') as f:
    f.write(lr_report)


# feature names after one-hot
feature_names = lr_X_train.columns
# coef matrix
coef_matrix = lr_model.coef_
coef_df = pd.DataFrame(coef_matrix, columns=feature_names, index=classes)

# Draw heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(coef_df, annot=True, cmap="coolwarm", center=0, fmt=".2f", linewidths=0.5)
plt.title("Logistic Regression Coefficient Matrix")
plt.xlabel("Features")
plt.ylabel("Classes")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("output_images\LR_coef_matrix_heatmap.jpg")


# Extract columens Decision Tree model need
dt = accidents[['ROAD_GEOMETRY', 'SEVERITY', 'ROAD_TYPE_CAT', 'ROAD_TYPE_INT_CAT', 'SPEED_ZONE_CAT']].copy()

# Splite features and label columens
dt_y = dt['SEVERITY']
dt_X = dt.drop(columns=['SEVERITY'])
# encode these categorical features
dt_X = pd.get_dummies(dt_X, columns=['ROAD_GEOMETRY', 'ROAD_TYPE_CAT', 'ROAD_TYPE_INT_CAT'])
# Splite train and test dataset
dt_X_train, dt_X_test, dt_y_train, dt_y_test = train_test_split(
    dt_X, dt_y,
    stratify=dt_y,
    test_size=0.2,
    random_state=42
)

# Get labels
dt_classes = dt_y_train.unique()
# Calculate weights
dt_weights = compute_class_weight('balanced', classes=dt_classes, y=dt_y_train)
dt_class_weights = dict(zip(dt_classes, dt_weights))
# Custom ID3 decision tree with paremeters
dt_model = tree.DecisionTreeClassifier(max_depth=3, criterion='entropy', class_weight=dt_class_weights, random_state=42)

# Train model
dt_model.fit(dt_X_train, dt_y_train)

# Prediction
dt_y_pred = dt_model.predict(dt_X_test)

# Evaluation and save result
dt_report = classification_report(dt_y_test, dt_y_pred, zero_division=0)
with open('output_images\DT_report.txt', 'w') as f:
    f.write(dt_report)


# Draw the decision tree
class_names = [str(c) for c in dt_model.classes_]
dot_data = tree.export_graphviz(
    dt_model, 
    out_file=None, 
    feature_names=dt_X.columns.tolist(),
    class_names=class_names,
    filled=True,
    rounded=True,  
    special_characters=True
    )
# Render and output
graph = graphviz.Source(dot_data)
graph.render(filename="output_images\decision_tree", format="jpg", cleanup=True) 