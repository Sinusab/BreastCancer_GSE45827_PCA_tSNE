# 🧬 Logistic Regression on CuMiDa Breast Cancer Gene Expression Dataset
### PCA-Based Dimensionality Reduction and Feature Interpretation

# Author: Sina Abyar
# Date: July 2025
# Dataset: GSE45827 (from CuMiDa)
# Tags: #logistic-regression #gene-expression #pca #classification #bioinformatics #breast-cancer #sklearn #visualization

# --- 1. Load Data & Preprocessing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, ConfusionMatrixDisplay, classification_report
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv('Breast_GSE45827.csv')
X = df.iloc[:, 2:]  # Gene expressions
y = df.iloc[:, 1]   # Labels

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2, random_state=42, shuffle=True, stratify=y)

# --- 2. Train Logistic Regression (on raw data)
model = LogisticRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- 3. Evaluation Metrics
print("\n=== Raw Logistic Regression Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, digits=3))

# --- 4. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_).plot()

# --- 5. Top Contributing Genes per Class (Raw Coefficients)
coef = model.coef_  # shape: (n_classes, n_features)
feature_names = np.array(X.columns)
class_names = model.classes_

print("\n=== Top Genes by Class ===")
for i, class_name in enumerate(class_names):
    top_idx = np.argsort(np.abs(coef[i]))[::-1][:10]
    print(f"\nTop 10 genes for class {class_name}:")
    for j in top_idx:
        print(f"{feature_names[j]:20s} coef = {coef[i, j]:.4f}")

# --- 6. Heatmap of Top Genes per Class
import seaborn as sns
import matplotlib.pyplot as plt

# Build dictionary of top genes per class
gene_coef_dict = {}
for i, class_name in enumerate(class_names):
    class_coef = coef[i]
    top_indices = np.argsort(np.abs(class_coef))[::-1][:10]
    for j in top_indices:
        gene = feature_names[j]
        if gene not in gene_coef_dict:
            gene_coef_dict[gene] = {}
        gene_coef_dict[gene][class_name] = class_coef[j]

# Build dataframe
df_coef = pd.DataFrame.from_dict(gene_coef_dict, orient='index')
df_coef = df_coef.fillna(0)[class_names]  # fill missing with 0 and order columns

# Plot heatmap
plt.figure(figsize=(10, max(5, df_coef.shape[0] * 0.4)))
sns.heatmap(df_coef, cmap='vlag', center=0, annot=True, fmt=".4f", linewidths=0.5)
plt.title("Top Gene Coefficients per Class")
plt.xlabel("Cancer Subtype")
plt.ylabel("Gene ID")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 7. PCA + Logistic Regression
from sklearn.decomposition import PCA

pca = PCA(n_components=0.6)  # Retain 60% variance
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)

pca_model = LogisticRegression()
pca_model.fit(X_train_pca, y_train)
y_pred_pca = pca_model.predict(X_test_pca)

print("\n=== Logistic Regression after PCA ===")
print("Accuracy:", accuracy_score(y_test, y_pred_pca))
print(classification_report(y_test, y_pred_pca, digits=3))
print("Explained variance (PCA):", pca.explained_variance_ratio_.sum())

# --- 8. Top Genes Based on PCA Loadings
loadings = pca.components_.T  # shape: (n_features, n_components)
top_genes_idx = np.argsort(np.sum(np.abs(loadings[:, :18]), axis=1))[::-1][:100]
top_genes = [feature_names[i] for i in top_genes_idx]
print("\nTop 100 PCA-contributing genes:")
print(top_genes)
logreg_pca_breastcancer
