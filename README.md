# 🧬 GSE45827 Breast Cancer Gene Expression Analysis

This project explores gene expression profiles of breast cancer subtypes using PCA and t-SNE dimensionality reduction techniques. The dataset used is **GSE45827**, obtained from the [CuMiDa](https://sbcb.inf.ufrgs.br/cumida) database and originally published on [NCBI GEO](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827).

## 📌 Project Goals

- Load and preprocess high-dimensional gene expression data  
- Perform exploratory data analysis (EDA)  
- Apply PCA and t-SNE for dimensionality reduction  
- Visualize subtype separability in 2D space  
- Extract biological insights to guide future classification modeling


## 📥 Dataset Information

- **Accession:** GSE45827  
- **Source:** CuMiDa / NCBI GEO  
- **Samples:** 151  
- **Features:** 54,676 genes  
- **Classes:** luminal_A, luminal_B, HER, basal, normal, cell_line  

> ⚠️ **Note:** Due to GitHub’s 100MB file size limit, the dataset file is not included in this repository.  
> Please download it manually from one of the following sources and place it in the same directory as the notebook or wherever you prefer:

### 🔗 Download Links

- [CuMiDa Repository](https://sbcb.inf.ufrgs.br/cumida)  
- [NCBI GEO Accession: GSE45827](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE45827)

**Expected filename:**

```
Breast_GSE45827.csv
```

## 📦 Requirements

Install required Python packages using:

```bash
pip install -r requirements.txt
```

**Minimal `requirements.txt`:**

```
pandas
numpy
scikit-learn
matplotlib
seaborn
```

## 🚀 How to Run

Clone the repository:

```bash
git clone https://github.com/your-username/BreastCancer_GSE45827_PCA_tSNE.git
cd BreastCancer_GSE45827_PCA_tSNE
```

Download the dataset and place it next to the notebook as:

```
notebooks/Breast_GSE45827.csv
```

Launch the notebook:

```bash
jupyter notebook notebooks/GSE45827_PCA_tSNE_Analysis.ipynb
```

## 📈 Summary of Results

- **t-SNE on raw X** achieved the best class separation.
- **PCA followed by t-SNE** reduced dimensionality and noise, improving speed but slightly lowering clarity.
- Gene expression shows high discriminatory power across subtypes, supporting its use for downstream classification.

## 🔮 Next Steps

- Train supervised classifiers (e.g., RandomForest, SVM) on PCA-reduced features  
- Compare model performance on raw vs reduced features  
- Extend analysis to other CuMiDa or TCGA datasets

## 🏷️ Tags

`#bioinformatics` `#breast-cancer` `#gene-expression` `#PCA` `#tSNE`  
`#dimensionality-reduction` `#CuMiDa` `#visualization` `#python` `#scikit-learn`

## 👤 Author

**Sina Abyar**  
Biotechnology Student, University of Tehran  
**Date:** July 2025
