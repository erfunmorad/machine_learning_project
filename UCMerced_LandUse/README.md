[README (3).md](https://github.com/user-attachments/files/35103045/README.3.md)

# UC Merced Land Use Classification

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![scikit-learn](https://img.shields.io/badge/scikit--learn-KNN%20%7C%20SVM-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-image%20processing-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
  
Dataset : https://www.kaggle.com/datasets/happyyang/ucmerced-landuse   

Classical machine learning (KNN and SVM) for classifying aerial images into 5 land-use categories, using raw pixel features and hyperparameter tuning with `GridSearchCV`.

## 📌 Overview

This project classifies high-resolution aerial images from the **UC Merced Land Use** dataset. Instead of deep learning, it uses a classical pipeline: images are resized, flattened into feature vectors, standardized, and passed to KNN and SVM classifiers whose hyperparameters are tuned with cross-validated grid search.

## 📊 Dataset

- **Name:** UC Merced Land Use Dataset (5-class subset)
- **Images used:** 500 aerial images, 100 per class
- **Classes:** 5 (airplane, beach, buildings, overpass, parkinglot)
- **Original size:** 256×256 px, ~0.3 m/pixel resolution
- **Split:** 400 training / 100 test images
- **Source:** [UC Merced Computer Vision Lab](http://weegee.vision.ucmerced.edu/datasets/landuse.html)

The full dataset has 21 classes (2,100 images); this project uses only the five classes above.

Expected folder layout:

```
ax/
|-- airplane/
|-- beach/
|-- buildings/
|-- overpass/
`-- parkinglot/
```

## 🛠️ Methodology

1. **Loading:** read each image with OpenCV and assign a label from its folder name.
2. **Preprocessing:** resize to 64×64 and flatten to a 12,288-dimensional vector.
3. **Split:** 80% train / 20% test (`random_state=42`).
4. **Pipeline:** `StandardScaler` → classifier. Scaling happens inside the pipeline, so there is no data leakage during cross-validation.
5. **Hyperparameter tuning:** `GridSearchCV` with the search spaces below.
6. **Evaluation:** accuracy, classification report, and confusion matrix on the held-out test set.

### Hyperparameter search space

| Model | Parameters searched | CV folds |
|-------|---------------------|----------|
| KNN | `n_neighbors` (3, 5, 7, 9, 11), `weights`, `metric` | 5 |
| SVM | `kernel` (linear, rbf), `C` (0.1, 1, 10, 100), `gamma` | 3 |

## 📈 Results

| Model | Best parameters | CV accuracy | Test accuracy |
|-------|-----------------|-------------|---------------|
| KNN | `metric=manhattan`, `n_neighbors=3`, `weights=distance` | 0.385 | 0.44 |
| SVM | `kernel=rbf`, `C=10`, `gamma=0.0001` | 0.700 | **0.69** |

**Best model:** SVM (RBF kernel) with **69% test accuracy**, compared with 44% for KNN.

### Per-class performance (SVM, test set)

| Class | Precision | Recall | F1-score | Support |
|-------|-----------|--------|----------|---------|
| airplane | 0.83 | 0.68 | 0.75 | 28 |
| beach | 0.78 | 1.00 | 0.88 | 14 |
| buildings | 0.48 | 1.00 | 0.65 | 10 |
| overpass | 0.78 | 0.58 | 0.67 | 24 |
| parkinglot | 0.60 | 0.50 | 0.55 | 24 |
| **Macro avg** | 0.69 | 0.75 | 0.70 | 100 |

### Visualizations

![Confusion Matrix](confusion_matrix.png)

![Model Comparison](model_comparison.png)

### Observations

- **SVM clearly outperforms KNN** (69% vs 44%). Distances between raw 12,288-dimensional pixel vectors are not very informative, which hurts KNN more than an RBF-kernel SVM.
- **Beach** is recognized perfectly (recall 1.00) and **buildings** are never missed, but many other images are wrongly predicted as buildings (precision 0.48).
- **Parkinglot** is the hardest class (F1 0.55); it is most often confused with buildings (6 of 24 images).
- **Overpass** is confused with beach and parkinglot (4 images each), and **airplane** with buildings and parkinglot (4 images each).
- The test set has only 100 images, so these numbers carry some uncertainty.

## 🚀 Getting Started

```bash
git clone https://github.com/erfunmorad/machine_learning_project.git
cd machine_learning_project
pip install numpy pandas matplotlib opencv-python scikit-learn pillow jupyter
jupyter notebook UCMerced_LandUse_ML.ipynb
```

Before running, download the dataset and set `folderpath` in the "Loading the dataset" cell of the notebook to the folder that contains the class subfolders (`airplane`, `beach`, `buildings`, `overpass`, `parkinglot`).

## 📁 Project Structure

```
|-- UCMerced_LandUse_ML.ipynb   # full workflow: loading, training, tuning, evaluation
|-- confusion_matrix.png
|-- model_comparison.png
`-- README.md
```

## 🔮 Future Work

- Use all 21 classes of the dataset
- Replace raw pixels with HOG, color histograms, or LBP features
- Add PCA to reduce dimensionality and speed up training
- Try Random Forest, XGBoost, and a CNN with transfer learning (ResNet, EfficientNet)
- Add data augmentation

## 📚 Citation

Yang, Y., & Newsam, S. (2010). *Bag-of-Visual-Words and Spatial Extensions for Land-Use Classification.* ACM SIGSPATIAL GIS.

## 👤 Author

**Erfun Morad** – [GitHub](https://github.com/erfunmorad)

## 📄 License

This project is licensed under the MIT License. The dataset is subject to its own terms from UC Merced.
