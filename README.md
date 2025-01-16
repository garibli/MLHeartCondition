# ML Heart Condition Prediction Project

## Overview
This project focuses on predicting the likelihood of stroke occurrences using machine learning models. By leveraging various preprocessing techniques, balancing methods, and supervised learning algorithms, the project aims to provide an insightful analysis and accurate predictions for health-related datasets.

## Features
- **Data Preprocessing**: Handles missing values, encodes categorical features, and standardizes numerical data.
- **Synthetic Data Generation**: Uses SMOTE (Synthetic Minority Oversampling Technique) to balance the dataset.
- **Exploratory Data Analysis (EDA)**: Generates visualizations like heatmaps, histograms, and boxplots to analyze the dataset's characteristics.
- **Machine Learning Models**: Trains and compares the performance of multiple machine learning algorithms, including:
  - Decision Tree
  - Random Forest
  - Support Vector Machine (SVM)
  - Logistic Regression
  - XGBoost
  - K-Nearest Neighbors (KNN)
- **Model Evaluation**: Evaluates models based on accuracy, AUC-ROC, and cross-validation scores. Visualizes confusion matrices and ROC curves.
- **Performance Comparison**: Summarizes and compares the performance of all trained models.

## Data Preprocessing
1. **Handling Missing Values**: Uses KNN imputation to fill missing values in numerical columns.
2. **Categorical Encoding**: Converts categorical variables to one-hot encoded features.
3. **Age Grouping**: Divides the `age` column into age groups (child, young, middle, senior) for better feature representation.
4. **Standardization**: Scales numerical features using `StandardScaler`.

## Dataset Analysis
- Visualizes correlations among numerical features.
- Displays class distribution before and after balancing with SMOTE.
- Analyzes outliers and data distributions using histograms and boxplots.

## Machine Learning Algorithms
The project implements and compares the following algorithms:
- **Decision Tree**: Finds the best splits in the dataset for classification.
- **Random Forest**: Builds an ensemble of decision trees to improve accuracy and reduce overfitting.
- **SVM**: Separates classes using hyperplanes in a high-dimensional space.
- **Logistic Regression**: Predicts the probability of a binary outcome.
- **XGBoost**: Utilizes gradient boosting for robust and fast classification.
- **K-Nearest Neighbors**: Classifies data points based on the majority class among the nearest neighbors.

Each algorithm is tuned using grid search to find the best hyperparameters.

## Visualization
- **Heatmaps**: Illustrate correlations between numerical features.
- **Class Distribution**: Show before and after balancing the dataset.
- **Histograms and Boxplots**: Highlight data distributions and outliers.
- **Confusion Matrices**: Visualize model performance on test data.
- **ROC Curves**: Compare the AUC for different models.
- **Performance Bar Charts**: Summarize model metrics in a visually compelling way.

## Results
After training and evaluating the models, the results include:
- Accuracy
- AUC (Area Under the ROC Curve)
- Cross-validation scores

These metrics help determine the most effective model for stroke prediction.

## Project Structure
```
MLHeartCondition
|-- README.md
|-- requirements.txt
|-- dataset_analysis.py
|-- model_training.py
|-- utils.py
|-- heartdataset.csv
```
- `README.md`: This file provides an overview of the project.
- `requirements.txt`: Lists all dependencies for running the project.
- `dataset_analysis.py`: Contains functions for exploratory data analysis and preprocessing.
- `model_training.py`: Includes code for training, evaluating, and comparing machine learning models.
- `utils.py`: Contains helper functions for data handling.
- `heartdataset.csv`: The dataset used for analysis and prediction.

## How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/garibli/MLHeartCondition.git
   cd MLHeartCondition
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place the dataset in the project folder if not already present.
4. Run the main script:
   ```bash
   python main.py
   ```

## Dependencies
The project requires the following Python libraries:
- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `imbalanced-learn`
- `matplotlib`
- `seaborn`

Install them using:
```bash
pip install -r requirements.txt
```

## Future Improvements
- Adding more feature engineering techniques.
- Incorporating additional datasets for better generalization.
- Experimenting with advanced hyperparameter optimization techniques like Bayesian Optimization.

## Acknowledgments
Thanks to the contributors like Mustafa Ikbal Avcı and Yunus Ishler and open-source community for providing the tools and libraries used in this project.

