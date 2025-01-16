import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# veri setini yükle
def load_dataset():
    data = pd.read_csv("C:/Users/LENOVO/Desktop/healdataset.csv")
    data.replace({"N/A": np.nan, "Unknown": np.nan}, inplace=True)
    return data
#verisetini gösteren fonksiyon
def data_shower(df):
    return print(df.head(10))

#verisetini ön işleme yapan fonksiyon
def preprocess_data(df):
    print("\nÖnişlemeden önce veriseti bilgileri:\n")
    print(df.info())
    print(df.describe())

    imputer = KNNImputer(n_neighbors=5)
    numerical_columns = df.select_dtypes(include=["float64", "int64"]).columns
    df[numerical_columns] = imputer.fit_transform(df[numerical_columns])

    categorical_columns = df.select_dtypes(include=["object", "category"]).columns
    df = pd.get_dummies(df, columns=categorical_columns, drop_first=True)

    if "age" in df.columns:
        df["age_group"] = pd.cut(df["age"], bins=[0, 18, 35, 55, 100], labels=["child", "young", "middle", "senior"])
        df = pd.get_dummies(df, columns=["age_group"], drop_first=True)

    print("\nÖnişlemeden Sonra Veri Seti Bilgileri:\n")
    print(df.info())
    print(df.describe())

    return df

#smote kullanarak sentetik veri üreten fonksiyon
def balance_data(X, y):
    print("\nDataset Information Before SMOTE:")
    print("Class Distribution:", y.value_counts())

    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    print("\nDataset Information After SMOTE:")
    print("Class Distribution:", pd.Series(y_resampled).value_counts())

    return X_resampled, y_resampled


#veriseti analizi yapan kodlar
def dataset_analysis(df):
    numeric_df = df.select_dtypes(include=["float64", "int64"])
    plt.figure(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.show()

    if "stroke" in df.columns:
        plt.figure(figsize=(6, 4))
        sns.countplot(x="stroke", data=df, hue="stroke", palette="viridis", legend=False)
        plt.title("Sınıf Ayrımı")
        plt.show()

    numeric_df.hist(bins=15, figsize=(20, 15), color="steelblue")
    plt.suptitle("Numerik Özellikler için Histogram")
    plt.show()

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=numeric_df)
    plt.title("Ayrık Değer Analizi için Boxplot")
    plt.show()


#ML algoritmaları
def train_and_compare_models(X_train, X_test, y_train, y_test):
    models = {
        "Decision Tree": {
            "model": DecisionTreeClassifier(random_state=42),
            "params": {
                "max_depth": [3, 5, 10, None],
                "min_samples_split": [2, 5, 10]
            }
        },
        "Random Forest": {
            "model": RandomForestClassifier(random_state=42),
            "params": {
                "n_estimators": [50, 100, 200],
                "max_depth": [3, 5, 10, None]
            }
        },
        "SVM": {
            "model": SVC(probability=True, random_state=42),
            "params": {
                "C": [0.1, 1, 10],
                "kernel": ["linear", "rbf"]
            }
        },
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, random_state=42),
            "params": {
                "C": [0.1, 1, 10]
            }
        },
        "XGBoost": {
            "model": XGBClassifier(random_state=42, eval_metric='logloss'),
            "params": {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2]
            }
        },
        "KNN": {
            "model": KNeighborsClassifier(),
            "params": {
                "n_neighbors": list(range(1, 21)),
                "weights": ["uniform", "distance"]
            }
        }
    }

    results = {}

    for name, config in models.items():
        print(f"\nPerforming Grid Search for {name}...")

        # Perform Grid Search
        grid_search = GridSearchCV(config["model"], config["params"], cv=5, scoring="accuracy", verbose=1, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        print(f"Best Parameters for {name}: {grid_search.best_params_}")

        # Evaluate the model using cross-validation
        cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy")
        print(f"Cross-Validation Scores for {name}: {cv_scores}")
        print(f"Mean Cross-Validation Score for {name}: {cv_scores.mean()}")

        # Evaluate the model on the test set
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1] if hasattr(best_model, 'predict_proba') else None

        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None

        results[name] = {
            "Accuracy": acc,
            "AUC": auc,
            "Mean Cross-Val Score": cv_scores.mean()  # Adding cross-validation mean score to results
        }

        print(f"\n{name} Classification Report:\n", classification_report(y_test, y_pred))

        plt.figure(figsize=(6, 5))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
        plt.title(f"Confusion Matrix for {name}")
        plt.show()

        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.2f})")

    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.legend()
    plt.title("ROC Curves for Models")
    plt.show()

    return results

#model performansını gösteren grafikler
def plot_model_performance(results):
    metrics = pd.DataFrame(results).T
    metrics.plot(kind="bar", figsize=(10, 6), colormap="viridis")
    plt.title("Model Performansları Kıyaslanması")
    plt.ylabel("Score")
    plt.xticks(rotation=45)
    plt.legend(loc="best")
    plt.show()


#main fonksiyonu
def main():
    data = load_dataset()
    dataset_analysis(data)
    print("Standardizasyondan sonra veriseti bilgileri: ")
    data_shower(data)

    processed_data = preprocess_data(data)

    X = processed_data.drop("stroke", axis=1)
    y = processed_data["stroke"]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    X_balanced, y_balanced = balance_data(X, y)

    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

    results = train_and_compare_models(X_train, X_test, y_train, y_test)

    plot_model_performance(results)

main()
