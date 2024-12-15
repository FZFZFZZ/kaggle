import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import warnings

warnings.filterwarnings("ignore")

def load_data(file_path):
    try:
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        print(f"📂 File size: {file_size_mb:.2f} MB")
        print("🔄 Loading data...")
    
        data = pd.read_csv(file_path)
        print("✅ Successfully loaded data.")
        print("🔍 Preview of the dataset:")
        print(data.head())
        return data
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' does not exist.")
    except pd.errors.EmptyDataError:
        print(f"Error: The file '{file_path}' is empty.")
    except pd.errors.ParserError:
        print(f"Error: Failed to parse the file '{file_path}'. It might not be a valid CSV.")
    except Exception as e:
        print(f"An unexpected error occurred while reading '{file_path}': {e}")
    return None

def clean_data(raw_data):
    raw_data['FamilySize'] = raw_data['SibSp'] + raw_data['Parch'] + 1
    raw_data['IsAlone'] = 1
    raw_data['IsAlone'].loc[raw_data['FamilySize'] > 1] = 0
    data = raw_data[["Sex", "Pclass", "Fare", "Age", "Survived", "Embarked", "IsAlone"]]
    data['AgeGroup'] = data['Age'].apply(lambda x: 0 if x < 16 else 1)
    data['FareAbove20'] = (data['Fare'] > 15).astype(int)
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 0})
    data.drop("Age", axis=1, inplace=True)
    data.drop("Fare", axis=1, inplace=True)
    categorical_features = ['Pclass']
    data_encoded = pd.get_dummies(data, columns=categorical_features, drop_first=True)
    print(data_encoded.head())
    return data_encoded

def analyse(data):
    # Create a directory for saving plots
    plots_dir = "plots"
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Overview of the Dataset
    print("===== Dataset Overview =====")
    print(f"Shape: {data.shape}\n")
    
    print("===== Data Types =====")
    print(data.dtypes, "\n")
    
    print("===== First 5 Rows =====")
    print(data.head(), "\n")
    
    # 2. Summary Statistics
    print("===== Descriptive Statistics (Numerical Features) =====")
    print(data.describe(), "\n")
    
    print("===== Frequency Counts (Categorical Features) =====")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"--- {col} ---")
        print(data[col].value_counts())
        print("\n")
    
    # 3. Missing Values
    print("===== Missing Values =====")
    print(data.isnull().sum(), "\n")
    
    # Missing Data Heatmap
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.savefig(os.path.join(plots_dir, "missing_data_heatmap.png"))
    plt.close()
    
    # 4. Data Distribution and Visualization
    print("===== Numerical Features Distribution =====")
    data.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "numerical_histograms.png"))
    plt.close()
    
    print("===== Boxplots for Numerical Features =====")
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(os.path.join(plots_dir, f"boxplot_{col}.png"))
        plt.close()

    
    # Correlation Matrix
    print("===== Correlation Matrix =====")
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1
    data['IsAlone'] = 1  # Initialize to 1 (True)
    data['IsAlone'].loc[data['FamilySize'] > 1] = 0
    data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
    data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 0})
    data['AgeGroup'] = data['Age'].apply(lambda x: 0 if x < 16 else 1)
    data['FareAbove20'] = (data['Fare'] > 10).astype(int)
    data['Cabin'] = data['Cabin'].fillna('U')  # 'U' for Unknown
    data['Deck'] = data['Cabin'].str[0]
    data['Deck'] = data['Deck'].map({'A':1, 'B':1, 'C':2, 'D':2, 'E':3, 'F':3, 'G':3, 'T':3, 'U':4})
    numeric_data = data.select_dtypes(include=['int64', 'float64', 'float32', 'int32'])
    corr = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close()
    print(corr, "\n")
    print('''\n data['Sex'] = data['Sex'].map({'male': 0, 'female': 1})
             data['Embarked'] = data['Embarked'].map({'S': 0, 'C': 1, 'Q': 2})
             data['AgeGroup'] = data['Age'].apply(lambda x: 0 if x < 16 else 1 if x < 64 else 2)
             data['LogFare'] = np.log1p(data['Fare'])''')
    
    # 5. Feature-Specific Analysis
    print("===== Survival Rate by Categorical Features =====")
    for col in categorical_cols:
        if col != 'Survived':
            plt.figure(figsize=(8, 4))
            sns.barplot(x=col, y='Survived', data=data)
            plt.title(f"Survival Rate by {col}")
            plt.xticks(rotation=45)
            plt.savefig(os.path.join(plots_dir, f"survival_rate_{col}.png"))
            plt.close()
    
    print("===== Age Distribution by Survival Status =====")
    sns.histplot(data=data, x='Age', hue='Survived', multiple='stack', kde=True)
    plt.title("Age Distribution by Survival Status")
    plt.savefig(os.path.join(plots_dir, "age_distribution_survived.png"))
    plt.close()
    
    print("===== Fare Distribution by Survival Status =====")
    sns.histplot(data=data, x='Fare', hue='Survived', multiple='stack', kde=True)
    plt.title("Fare Distribution by Survival Status")
    plt.savefig(os.path.join(plots_dir, "fare_distribution_survived.png"))
    plt.close()
    
    # 6. Handling Outliers (Example: Fare)
    print("===== Outliers Detection =====")
    Q1 = data['Fare'].quantile(0.25)
    Q3 = data['Fare'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = data[(data['Fare'] < lower_bound) | (data['Fare'] > upper_bound)]
    print(f"Number of outliers in Fare: {outliers.shape[0]}\n")
    
    # Optional: visualize outliers
    plt.figure(figsize=(8, 4))
    sns.boxplot(x=data['Fare'])
    plt.title("Boxplot of Fare with Outliers")
    plt.savefig(os.path.join(plots_dir, "fare_boxplot_outliers.png"))
    plt.close()

    print("===== FE decision =====")
    print("Keep processed AgeGroup, logFare, Sex, Embark and Pclass column, add in IsAlone and Deck column")
    return None
