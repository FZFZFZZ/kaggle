import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import logging
from datetime import datetime
import math
import warnings

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

pd.set_option('display.float_format', '{:.5f}'.format)

plots_dir = "plots"
os.makedirs(plots_dir, exist_ok=True)

current_time = datetime.now()

missing_income_zscore = -0.674  # Estimate those missing income has income at lower quartile
missing_health_zscore = -0.253  # Estimate those missing health score has income at lower 40%
missing_credit_zscore = -0.253  # Estimate those missing credit score has income at lower 40%
missing_occupation_unemployed_p = 0.7  # Estimate those missing occupations have 70% probability to be unemployed
missing_occupation_self_employed_p = 0.3  # Estimate those missing occupations have 30% probability to be self-employed
missing_prev_claims = 0  # Guess missing prev claims means no claims before

def load_data(path):
    try:
        data = pd.read_csv(path)
        logger.info("Successfully Loaded data")
        return data
    except FileNotFoundError:
        logger.error(f"Error: The file '{path}' does not exist.")
    except pd.errors.EmptyDataError:
        logger.error(f"Error: The file '{path}' is empty.")
    except pd.errors.ParserError:
        logger.error(f"Error: Failed to parse the file '{path}'. It might not be a valid CSV.")
    except Exception as e:
        logger.error(f"An unexpected error occurred while reading '{path}': {e}")
    return None

def analyse(data, path):
    # File size
    size = os.path.getsize(path) / (1024 * 1024)
    logger.info(f"File Size in MB: {size}")

    # Peak
    logger.info("data.head")
    print(data.head())
    logger.info(f"shape {data.shape}")
    logger.info(f"data type {data.dtypes}")

    # Summary
    logger.info("Descriptive Statistics (Numerical Features)")
    print(data.describe())
    logger.info("Frequency Counts (Categorical Features)")
    categorical_cols = data.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        print(f"--- {col} ---")
        print(data[col].value_counts())
        print("\n")
    pass

    # Missing Values
    logger.info(f"Missing data Summary {data.isnull().sum()}")
    logger.info("Missing Data Heatmap")
    plt.figure(figsize=(12, 6))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title("Missing Data Heatmap")
    plt.savefig(os.path.join(plots_dir, "missing_data_heatmap.png"))
    plt.close()


    # Data Distribution and Visualization
    data.hist(bins=15, figsize=(15, 10), layout=(4, 4))
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "numerical_histograms.png"))
    plt.close()
    logger.info("Numerical Features Distribution Saved")
    numerical_cols = data.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        plt.figure(figsize=(8, 4))
        sns.boxplot(x=data[col])
        plt.title(f"Boxplot of {col}")
        plt.savefig(os.path.join(plots_dir, f"boxplot_{col}.png"))
        plt.close()
    logger.info("Categorical Features Distribution Saved")

    # Process Data
    data_ = data.copy()
    data_['Policy Start Date'] = pd.to_datetime(data_['Policy Start Date'])
    data_['days_diff'] = data_['Policy Start Date'].apply(lambda x: round((current_time - x).days))
    data_.drop(columns=['Policy Start Date'], inplace=True)
    columns_to_encode = ['Gender', 'Marital Status', 'Education Level', 
                         'Occupation', 'Location', 'Policy Type', 
                         'Customer Feedback', 'Smoking Status', 
                         'Exercise Frequency', 'Property Type']
    onehot_encoded_data = pd.get_dummies(data_[columns_to_encode])
    remaining_columns = data_.drop(columns=columns_to_encode)
    remaining_columns['P_Premium Amount'] = remaining_columns['Premium Amount'].apply(lambda x: math.log(1/x))
    remaining_columns['P_Annual Income'] = remaining_columns['Annual Income'].apply(lambda x: math.log(1/x))
    remaining_columns = remaining_columns.drop(['Premium Amount', 'Annual Income'], axis=1)
    final_data = pd.concat([remaining_columns, onehot_encoded_data], axis=1)
    
    # Correlation Matrix
    logger.info("Correlation Matrix")
    numeric_data = final_data.select_dtypes(include=['int64', 'float64'])
    numeric_data = numeric_data.apply(lambda x: (x - x.mean()) / x.std())
    corr = numeric_data.corr()
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(os.path.join(plots_dir, "correlation_heatmap.png"))
    plt.close()
    
    cat_data = final_data.drop(columns=numeric_data.columns, errors='ignore')
    cat_data['P_Premium Amount'] = numeric_data['P_Premium Amount']
    correlations = cat_data.corr()['P_Premium Amount'][:-1]
    print(correlations)



def clean_data(data):
    # Dtype Conversion
    data_ = data.copy()
    data_['Policy Start Date'] = pd.to_datetime(data_['Policy Start Date'])
    data_['days_diff'] = data_['Policy Start Date'].apply(lambda x: round((current_time - x).days))
    data_.drop(columns=['Policy Start Date'], inplace=True)

    # Missing value handling
    columns_for_row_erase = ['Gender', 'Education Level', 'Location', 'Policy Type',
                             'Vehicle Age', 'Insurance Duration', 'days_diff', 
                             'Smoking Status', 'Exercise Frequency', 'Property Type',
                             'Premium Amount']
    data_ = data_.dropna(subset=columns_for_row_erase)
    columns_for_mode_use = ['Age', 'Marital Status', 'Number of Dependents', 'Customer Feedback']
    for col in columns_for_mode_use:
        data_[col].fillna(data_[col].mode()[0], inplace=True)
    data_['Not Claimed Before'] = data_['Previous Claims'].isnull().astype(int)
    data_['Previous Claims'] = data_['Previous Claims'].fillna(missing_prev_claims)
    data_['Occupation'] = data_['Occupation'].apply(
        lambda x: np.random.choice(['Unemployed', 'Self-Employed'], p=[missing_occupation_unemployed_p, missing_occupation_self_employed_p]) if pd.isnull(x) else x
    )

    # One hot encoding
    columns_to_encode = ['Gender', 'Marital Status', 'Education Level', 
                         'Occupation', 'Location', 'Policy Type', 
                         'Customer Feedback', 'Smoking Status', 
                         'Exercise Frequency', 'Property Type', 'Not Claimed Before']
    onehot_encoded_data = pd.get_dummies(data_[columns_to_encode])
    remaining_columns = data_.drop(columns=columns_to_encode)

    # Rescaling
    remaining_columns['P_Premium Amount'] = remaining_columns['Premium Amount'].apply(lambda x: math.log(1/x))
    remaining_columns['P_Annual Income'] = remaining_columns['Annual Income'].apply(lambda x: math.log(1/x))
    remaining_columns = remaining_columns.drop(['Premium Amount', 'Annual Income'], axis=1)
    
    # Standardise numerics and final missing value filling
    remaining_columns = remaining_columns.apply(lambda x: (x - x.mean()) / x.std())
    data__ = pd.concat([remaining_columns, onehot_encoded_data], axis=1)
    data__['P_Annual Income'] = data__['P_Annual Income'].apply(
        lambda x: missing_income_zscore if pd.isnull(x) else x
    )
    data__['Credit Score'] = data__['Credit Score'].apply(
        lambda x: missing_credit_zscore if pd.isnull(x) else x
    )
    data__['Health Score'] = data__['Health Score'].apply(
        lambda x: missing_health_zscore if pd.isnull(x) else x
    )

    #missing_data = data__[data__['Credit Score'].isnull()]
    #缺失income的人很倾向于拥有低信用分 比较倾向于申请更多premium amount
    #缺失health score的人比较倾向于拥有高信用分与申请更多premium amount
    #缺失credit score的人有一些些倾向于申请更多premium amount

    #print(missing_data.describe())
    #categorical_cols = missing_data.select_dtypes(bool).columns
    #for col in categorical_cols:
    #    true_percentage_missing = (missing_data[col].sum() / len(missing_data[col])) * 100
    #    true_percentage_data = (data__[col].sum() / len(data__[col])) * 100

    #    print(f"Column: {col}")
    #    print(f"Percentage of True values in 'missing_age_data': {true_percentage_missing:.2f}%")
    #    print(f"Percentage of True values in 'data__': {true_percentage_data:.2f}%")
    #    print("-" * 50)

    # Remove Anomalies
    #float_columns = data__.select_dtypes(include=['float64']).columns
    #data___ = data__[(data__[float_columns] <= 3).all(axis=1) & (data__[float_columns] >= -3).all(axis=1)]
    data___=data__
    data___["P_Premium Amount"] = data___["P_Premium Amount"].apply(lambda x: 1 / math.exp(x * 1.101872263802453 - 6.590525060877523))

    data___ = data___.drop(columns=[
        'id', 'Previous Claims', 'Number of Dependents', 'Vehicle Age',
        'Insurance Duration', 'days_diff', 'Gender_Female', 'Gender_Male',
        'Marital Status_Single', 'Marital Status_Divorced',
        "Education Level_Bachelor's", 'Education Level_High School',
        'Occupation_Employed', 'Occupation_Self-Employed',
        'Location_Suburban', 'Location_Urban', 'Policy Type_Basic', 'Policy Type_Comprehensive',
        'Customer Feedback_Average', 'Customer Feedback_Good', 'Customer Feedback_Poor',
        'Smoking Status_No', 'Smoking Status_Yes',
        'Exercise Frequency_Daily', 'Exercise Frequency_Monthly', 'Exercise Frequency_Rarely', 'Exercise Frequency_Weekly'
    ])

    # Replace boolean values with integers
    data___ = data___.replace({True: 1, False: 0})
    logger.info("Successfully Processed data")
    head_data = data___.head()
    head_data.to_csv("head_output.csv", index=False)
    return data___

