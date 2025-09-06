import pandas as pd
import numpy as np
import os

def convert_gestational_age_to_days(age_str):
    """Converts gestational age string (e.g., '16w+3') to days."""
    if isinstance(age_str, str):
        if 'w+' in age_str:
            weeks, days = age_str.split('w+')
            return int(weeks) * 7 + int(days)
        elif 'w' in age_str:
            weeks = age_str.replace('w', '')
            return int(weeks) * 7
    return np.nan # Return NaN for invalid formats

def clean_data(data_a_path, data_b_path, output_path):
    """
    Cleans and preprocesses fetal data from two sources, now using utf-8-sig encoding.

    Args:
        data_a_path (str): The file path for the first dataset (dataA.csv).
        data_b_path (str): The file path for the second dataset (dataB.csv).
        output_path (str): The file path to save the cleaned data.
    """
    try:
        # Load datasets with 'utf-8-sig' encoding
        data_a = pd.read_csv(data_a_path, encoding='utf-8-sig')
        data_b = pd.read_csv(data_b_path, encoding='utf-8-sig')
    except FileNotFoundError as e:
        print(f"Error loading data files: {e}")
        return
    except Exception as e:
        print(f"An error occurred while reading files: {e}")
        return

    # Combine the datasets
    data = pd.concat([data_a, data_b], ignore_index=True)

    # Convert gestational age to days
    data['检测孕周_天数'] = data['检测孕周'].apply(convert_gestational_age_to_days)

    # 1. Filter for male fetuses
    data_male = data[data['Y染色体的Z值'].notna() & data['Y染色体浓度'].notna()].copy()

    # 2. Handle missing values for '检测孕周' and '孕妇BMI'
    bmi_groups = pd.cut(data_male['孕妇BMI'], bins=[0, 20, 28, 35, np.inf], labels=['<20', '20-28', '28-35', '>=35'])
    data_male['bmi_group'] = bmi_groups
    data_male['检测孕周_天数'] = data_male.groupby('bmi_group')['检测孕周_天数'].transform(lambda x: x.fillna(x.median()))

    if data_male['孕妇BMI'].isnull().sum() / len(data_male) < 0.05:
        bmi_median = data_male['孕妇BMI'].median()
        data_male['孕妇BMI'].fillna(bmi_median, inplace=True)
    else:
        data_male.dropna(subset=['孕妇BMI'], inplace=True)

    data_male.dropna(subset=['检测孕周_天数'], inplace=True)

    # 3. Ensure fetal health data is present
    data_male.dropna(subset=['胎儿是否健康'], inplace=True)

    # Save the cleaned data
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    data_male.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Cleaned data successfully saved to {output_path}")

if __name__ == '__main__':
    # 获取项目根目录路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))

    data_a_path = os.path.join(project_root, 'data', 'common', 'source', 'dataA.csv')
    data_b_path = os.path.join(project_root, 'data', 'common', 'source', 'dataB.csv')
    output_file_path = os.path.join(project_root, 'data', 'common', 'processed', 'cleaned_fetal_data.csv')

    clean_data(data_a_path, data_b_path, output_file_path)