import pandas as pd
import numpy as np
from pygam import LinearGAM, s
import os
import io
from contextlib import redirect_stdout

def parse_gestational_age(s):
    try:
        s = str(s).strip()
        if "w+" in s:
            w, d = s.replace("w", "").split("+")
            return int(w) + int(d) / 7
        elif "w" in s:
            return int(s.replace("w", ""))
        else:
            return np.nan
    except:
        return np.nan

# Define paths relative to the project root
# 获取项目根目录路径
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(script_dir))
data_path = os.path.join(project_root, 'data', 'common', 'source', 'dataA.csv')
output_path = os.path.join(project_root, 'data', 'T1', 'processed', 'dataA_Processed_v13.csv')
results_dir = os.path.join(project_root, 'results', 'T1', 'v1.3')

# Create results directory if it doesn't exist
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

df = pd.read_csv(data_path, encoding="utf-8")
df["G"] = df["检测孕周"].apply(parse_gestational_age)
# Additional cleaning: drop rows with NaN in key columns
df_clean = df.dropna(subset=["G", "Y染色体浓度", "孕妇BMI"])
df_clean.to_csv(output_path, index=False, encoding="utf-8")
print(f"Processed data exported to {output_path}")


# --- GAM Model Analysis ---
print("\n--- Starting GAM Model Analysis ---")

# Prepare data for GAM
# Using G (gestational age) and BMI as predictors for Y chromosome concentration
X = df_clean[['G', '孕妇BMI']].values
y = df_clean['Y染色体浓度'].values

# Build and fit the GAM model
# Using splines for both features
gam = LinearGAM(s(0, n_splines=20) + s(1, n_splines=20)).fit(X, y)

# Save the summary to a file
summary_file_path = os.path.join(results_dir, 'T1_gam_from_clean_script_results.txt')

print(f"Saving GAM model summary to: {summary_file_path}")
with open(summary_file_path, 'w', encoding='utf-8') as f:
    with redirect_stdout(f):
        gam.summary()

print("GAM model analysis complete.")