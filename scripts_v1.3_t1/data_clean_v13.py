import pandas as pd
import numpy as np

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

df = pd.read_csv("dataA.csv", encoding="utf-8")
df["G"] = df["检测孕周"].apply(parse_gestational_age)
# Additional cleaning: drop rows with NaN in key columns
df_clean = df.dropna(subset=["G", "Y染色体浓度", "孕妇BMI"])
df_clean.to_csv("dataA_Processed_v13.csv", index=False, encoding="utf-8")
print("Processed data exported to dataA_Processed_v13.csv")