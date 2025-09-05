import sys
print("Python version:", sys.version)
print("Current directory:", sys.path[0])

try:
    import pandas as pd
    print("Pandas imported successfully")
    
    # 尝试读取数据文件
    print("Trying to read data.xlsx...")
    df = pd.read_excel('data.xlsx')
    print(f"Successfully read data.xlsx, shape: {df.shape}")
    print("Columns:", list(df.columns))
    print("First few rows:")
    print(df.head())
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

