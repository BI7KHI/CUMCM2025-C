import pandas as pd
import numpy as np
import os
import re

def convert_gestational_week(gestational_week_str):
    """
    将孕周字符串（如 '11w+6' 或 '12w'）转换为连续周数。
    """
    if pd.isna(gestational_week_str):
        return np.nan
    
    gestational_week_str = str(gestational_week_str).strip()
    
    match_wd = re.match(r'(\d+)w\+(\d+)', gestational_week_str)
    if match_wd:
        weeks, days = map(int, match_wd.groups())
        return weeks + days / 7.0
        
    match_w = re.match(r'(\d+)w', gestational_week_str)
    if match_w:
        weeks = int(match_w.groups()[0])
        return float(weeks)
        
    try:
        return float(gestational_week_str)
    except (ValueError, TypeError):
        return np.nan

def prepare_data_for_task2(data_a_path, data_b_path, output_dir):
    """
    为任务二准备数据，根据首次达标时间定义生存分析数据集，并使用BMI四分位数进行分组。
    """
    # 1. 数据读取与合并
    try:
        data_a = pd.read_csv(data_a_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        data_a = pd.read_csv(data_a_path, encoding='gbk')
    data_a.columns = data_a.columns.str.replace('\ufeff', '', regex=True).str.strip()

    try:
        data_b = pd.read_csv(data_b_path, encoding='utf-8-sig')
    except UnicodeDecodeError:
        data_b = pd.read_csv(data_b_path, encoding='gbk')
    data_b.columns = data_b.columns.str.strip()
    
    df = pd.merge(data_a, data_b, on='孕妇代码', how='left', suffixes=('', '_b'))
    df.columns = df.columns.str.strip()

    # 2. 数据清洗和质量控制
    df.dropna(subset=['Y染色体浓度', '检测孕周', '孕妇BMI'], inplace=True)
    df = df[df['原始读段数'] > 1000000]
    df = df[(df['GC含量'] > 0.3) & (df['GC含量'] < 0.5)]
    df = df[df['被过滤掉读段数的比例'] < 0.1]
    df = df[df['唯一比对的读段数'] > 2000000]

    # 2.5 Y染色体浓度端点修正 (Winsorization)
    epsilon = 1e-9
    df['Y染色体浓度'] = df['Y染色体浓度'].apply(lambda p: min(max(p, epsilon), 1 - epsilon))

    # 3. 孕周格式转换
    df['孕周_周数'] = df['检测孕周'].apply(convert_gestational_week)
    df.dropna(subset=['孕周_周数'], inplace=True)

    # 4. 异常值处理
    for col in ['孕妇BMI', '孕周_周数']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])

    # 5. BMI分组 (基于四分位数)
    patient_bmi = df.loc[df.groupby('孕妇代码')['孕周_周数'].idxmin()][['孕妇代码', '孕妇BMI']].set_index('孕妇代码')
    
    q1 = patient_bmi['孕妇BMI'].quantile(0.25)
    q2 = patient_bmi['孕妇BMI'].quantile(0.50)
    q3 = patient_bmi['孕妇BMI'].quantile(0.75)
    
    bins = [0, q1, q2, q3, np.inf]
    labels = [f"Q1 (<{q1:.2f})", f"Q2 ({q1:.2f}-{q2:.2f})", f"Q3 ({q2:.2f}-{q3:.2f})", f"Q4 (>{q3:.2f})"]
    patient_bmi['bmi_group'] = pd.cut(patient_bmi['孕妇BMI'], bins=bins, labels=labels, right=False)
    
    df = df.join(patient_bmi['bmi_group'], on='孕妇代码')
    df.dropna(subset=['bmi_group'], inplace=True)

    # 6. 定义“首次达标时间”生存数据集 (T_i, delta_i)
    Y_CONCENTRATION_THRESHOLD = 0.04
    df_sorted = df.sort_values(by=['孕妇代码', '孕周_周数'])

    def get_time_to_event(group):
        """
        对于每个孕妇分组，确定其事件时间和状态。
        """
        event_records = group[group['Y染色体浓度'] >= Y_CONCENTRATION_THRESHOLD]
        if not event_records.empty:
            # 事件发生：时间为首次达标的孕周
            time = event_records['孕周_周数'].iloc[0]
            event = 1
        else:
            # 删失：时间为最后一次观测的孕周
            time = group['孕周_周数'].iloc[-1]
            event = 0
        
        # 返回该孕妇的事件/删失记录
        return pd.Series({
            '时间': time,
            '达标': event,
            '孕妇BMI': group['孕妇BMI'].iloc[0], # 保证BMI值唯一
            'bmi_group': group['bmi_group'].iloc[0] # 保证分组唯一
        })

    # 将函数应用到每个孕妇分组
    df_time_to_event = df_sorted.groupby('孕妇代码').apply(get_time_to_event).reset_index()

    # 7. 保存结果
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'time_to_event_dataset_grouped.csv')
    df_time_to_event.to_csv(output_path, index=False, encoding='utf-8-sig')

    # 8. 更新报告
    report_path = os.path.join(output_dir, 'data_processing_report.md')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("# 数据处理报告 (任务二)\n\n")
        f.write("## 1. 数据加载与合并...\n")
        f.write("## 2. 数据清洗与质量控制...\n")
        f.write("## 2.5. Y染色体浓度端点修正 (Winsorization)...\n")
        f.write("## 3. 孕周格式转换与异常值处理\n")
        f.write("- **孕周格式转换**: 将字符串格式的孕周（例如 '11w+6' 或 '12w'）统一转换为连续周数（`周 + 天/7`），以便进行数值计算。\n")
        f.write("- **异常值处理**: 对孕妇BMI和转换后的孕周数使用IQR方法进行异常值检测和修正。\n\n")
        f.write("## 4. BMI分组 (基于四分位数)...\n")
        f.write("## 5. '首次达标时间' 数据集构建\n")
        f.write(f"- **事件定义**: Y染色体浓度 >= {Y_CONCENTRATION_THRESHOLD}。\n")
        f.write("- 对每个孕妇，按孕周排序所有检测记录。\n")
        f.write("- **事件时间 (T_i)**: 若存在达标记录，则取首次达标的孕周作为事件时间 (达标=1)。\n")
        f.write("- **删失时间 (C_i)**: 若不存在达标记录，则取最后一次检测的孕周作为删失时间 (达标=0)。\n")
        f.write("- 使用 `groupby().apply()` 方法高效计算每个孕妇的时间和事件状态。\n\n")
        f.write(f"## 6. 输出\n")
        f.write(f"- 处理后的时间-事件数据集已保存到: `{output_path}`\n")

    print(f"数据处理完成，结果已保存至 {output_dir}")
    print(f"处理报告已生成: {report_path}")


if __name__ == '__main__':
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    # 脚本在 scripts/T2/v1.1/ 中，需要回到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(scripts_dir)))
    
    data_a_path = os.path.join(project_root, 'data', 'common', 'source', 'dataA.csv')
    data_b_path = os.path.join(project_root, 'data', 'common', 'source', 'dataB.csv')
    output_dir = os.path.join(project_root, 'data', 'T2', 'processed')
    
    prepare_data_for_task2(data_a_path, data_b_path, output_dir)