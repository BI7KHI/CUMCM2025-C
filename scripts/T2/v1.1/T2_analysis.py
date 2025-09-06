import pandas as pd
import numpy as np
from lifelines import KaplanMeierFitter
from sklearn.utils import resample
import os
import matplotlib.pyplot as plt
import seaborn as sns

def load_and_preprocess_data(data_path):
    """
    Loads and preprocesses the time-to-event data.
    """
    df = pd.read_csv(data_path)
    # The new dataset comes with 'bmi_group', so we just need to ensure column names are consistent.
    df.rename(columns={'时间': 'time', '达标': 'event', '孕妇BMI': 'BMI'}, inplace=True)
    df['time'] = pd.to_numeric(df['time'], errors='coerce')
    df['BMI'] = pd.to_numeric(df['BMI'], errors='coerce')
    df['event'] = pd.to_numeric(df['event'], errors='coerce')
    # No need to drop 'BMI' NA here if it's only used for grouping that's already done.
    df.dropna(subset=['time', 'event', 'bmi_group'], inplace=True)
    return df


def plot_kaplan_meier_curves(df, output_dir):
    """
    Generates and saves Kaplan-Meier survival curves for each BMI group.
    The plot shows the probability of NOT having a conclusive Y-chromosome
    result over gestational time.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 7))

    for group_name, group_df in df.groupby('bmi_group'):
        if group_df.empty:
            print(f"Warning: BMI group '{group_name}' is empty. Skipping.")
            continue
        kmf = KaplanMeierFitter()
        kmf.fit(group_df['time'], event_observed=group_df['event'], label=f"{group_name} (n={len(group_df)})")
        kmf.plot_survival_function(ax=ax)

    ax.set_title('Kaplan-Meier Curves for Time to Conclusive NIPT Result by BMI Group', fontsize=16)
    ax.set_xlabel('Gestational Age (weeks)', fontsize=12)
    ax.set_ylabel('Probability of Awaiting Conclusive Result (Survival Probability)', fontsize=12)
    ax.legend(title='BMI Group', fontsize=10)
    ax.set_ylim(0, 1)
    # Ensure xlim starts from a sensible place, e.g., 0 or the minimum observation time.
    ax.set_xlim(left=max(0, df['time'].min() - 1))
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'kaplan_meier_curves_grouped.png'))
    plt.close()


def find_optimal_testing_week(df, cost_failure, cost_delay_per_week, min_gestation_weeks, max_gestation_weeks):
    """
    Calculates the optimal testing week for each BMI group by minimizing a risk function.
    The risk function balances the cost of a failed test (requiring a re-test) against
    the cost of delaying the test.
    Risk(g) = C_failure * S(g) + C_delay * (g - g_min)
    where S(g) is the survival probability (failure probability) at week g.
    """
    optimal_weeks = {}
    risk_data = []

    for group_name, group_df in df.groupby('bmi_group', observed=False):
        if group_df.empty:
            print(f"Warning: BMI group '{group_name}' is empty in find_optimal_testing_week. Skipping.")
            continue

        kmf = KaplanMeierFitter()
        kmf.fit(group_df['time'], event_observed=group_df['event'])
        survival_func = kmf.survival_function_

        group_risks = {}
        gestational_range_weeks = range(min_gestation_weeks, max_gestation_weeks + 1)

        for week in gestational_range_weeks:
            # Get survival probability at this time (which is now in weeks). Use forward-fill.
            s_g = survival_func.loc[survival_func.index <= week, 'KM_estimate']
            prob_failure = s_g.iloc[-1] if not s_g.empty else 1.0

            delay_cost = cost_delay_per_week * max(0, week - min_gestation_weeks)
            risk = cost_failure * prob_failure + delay_cost
            group_risks[week] = risk
            risk_data.append({'bmi_group': group_name, 'week': week, 'risk': risk, 'prob_failure': prob_failure})

        if group_risks:
            optimal_week = min(group_risks, key=group_risks.get)
            optimal_weeks[group_name] = optimal_week

    risk_df = pd.DataFrame(risk_data)
    return optimal_weeks, risk_df

def plot_risk_curves(risk_df, output_dir):
    """
    Plots the calculated risk for each week, showing how the optimal week is determined.
    """
    plt.style.use('seaborn-v0_8-whitegrid')
    g = sns.relplot(
        data=risk_df,
        x="week", y="risk",
        hue="bmi_group", col="bmi_group",
        kind="line", col_wrap=2,
        height=4, aspect=1.5,
        legend=False
    )
    g.set_titles("Risk vs. Gestational Week for {col_name} Group", fontsize=14)
    g.set_axis_labels("Gestational Week", "Calculated Risk", fontsize=12)
    
    # Highlight the minimum risk point for each group
    for ax, (group_name, group_df) in zip(g.axes.flatten(), risk_df.groupby('bmi_group')):
        if group_df.empty:
            continue
        min_risk_point = group_df.loc[group_df['risk'].idxmin()]
        ax.axvline(x=min_risk_point['week'], color='red', linestyle='--', label=f"Optimal Week: {min_risk_point['week']:.0f}")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'risk_vs_gestational_week_grouped.png'))
    plt.close()


def perform_bootstrap(df, n_iterations, cost_failure, cost_delay_per_week, min_gestation_weeks, max_gestation_weeks):
    """
    Performs bootstrap resampling to estimate the confidence intervals for the
    optimal testing week. This helps assess the stability of the result.
    """
    bootstrap_results = {group: [] for group in df['bmi_group'].unique() if pd.notna(group)}

    for i in range(n_iterations):
        # Stratified resampling to maintain group proportions
        sample_df = df.groupby('bmi_group', group_keys=False).apply(lambda x: x.sample(frac=1.0, replace=True))
        
        try:
            optimal_weeks_sample, _ = find_optimal_testing_week(
                sample_df, cost_failure, cost_delay_per_week, min_gestation_weeks, max_gestation_weeks
            )
            for group, week in optimal_weeks_sample.items():
                if group in bootstrap_results:
                    bootstrap_results[group].append(week)
        except Exception:
            # This can happen if a bootstrap sample for a group has no events
            continue

    confidence_intervals = {}
    for group, weeks in bootstrap_results.items():
        if weeks:
            lower = np.percentile(weeks, 2.5)
            upper = np.percentile(weeks, 97.5)
            confidence_intervals[group] = (lower, upper)

    return confidence_intervals


def main():
    """    Main function to run the analysis.
    """    # --- Setup Paths ---
    # Get the absolute path of the script's directory
    scripts_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 脚本在 scripts/T2/v1.1/ 中，需要回到项目根目录
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(scripts_dir)))
    data_path = os.path.join(project_root, 'data', 'T2', 'processed', 'time_to_event_dataset_grouped.csv')
    output_dir = os.path.join(scripts_dir, 'results_T2')
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Load and preprocess data ---\
    print("1. Loading and preprocessing data...")
    df = load_and_preprocess_data(data_path)
    print(f"Loaded {len(df)} records for survival analysis.")

    # --- 2. BMI group counts (no assignment needed) ---\
    print("\\n--- BMI Group Counts ---")
    print(df['bmi_group'].value_counts())
    print("------------------------\\n")

    # --- 3. Plot Kaplan-Meier curves ---\
    print("3. Plotting Kaplan-Meier curves...")
    plot_kaplan_meier_curves(df, output_dir)

    # --- 4. Define cost parameters and find optimal testing week ---
    print("4. Finding optimal testing week with risk analysis...")
    COST_FAILURE = 1.0
    COST_DELAY_PER_WEEK = 0.05
    MIN_GESTATION_WEEKS = 10
    MAX_GESTATION_WEEKS = 24

    optimal_weeks, risk_df = find_optimal_testing_week(
        df,
        cost_failure=COST_FAILURE,
        cost_delay_per_week=COST_DELAY_PER_WEEK,
        min_gestation_weeks=MIN_GESTATION_WEEKS,
        max_gestation_weeks=MAX_GESTATION_WEEKS
    )

    print("\n--- Optimal Testing Weeks ---")
    for group, week in optimal_weeks.items():
        print(f"  - {group}: Week {week}")

    # --- 5. Plot risk curves ---
    print("\n5. Plotting risk analysis curves...")
    plot_risk_curves(risk_df, output_dir)

    # --- 6. Perform bootstrap analysis for confidence intervals ---
    print("\n6. Performing bootstrap for confidence intervals (this may take a moment)...")
    n_bootstraps = 500
    confidence_intervals = perform_bootstrap(
        df,
        n_iterations=n_bootstraps,
        cost_failure=COST_FAILURE,
        cost_delay_per_week=COST_DELAY_PER_WEEK,
        min_gestation_weeks=MIN_GESTATION_WEEKS,
        max_gestation_weeks=MAX_GESTATION_WEEKS
    )

    print("\n--- 95% Confidence Intervals for Optimal Week ---")
    for group, ci in confidence_intervals.items():
        print(f"  - {group}: ({ci[0]:.2f}, {ci[1]:.2f}) weeks")

    print(f"\nAnalysis complete. Results are saved in the '{output_dir}' directory.")


if __name__ == "__main__":
    main()