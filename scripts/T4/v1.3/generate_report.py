import os
import io
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)
import joblib

plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial']
plt.rcParams['axes.unicode_minus'] = False


def robust_read_predictions(csv_path: str) -> pd.DataFrame:
    """Robustly read test_predictions.csv which may contain multiple header lines.
    Returns a DataFrame with at least columns: 'y_true', 'prob'.
    """
    # Read raw text first to find the last header line containing y_true
    encodings = ['utf-8', 'utf-8-sig', 'gbk', 'cp936']
    last_header_idx = -1
    lines = None
    for enc in encodings:
        try:
            with open(csv_path, 'r', encoding=enc, errors='ignore') as f:
                lines = f.readlines()
            break
        except Exception:
            continue
    if lines is None:
        raise RuntimeError(f"无法读取文件: {csv_path}")

    for i, ln in enumerate(lines):
        if 'y_true' in ln and 'prob' in ln:
            last_header_idx = i
    if last_header_idx == -1:
        # Fallback: try direct read
        df = pd.read_csv(csv_path, encoding='utf-8', engine='python')
        if 'y_true' not in df.columns or 'prob' not in df.columns:
            raise RuntimeError('未能在预测文件中找到 y_true/prob 列')
        return df

    header_line = lines[last_header_idx].strip('\n')
    data_text = header_line + '\n' + ''.join(lines[last_header_idx + 1:])
    df = pd.read_csv(io.StringIO(data_text), engine='python')
    if 'y_true' not in df.columns or 'prob' not in df.columns:
        raise RuntimeError('未能在预测文件中找到 y_true/prob 列')
    # Coerce numeric
    df['y_true'] = pd.to_numeric(df['y_true'], errors='coerce').fillna(0).astype(int)
    df['prob'] = pd.to_numeric(df['prob'], errors='coerce')
    return df.dropna(subset=['prob']).reset_index(drop=True)


def load_json(path: str, default=None):
    if default is None:
        default = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return default


def plot_roc_pr(y_true, prob):
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    # ROC
    axs[0].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}', color='#1f77b4')
    axs[0].plot([0, 1], [0, 1], 'k--', alpha=0.4)
    axs[0].set_title('ROC 曲线（测试集）')
    axs[0].set_xlabel('FPR')
    axs[0].set_ylabel('TPR')
    axs[0].legend(loc='lower right')
    axs[0].grid(alpha=0.2)
    # PR
    axs[1].plot(rec, prec, label=f'AP = {ap:.3f}', color='#d62728')
    axs[1].set_title('PR 曲线（测试集）')
    axs[1].set_xlabel('Recall')
    axs[1].set_ylabel('Precision')
    axs[1].legend(loc='lower left')
    axs[1].grid(alpha=0.2)
    plt.tight_layout()
    return fig


def plot_confmats(y_true, prob, thr_f1, thr_rec):
    y_pred_f1 = (prob >= thr_f1).astype(int)
    y_pred_rec = (prob >= thr_rec).astype(int)

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    for ax, y_pred, title in [
        (axs[0], y_pred_f1, f'混淆矩阵 - F1最优阈值 ({thr_f1:.3f})'),
        (axs[1], y_pred_rec, f'混淆矩阵 - 召回优先阈值 ({thr_rec:.3f})'),
    ]:
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(cm, display_labels=['正常', '异常'])
        disp.plot(ax=ax, cmap='Blues', colorbar=False)
        ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_coefficients(model_path: str, feature_list_path: str):
    pipe = joblib.load(model_path)
    # Try to locate the logistic regression estimator
    clf = None
    if hasattr(pipe, 'named_steps'):
        clf = pipe.named_steps.get('clf', None)
    if clf is None and hasattr(pipe, 'steps') and len(pipe.steps) > 0:
        clf = pipe.steps[-1][1]
    if clf is None or not hasattr(clf, 'coef_'):
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.text(0.5, 0.5, '无法获取系数（可能为校准或非线性模型）', ha='center', va='center')
        ax.axis('off')
        return fig

    coef = clf.coef_.ravel()
    features = load_json(feature_list_path, default=[])
    if not features or len(features) != len(coef):
        features = [f'f{i}' for i in range(len(coef))]

    imp = pd.DataFrame({'feature': features, 'coef': coef})
    imp['abs'] = imp['coef'].abs()
    topk = imp.sort_values('abs', ascending=False).head(10)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['#2ca02c' if v > 0 else '#d62728' for v in topk['coef']]
    ax.barh(topk['feature'][::-1], topk['coef'][::-1], color=colors[::-1])
    ax.set_title('Logistic 回归系数（按绝对值Top10）')
    ax.set_xlabel('系数')
    plt.tight_layout()
    return fig


def build_summary_figure(y_true, prob, thr_f1, thr_rec, model_path, feature_list_path, out_png, out_svg):
    # Create a multi-panel summary figure
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.25)

    # ROC & PR
    ax1 = fig.add_subplot(gs[0, 0])
    fpr, tpr, _ = roc_curve(y_true, prob)
    roc_auc = auc(fpr, tpr)
    ax1.plot(fpr, tpr, label=f'AUC={roc_auc:.3f}', color='#1f77b4')
    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.3)
    ax1.set_title('ROC（测试集）')
    ax1.set_xlabel('FPR')
    ax1.set_ylabel('TPR')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.2)

    ax2 = fig.add_subplot(gs[0, 1])
    prec, rec, _ = precision_recall_curve(y_true, prob)
    ap = average_precision_score(y_true, prob)
    ax2.plot(rec, prec, label=f'AP={ap:.3f}', color='#d62728')
    ax2.set_title('PR（测试集）')
    ax2.set_xlabel('Recall')
    ax2.set_ylabel('Precision')
    ax2.legend(loc='lower left')
    ax2.grid(alpha=0.2)

    # Confusion matrices
    ax3 = fig.add_subplot(gs[1, 0])
    y_pred_f1 = (prob >= thr_f1).astype(int)
    cm1 = confusion_matrix(y_true, y_pred_f1)
    ConfusionMatrixDisplay(cm1, display_labels=['正常', '异常']).plot(ax=ax3, cmap='Blues', colorbar=False)
    ax3.set_title(f'混淆矩阵 - F1最优阈值 ({thr_f1:.3f})')

    ax4 = fig.add_subplot(gs[1, 1])
    y_pred_rec = (prob >= thr_rec).astype(int)
    cm2 = confusion_matrix(y_true, y_pred_rec)
    ConfusionMatrixDisplay(cm2, display_labels=['正常', '异常']).plot(ax=ax4, cmap='Blues', colorbar=False)
    ax4.set_title(f'混淆矩阵 - 召回优先阈值 ({thr_rec:.3f})')

    plt.tight_layout()
    fig.savefig(out_png, dpi=300)
    try:
        fig.savefig(out_svg)
    except Exception:
        pass
    return fig


def main():
    parser = argparse.ArgumentParser(description='T4 v1.3 报告自动生成')
    parser.add_argument('--results_dir', default=os.path.join('results', 'T4', 'v1.3'))
    parser.add_argument('--output_pdf', default=os.path.join('docs', 'T4_v1.3_report.pdf'))
    parser.add_argument('--output_fig_png', default=os.path.join('results', 'T4', 'v1.3', 'summary_figure.png'))
    parser.add_argument('--output_fig_svg', default=os.path.join('results', 'T4', 'v1.3', 'summary_figure.svg'))
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.output_pdf), exist_ok=True)

    # Load files
    metrics_f1 = load_json(os.path.join(args.results_dir, 'metrics_f1.json'))
    metrics_rec = load_json(os.path.join(args.results_dir, 'metrics_recall_priority.json'))
    thresholds = load_json(os.path.join(args.results_dir, 'thresholds.json'))
    cv_best = load_json(os.path.join(args.results_dir, 'cv_best_params.json'))
    feat_path = os.path.join(args.results_dir, 'feature_list.json')
    model_path = os.path.join(args.results_dir, 'model.joblib')
    female_stats = load_json(os.path.join(args.results_dir, 'female_selection_stats.json'))

    pred_path = os.path.join(args.results_dir, 'test_predictions.csv')
    df_pred = robust_read_predictions(pred_path)
    y_true = df_pred['y_true'].values
    prob = df_pred['prob'].values

    thr_f1 = thresholds.get('best_threshold_f1') or thresholds.get('best_threshold') or metrics_f1.get('best_threshold')
    thr_rec = thresholds.get('recall_priority_threshold') or metrics_rec.get('threshold')
    if thr_f1 is None or thr_rec is None:
        raise RuntimeError('未找到阈值，请检查 thresholds.json 或 metrics 文件')

    # Build figures
    fig_curves = plot_roc_pr(y_true, prob)
    fig_cms = plot_confmats(y_true, prob, thr_f1, thr_rec)
    fig_coef = plot_coefficients(model_path, feat_path)
    build_summary_figure(y_true, prob, thr_f1, thr_rec, model_path, feat_path, args.output_fig_png, args.output_fig_svg)

    # Prepare text blocks
    title_text = (
        'T4 女胎异常判定方法与模型效果（v1.3）\n'\
        '标签：T13/T18/T21 非整倍体为异常；女胎筛选：Y浓度GMM聚类（不足时回退分位数）；'\
        '特征：Z13/Z18/Z21/ZX + GC + 读段与比例 + BMI/孕周；'\
        '模型：StandardScaler+LogReg(class_weight=balanced)，GroupKFold网格搜索；'\
        '阈值：F1最优与召回优先（目标召回默认0.90）。'
    )

    metrics_text = (
        f"F1最优阈值: P={metrics_f1.get('precision', float('nan')):.3f}, "
        f"R={metrics_f1.get('recall', float('nan')):.3f}, F1={metrics_f1.get('f1', float('nan')):.3f}, "
        f"ROC-AUC={metrics_f1.get('roc_auc', float('nan')):.3f}, thr={metrics_f1.get('best_threshold', float('nan')):.3f}\n"\
        f"召回优先阈值: P={metrics_rec.get('precision', float('nan')):.3f}, "
        f"R={metrics_rec.get('recall', float('nan')):.3f}, F1={metrics_rec.get('f1', float('nan')):.3f}, "
        f"ROC-AUC={metrics_rec.get('roc_auc', float('nan')):.3f}, thr={metrics_rec.get('threshold', float('nan')):.3f}"
    )

    cv_text = json.dumps(cv_best, ensure_ascii=False, indent=2)
    fem_text = json.dumps(female_stats, ensure_ascii=False, indent=2)

    # Classification reports if exist
    cr_f1_path = os.path.join(args.results_dir, 'classification_report_f1.txt')
    cr_rec_path = os.path.join(args.results_dir, 'classification_report_recall_priority.txt')
    cr_f1 = ''
    cr_rec = ''
    try:
        cr_f1 = open(cr_f1_path, 'r', encoding='utf-8').read()
    except Exception:
        pass
    try:
        cr_rec = open(cr_rec_path, 'r', encoding='utf-8').read()
    except Exception:
        pass

    with PdfPages(args.output_pdf) as pdf:
        # Page 1: Title + summary
        fig1 = plt.figure(figsize=(8.27, 11.69))  # A4 portrait
        fig1.text(0.5, 0.93, 'T4 女胎异常判定方法与模型效果（v1.3）', ha='center', va='center', fontsize=16, weight='bold')
        fig1.text(0.1, 0.86, title_text, fontsize=11)
        fig1.text(0.1, 0.74, '女胎筛选统计：', fontsize=12, weight='bold')
        fig1.text(0.1, 0.72, fem_text[:1400] + ('...' if len(fem_text) > 1400 else ''), fontsize=9)
        fig1.text(0.1, 0.42, '关键指标：', fontsize=12, weight='bold')
        fig1.text(0.1, 0.40, metrics_text, fontsize=11)
        fig1.tight_layout()
        pdf.savefig(fig1)
        plt.close(fig1)

        # Page 2: ROC & PR
        pdf.savefig(fig_curves)
        plt.close(fig_curves)

        # Page 3: Confusion matrices
        pdf.savefig(fig_cms)
        plt.close(fig_cms)

        # Page 4: Coefficients
        pdf.savefig(fig_coef)
        plt.close(fig_coef)

        # Page 5: CV best params
        fig5 = plt.figure(figsize=(8.27, 11.69))
        fig5.text(0.5, 0.95, '交叉验证最优参数', ha='center', va='center', fontsize=14, weight='bold')
        fig5.text(0.05, 0.90, cv_text[:3800] + ('...' if len(cv_text) > 3800 else ''), fontsize=10, family='monospace')
        fig5.tight_layout()
        pdf.savefig(fig5)
        plt.close(fig5)

        # Page 6: Classification reports
        if cr_f1 or cr_rec:
            fig6 = plt.figure(figsize=(8.27, 11.69))
            fig6.text(0.5, 0.95, '分类报告（测试集）', ha='center', va='center', fontsize=14, weight='bold')
            fig6.text(0.05, 0.90, 'F1最优阈值：', fontsize=12, weight='bold')
            fig6.text(0.05, 0.88, cr_f1[:1700] + ('...' if len(cr_f1) > 1700 else ''), fontsize=9, family='monospace')
            fig6.text(0.05, 0.55, '召回优先阈值：', fontsize=12, weight='bold')
            fig6.text(0.05, 0.53, cr_rec[:1700] + ('...' if len(cr_rec) > 1700 else ''), fontsize=9, family='monospace')
            fig6.tight_layout()
            pdf.savefig(fig6)
            plt.close(fig6)

    print(f"PDF 已生成：{args.output_pdf}")
    print(f"汇总图已生成：{args.output_fig_png}")


if __name__ == '__main__':
    main()