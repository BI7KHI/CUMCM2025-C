# T4 女胎样本判定与异常检测建模思路总览

本文档概述 `t4_female_fetus_analysis.py` 的完整建模流程与设计取舍，覆盖数据处理、样本筛选、特征工程、分组切分、模型与超参搜索、阈值策略、评估与可解释性、以及结果持久化。

## 1. 数据读取与统一
- 数据源：`Source_DATA/dataA.csv`、`Source_DATA/dataB.csv` 合并。
- 列名统一：将关键字段映射为统一命名（Z13/Z18/Z21、ZX/ZY、Y浓度/X浓度、GC相关、比对/重复/过滤比例、BMI、孕周文本、孕妇代码、AB标签等）。
- 孕周解析：将“13w+6”一类文本解析为 `孕周_float` 与 `孕周_天数`。
- 缺失处理：关键特征缺失列补齐为 NaN，保证后续流水线健壮。

## 2. 女胎样本筛选（先验样本空间约束）
- 以“Y浓度”为主，若有效样本 ≥ 20：使用两成分 GaussianMixture 做双峰聚类，低均值簇视为“女胎簇”；缺失 Y 浓度样本默认并入女胎（符合任务背景）。
- 若样本不足：采用经验分位数（如 P40）阈值策略。
- 输出：女胎子集与筛选统计信息；作为后续建模输入。

## 3. 特征工程与标签构造
- 标签 `y`：AB 列包含 T13/T18/T21 任一即视为阳性（异常）。
- 特征 `X`：
  - Z 相关：Z13、Z18、Z21、ZX；
  - GC 相关：GC_total、GC13、GC18、GC21；
  - 质量相关：原始读段数、比对比例、重复比例、过滤比例、唯一比对率（唯一比对读段 / 原始读段）；
  - 体征/孕期：BMI、孕周_float。
- 数值化与填补：全量特征转为数值，按列中位数填补缺失。

## 4. 分组切分避免泄漏
- 使用孕妇代码作为分组变量，`GroupShuffleSplit` 将数据划分为训练/测试集，确保同一孕妇的样本不跨集合，避免样本泄漏与过拟合。

## 5. 模型管道与分组交叉验证的超参搜索
- 管道：`StandardScaler` + `LogisticRegression(liblinear, class_weight='balanced', max_iter=200)`。
- 超参网格：`C ∈ {0.1, 1.0, 3.0, 10.0}`, `penalty ∈ {l1, l2}`。
- 交叉验证：`GroupKFold(5)` 按孕妇分组，`GridSearchCV(scoring='f1')` 选择最佳参数与管道；输出 `cv_best_params.json` 与 `cv_results.csv` 以便复盘与比较。
- 可选概率校准（命令行开关）：在训练集内再划一部分作校准集（`--calibrate --calib_size 0.2 --calib_method sigmoid|isotonic`），并保持孕妇分组不泄漏；最终将校准后的模型用于阈值选择与测试评估。

## 6. 概率阈值策略（训练集上定阈）
- 在训练集预测概率上绘制 PR 曲线，提供两种阈值：
  - 方案 A（F1 优先）：选择 F1 最大对应阈值 `best_threshold_f1`；
  - 方案 B（召回优先）：在 `recall ≥ target_recall` 的候选阈值中选 F1 最优，得到 `recall_priority_threshold`；
- 这两套阈值均用于测试集评估与混淆矩阵对比，满足“综合平衡”与“不漏检优先”的不同业务诉求。

## 7. 评估与可解释性
- 可视化：
  - 训练集 PR 曲线（标注阈值点）；
  - 测试集 ROC 曲线；
  - 测试集的两种阈值下混淆矩阵图（F1 优先 / 召回优先）。
- 指标：precision / recall / F1 / ROC-AUC 与 classification report 等。
- 可解释性：
  - 训练浅层决策树（如 `max_depth=3`）并导出规则文本，辅助理解主要变量在决策边界中的作用。

## 8. 结果与产物持久化
- 指标：
  - `results/T4/metrics_f1.json`
  - `results/T4/metrics_recall_priority.json`
- 预测明细：`results/T4/test_predictions.csv`
- 模型与阈值：
  - `results/T4/model.joblib`
  - `results/T4/thresholds.json`（包含 `best_threshold_f1`, `recall_priority_threshold`, `target_recall`）
- 超参搜索：
  - `results/T4/cv_best_params.json`
  - `results/T4/cv_results.csv`
- 图形文件：训练集 PR、测试集 ROC、两种阈值的混淆矩阵（保存在 `results/T4/`）。
- 中文显示：脚本设置了 Matplotlib 中文字体（SimHei、Microsoft YaHei、DejaVu Sans、Arial Unicode MS）并处理了负号显示。

## 9. 运行示例
- 默认运行：
  ```powershell
  python .\scripts\T4\v1.0\t4_female_fetus_analysis.py --target_recall 0.90 --test_size 0.2 --seed 42
  ```
- 启用概率校准（Platt Sigmoid）：
  ```powershell
  python .\scripts\T4\v1.0\t4_female_fetus_analysis.py --target_recall 0.90 --test_size 0.2 --seed 42 --calibrate --calib_size 0.2 --calib_method sigmoid
  ```

## 10. 注意事项与后续优化
- 由于按孕妇代码分组，数据量较小时应关注折数与测试集占比的稳定性。
- 若类别极度不平衡，阈值策略可结合代价敏感（如 FN/FP 代价）进一步优化。
- 校准方法 `isotonic` 需更大样本量以避免过拟合；`sigmoid` 更稳健。
- 可进一步加入模型对比（如树、集成、线性可解释模型），并同样采用分组交叉验证与统一阈值策略。