---
title: "基于 CGSS 数据的生育意愿预测与多分类模型对比分析"
collection: portfolio
type: "Machine Learning"
permalink: /portfolio/fertility-prediction
date: 2026-01-17
excerpt: "基于中国综合社会调查数据，构建逻辑回归、随机森林、支持向量机多分类模型预测个体生育意愿，并进行模型性能对比分析"
header:
  teaser: /images/portfolio/fertility/fertility_distribution.png
tags:
  - 机器学习
  - 多分类
  - 数据挖掘
  - CGSS数据
  - 模型对比
tech_stack:
  - name: Python
  - name: Scikit-learn
  - name: Pandas
  - name: Matplotlib
---

## 项目背景

本项目基于中国综合社会调查（CGSS）数据，对15-49岁育龄人群的生育意愿进行多分类预测研究。生育意愿受经济因素、社会政策、个人特征等多维度因素影响，具有复杂的社会学意义。

研究目标包括：
1. 数据清洗与特征工程，处理缺失值与异常值
2. 描述性统计分析与可视化探索
3. 构建三种机器学习模型（逻辑回归、随机森林、SVM）进行多分类预测
4. 对比模型性能，找出最优预测方案

---

## 数据预处理

数据来源于 CGSS 问卷调查，包含年龄、性别、教育程度、健康状况、家庭收入、养老保障等特征。预处理流程如下：

### 1. 变量处理
- **年龄计算**：根据出生年份计算实际年龄，筛选15-49岁育龄人群
- **生育意愿编码**：剔除无效值（"无所谓"、"不知道"、"拒绝回答"），保留有效值0-3（表示不同的生育意愿等级）
- **分类变量编码**：使用 LabelEncoder 对性别、教育、健康状况等分类变量进行编码

```python
# 处理age变量
data['age'] = 2021 - data['age_year']
data = data[(data['age'] >= 15) & (data['age'] <= 49)]

# 清理fertility变量
invalid_fertility = ['无所谓', '不知道', '拒绝回答', 62]
data = data[~data['fertility'].isin(invalid_fertility)]
data['fertility'] = pd.to_numeric(data['fertility'], errors='coerce')
data = data[data['fertility'].isin([0, 1, 2, 3])]

# 分类变量编码
cat_cols = ['gender', 'education', 'health', 'pension', 'family_status', 'marital']
le_dict = {}
for col in cat_cols:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    le_dict[col] = le
```

### 2. 特征选择与数据集划分
选择9个关键特征进行建模，包括养老保障、性别、年龄、教育程度、健康状况、家庭收入、家庭状态、财产状况、婚姻状况。

```python
# 筛选建模变量
model_cols = ['pension', 'gender', 'age', 'education', 'health', 'lnfamily_income', 
              'family_status', 'property', 'marital']
X = data[model_cols]
y = data['fertility']

# 划分训练集和测试集（7:3）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 数值变量标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

---

## 描述性统计与可视化

### 数值变量分布

![数值变量分布](/images/numerical_distribution.png)

从数值变量的描述性统计可以看出：
- **年龄**：样本平均年龄34.65岁，标准差9.00，覆盖18-49岁育龄范围
- **家庭收入（对数）**：均值12.31，反映样本家庭经济状况存在一定差异
- **财产状况**：均值1.26，表明样本整体财产水平较低，但存在高值异常点（最大值30）

### 分类变量分布

![分类变量分布](/images/categorical_distribution.png)

分类变量的频次分析揭示了样本的社会人口学特征：
- **性别分布**：女性占比56.83%，略高于男性
- **教育程度**：样本教育水平呈现多样化分布
- **养老保障**：大部分样本享有某种形式的养老保障

### 生育意愿目标变量分布

![生育意愿分布](/images/fertility_distribution.png)

生育意愿作为目标变量，其分布呈现：
- 生育意愿=2 的样本占比最高（62.77%），表明大部分受访者倾向于中等生育意愿
- 存在明显的类别不平衡问题，需要考虑在建模中进行相应处理

---

## 模型构建与评估

本研究构建了三种多分类模型进行对比：逻辑回归、随机森林、支持向量机（SVM）。

### 模型评估框架

定义统一的评估函数，计算准确率、召回率、ROC-AUC，并生成混淆矩阵和ROC曲线。针对SVM的多分类ROC-AUC计算问题，采用softmax归一化将决策分数转换为概率分布。

```python
def evaluate_model(model, X_train, X_test, y_train, y_test, model_name, is_svm=False):
    # 训练模型
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred, average='macro')
    
    # SVM多分类ROC-AUC计算
    if is_svm:
        y_score = model.decision_function(X_test)
        y_proba = np.exp(y_score) / np.sum(np.exp(y_score), axis=1, keepdims=True)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    else:
        y_proba = model.predict_proba(X_test)
        roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr')
    
    return accuracy, recall, roc_auc
```

### 模型1：逻辑回归

![逻辑回归混淆矩阵与ROC曲线](/images/logistic_regression_results.png)

逻辑回归作为基准模型，表现：
- **准确率**：63.25%
- **召回率**：25.00%
- **ROC-AUC**：0.7043

混淆矩阵显示，模型在预测生育意愿=2 的类别上表现较好，但对其他类别的识别能力有限。

### 模型2：随机森林

![随机森林混淆矩阵与ROC曲线](/images/random_forest_results.png)

随机森林模型表现：
- **准确率**：54.64%
- **召回率**：25.68%
- **ROC-AUC**：0.6536

随机森林在准确率和ROC-AUC上均低于逻辑回归，可能由于特征重要性分布不均或类别不平衡影响。

### 模型3：支持向量机（SVM）

![支持向量机混淆矩阵与ROC曲线](/images/svm_results.png)

支持向量机模型表现：
- **准确率**：63.13%
- **召回率**：25.96%
- **ROC-AUC**：0.6608

SVM 在准确率上接近逻辑回归，但ROC-AUC略低，表明在类别区分能力上仍有提升空间。

---

## 模型性能对比与结论

![模型性能对比](/images/model_comparison.png)

### 综合对比

| 模型 | 准确率 | 召回率 | ROC-AUC |
|------|--------|--------|---------|
| 逻辑回归 | 63.25% | 25.00% | 0.7043 |
| 随机森林 | 54.64% | 25.68% | 0.6536 |
| 支持向量机 | 63.13% | 25.96% | 0.6608 |

### 核心发现

1. **逻辑回归表现最优**：在准确率和ROC-AUC指标上均领先，表明线性模型在该数据集上具有较好的泛化能力
2. **召回率普遍偏低**：三种模型的召回率均在25%-26%之间，反映模型对少数类别的识别能力不足，建议后续尝试过采样、类别权重调整等策略
3. **多分类挑战**：生育意愿作为四分类问题，类别不平衡（类别2占62.77%）显著影响模型性能，可考虑合并类别或采用分层采样
4. **特征工程空间**：可进一步探索特征交互项、多项式特征或使用特征选择方法提升模型表现

---

## 技术亮点

- **完整数据预处理流程**：处理缺失值、异常值、类别编码、数据标准化
- **多分类模型对比**：系统评估三种主流机器学习算法在生育意愿预测任务上的表现
- **SVM多分类ROC-AUC修复**：采用softmax归一化解决SVM在多分类场景下的ROC-AUC计算问题
- **可视化分析**：生成描述性统计图表、混淆矩阵、ROC曲线等多维度可视化

---

## 项目价值

本项目为理解中国育龄人群生育意愿的影响因素提供了数据驱动的分析框架，对人口政策制定、社会福利资源配置具有参考意义。通过机器学习方法的探索，也为后续更复杂的社会学量化研究奠定了技术基础。
