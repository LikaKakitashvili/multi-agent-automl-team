# Multi-Agent AutoML — Final Report

## Pipeline Overview

| Stage | Artifact | Shape | Structured Report |
|-------|----------|-------|-------------------|
| Raw | `raw_data.csv` | 891 rows × 12 columns | — |
| Cleaned | `clean_data.csv` | 891 rows × 11 columns | `agent1_structured_report.json` |
| Engineered | `engineered_data.csv` | 891 rows × 6 columns | `agent2_structured_report.json` |
| Model | XGBoost | — | `agent3_structured_report.json` |

## Agent 1 — Data Cleaner

The dataset has been cleaned by imputing missing values, handling outliers, and dropping unnecessary columns. The 'Cabin' column was dropped due to excessive missing values, while 'Age' was imputed with the median, and 'Embarked' with the mode. Outliers in 'Age' and 'Fare' were capped based on the IQR method. Column types were cast to more appropriate dtypes, improving data quality for analysis and modeling.

**Actions taken:**
- impute on `Age`: Filled nulls with median to address missing data.
- cap on `Age`: Handled outliers using IQR method to ensure data consistency.
- cap on `Fare`: Handled outliers using IQR method to remove extreme values.
- drop on `Cabin`: Dropped due to having >90% missing values.
- impute on `Embarked`: Filled nulls with mode to address missing data.
- cast on `Sex`: Converted 'Sex' to categorical for better analysis.
- cast on `Embarked`: Converted 'Embarked' to categorical for better analysis.
- cast on `Ticket`: Converted 'Ticket' to string as it represents categorical data.

**Notes for next agent:** Continue feature engineering and prepare for modeling. Pay attention to the categorical nature of 'Sex' and 'Embarked' for encoding.

**Structured JSON:**

```json
{
  "agent": "Agent 1 \u2014 The Data Cleaner (The Auditor)",
  "summary": "The dataset has been cleaned by imputing missing values, handling outliers, and dropping unnecessary columns. The 'Cabin' column was dropped due to excessive missing values, while 'Age' was imputed with the median, and 'Embarked' with the mode. Outliers in 'Age' and 'Fare' were capped based on the IQR method. Column types were cast to more appropriate dtypes, improving data quality for analysis and modeling.",
  "actions": [
    {
      "action": "impute",
      "target": "Age",
      "reason": "Filled nulls with median to address missing data."
    },
    {
      "action": "cap",
      "target": "Age",
      "reason": "Handled outliers using IQR method to ensure data consistency."
    },
    {
      "action": "cap",
      "target": "Fare",
      "reason": "Handled outliers using IQR method to remove extreme values."
    },
    {
      "action": "drop",
      "target": "Cabin",
      "reason": "Dropped due to having >90% missing values."
    },
    {
      "action": "impute",
      "target": "Embarked",
      "reason": "Filled nulls with mode to address missing data."
    },
    {
      "action": "cast",
      "target": "Sex",
      "reason": "Converted 'Sex' to categorical for better analysis."
    },
    {
      "action": "cast",
      "target": "Embarked",
      "reason": "Converted 'Embarked' to categorical for better analysis."
    },
    {
      "action": "cast",
      "target": "Ticket",
      "reason": "Converted 'Ticket' to string as it represents categorical data."
    }
  ],
  "artifacts": {
    "cleaned_file_path": "/Users/lika/Desktop/KIU/Semester_6/LLM/Homework2/output/clean_data.csv",
    "shape": [
      891,
      11
    ]
  },
  "notes_for_next_agent": "Continue feature engineering and prepare for modeling. Pay attention to the categorical nature of 'Sex' and 'Embarked' for encoding.",
  "metrics": {}
}
```

## Agent 2 — Feature Engineer

I engineered new features from the cleaned dataset to enhance model performance. A new feature, 'Fare_Per_Person', was created to analyze fare contribution per individual in a family group. Categorical features 'Sex' and 'Embarked' were one-hot encoded to convert them into numerical format suitable for modeling. Additionally, the 'Pclass' feature was label encoded due to its low cardinality. After performing correlation analysis, I selected the top 5 predictive features related to the survival rate: 'Fare_Per_Person', 'Sex_male', 'Ticket', 'Fare', and 'SibSp'. This refinement ensures that redundant features are removed while retaining those with the highest predictive power.

**Actions taken:**
- create_feature on `Fare_Per_Person`: Analyzed fare contribution per individual.
- encode on `Sex`: One-hot encoding for categorical conversion.
- encode on `Embarked`: One-hot encoding for categorical conversion.
- encode on `Pclass`: Label encoding for low cardinality.
- select_features on `Top 5 Features`: Identified most predictive features to reduce redundancy.

**Notes for next agent:** When proceeding with the modeling phase, focus on the selected top features. Evaluate the impact of the newly engineered feature 'Fare_Per_Person' and consider further interactions or polynomial features if needed.

**Structured JSON:**

```json
{
  "agent": "Agent 2 \u2014 The Feature Engineer (The Architect)",
  "summary": "I engineered new features from the cleaned dataset to enhance model performance. A new feature, 'Fare_Per_Person', was created to analyze fare contribution per individual in a family group. Categorical features 'Sex' and 'Embarked' were one-hot encoded to convert them into numerical format suitable for modeling. Additionally, the 'Pclass' feature was label encoded due to its low cardinality. After performing correlation analysis, I selected the top 5 predictive features related to the survival rate: 'Fare_Per_Person', 'Sex_male', 'Ticket', 'Fare', and 'SibSp'. This refinement ensures that redundant features are removed while retaining those with the highest predictive power.",
  "actions": [
    {
      "action": "create_feature",
      "target": "Fare_Per_Person",
      "reason": "Analyzed fare contribution per individual."
    },
    {
      "action": "encode",
      "target": "Sex",
      "reason": "One-hot encoding for categorical conversion."
    },
    {
      "action": "encode",
      "target": "Embarked",
      "reason": "One-hot encoding for categorical conversion."
    },
    {
      "action": "encode",
      "target": "Pclass",
      "reason": "Label encoding for low cardinality."
    },
    {
      "action": "select_features",
      "target": "Top 5 Features",
      "reason": "Identified most predictive features to reduce redundancy."
    }
  ],
  "artifacts": {
    "engineered_file_path": "/Users/lika/Desktop/KIU/Semester_6/LLM/Homework2/output/engineered_data.csv",
    "shape": [
      891,
      6
    ]
  },
  "notes_for_next_agent": "When proceeding with the modeling phase, focus on the selected top features. Evaluate the impact of the newly engineered feature 'Fare_Per_Person' and consider further interactions or polynomial features if needed.",
  "metrics": {}
}
```

## Agent 3 — Model Trainer

Trained an XGBoost classifier on the provided dataset for predicting survival. Several iterations were run to optimize the model's performance based on accuracy, recall, and F1 score metrics.

**Actions taken:**
- train on `XGBoost`: Initial run to establish baseline metrics.
- tune on `hyperparameters`: Adjusted learning rate, max depth and number of estimators based on performance.
- retrain on `XGBoost`: Final adjustments aimed to improve model performance further.

**Notes for next agent:** Explore additional feature engineering or tuning for improvement. Consider ensemble methods or alternative algorithms if needed.

**Metrics:** accuracy=0.7948, recall=0.6847, f1=0.7343

**Structured JSON:**

```json
{
  "agent": "Agent 3 \u2014 The Model Trainer (The Coder)",
  "summary": "Trained an XGBoost classifier on the provided dataset for predicting survival. Several iterations were run to optimize the model's performance based on accuracy, recall, and F1 score metrics.",
  "actions": [
    {
      "action": "train",
      "target": "XGBoost",
      "reason": "Initial run to establish baseline metrics."
    },
    {
      "action": "tune",
      "target": "hyperparameters",
      "reason": "Adjusted learning rate, max depth and number of estimators based on performance."
    },
    {
      "action": "retrain",
      "target": "XGBoost",
      "reason": "Final adjustments aimed to improve model performance further."
    }
  ],
  "artifacts": {
    "trained_model": "XGBoostClassifier",
    "final_metrics": {
      "accuracy": 0.7947761194029851,
      "recall": 0.6846846846846847,
      "f1": 0.7342995169082126
    }
  },
  "notes_for_next_agent": "Explore additional feature engineering or tuning for improvement. Consider ensemble methods or alternative algorithms if needed.",
  "metrics": {
    "accuracy": 0.7947761194029851,
    "recall": 0.6846846846846847,
    "f1": 0.7342995169082126
  }
}
```

### Final Model Metrics

- **accuracy**: 0.7948
- **recall**: 0.6847
- **f1**: 0.7343

### Training Iterations

The model trainer executed **5** script(s) during the feedback loop.

## Execution Logs

Full agent conversation log: `agent_logs.txt`

---
*Generated by the Multi-Agent AutoML Team pipeline.*
