# 🌟 Pulmonary_Detection: A Machine Learning Model for Patient Data Analysis 🏥

## 🚀 Overview

Welcome to the Pulmonary_Detection project, a collaboration between Moonshot AI and the Orthopedic Department of Wuhan Fourth Hospital. This project aims to analyze patient data to predict pulmonary complications using machine learning techniques. The data used in this project has been collected with informed consent from patients.

## 📁 Contents

- **Pulmonary_Detection.py**: The main Python script for data analysis and model training.
- **Data**: The original patient data file (not included in this repository due to privacy concerns).

## 🔧 Data Preprocessing
1. **Data Cleaning**: Removes the "住院号" column and maps the 'Group' column to numerical values.
   ```python
   df = df.drop("住院号", axis=1)
   df['Group'] = df['Group'].map({'No infection group': 0, 'Infection group': 1})
   ```

2. **Feature Selection**: Selects categorical and numerical features relevant to the prediction task.
   ```python
   cat_features = ['Male/female', 'Hypertension', ...]
   num_features = [col for col in df.columns if col not in cat_features and col != target_col]
   ```

3. **Correlation Analysis**: Visualizes the correlation between numerical features using a heatmap.
   ```python
   sns.heatmap(corr, xticklabels=corr.columns.values, yticklabels=corr.columns.values, cmap='GnBu')
   ```

4. **Data Visualization**: Plots various distributions and relationships within the data.
   ```python
   sns.countplot(x=df['Group'])
   sns.kdeplot(df['Platelet count (×109 ·L-1)'], color='blue', label='DV')
   ```

## 🛠 Feature Engineering

1. **Encoding**: Converts categorical features into numerical using one-hot encoding.
   ```python
   X_trans = pd.get_dummies(X, columns=cat_features)
   ```

2. **Feature Selection with LassoCV**: Uses LassoCV to select features that contribute most to the model.
   ```python
   lasso = LassoCV(cv=5)
   lasso.fit(X_trans, y)
   selected_features = X_trans.columns[lasso.coef_ != 0]
   ```

3. **Recursive Feature Elimination (RFECV)**: Further refines feature selection using RFECV.
   ```python
   rfecv = RFECV(estimator=clf_rf_4, step=1, cv=5, scoring='roc_auc')
   rfecv.fit(X_trans, y)
   ```

## 🧠 Model Training

1. **LightGBM**: Trains a LightGBM model using the selected features.
   ```python
   model = lgb.LGBMClassifier(**study.best_params)
   model.fit(train_x, train_y)
   ```

2. **Optuna for Hyperparameter Tuning**: Utilizes Optuna for optimizing model hyperparameters.
   ```python
   study = optuna.create_study(pruner=optuna.pruners.MedianPruner(n_warmup_steps=10), direction="maximize")
   study.optimize(objective, n_trials=50)
   ```

3. **Model Evaluation**: Evaluates the model using ROC AUC score.
   ```python
   y_pred = model.predict(xtest)
   metrics.roc_auc_score(ytest, y_pred)
   ```

## 📊 Model Interpretation

1. **SHAP Values**: Uses SHAP to interpret the model predictions and understand feature importance.
   ```python
   explainer = shap.TreeExplainer(model1)
   shap_values = explainer.shap_values(train_x)
   ```

2. **Visualizations**: Generates various plots to visualize the SHAP values and model decisions.
   ```python
   shap.summary_plot(shap_values, train_x)
   shap.plots.beeswarm(shap_values)
   ```

## 🔏 Data Privacy

- All patient data used in this project has been anonymized and used with informed consent.
- Data privacy and ethical considerations have been strictly adhered to.

## 🤝 Acknowledgments

- Special thanks to the Orthopedic Department of Wuhan Fourth Hospital for providing the data and their collaboration.
