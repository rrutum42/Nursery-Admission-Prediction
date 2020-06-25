# Nursery-Admission-Prediction
Nursery Admission Prediction uses Machine Learning classification algorithms to categorize whether the candidate is priority, recommended or not recommended to be admitted.  
#### Classification algorithms used are:  
 - Support Vector Machines  
 - Random Forest  
 - Logistic Regression  
 - XGBoost  
   
The dataset is [Nursery Data Set]( https://archive.ics.uci.edu/ml/datasets/nursery ).  
The accuracy of prediction is displayed for each algorithm.  
**Label Encoder** and **`get_dummies()`** is used for converting categorical data into numeric data,  
**Extra Tree Classifier**,**SelectKBest** and **Chi-Square**  for Feature Selection

### BEFORE RUNNING THE CODE  
- Paste the address of the directory where your Nursery.csv file is saved in `dataset = pd.read_csv(r'Nursery.csv')`.    
- Install XGBoost via the command prompt.   
`conda install -c anaconda py-xgboost`(for Anaconda prompt)
  
