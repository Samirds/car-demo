# ================================== IMPORT ============================================>
import numpy as np
import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesRegressor  # FOR FEATURE IMPORTANCE

import pickle
from sklearn.ensemble import RandomForestRegressor  # FOR MODEL CREATION
from sklearn.model_selection import RandomizedSearchCV  # FOR PARAMETER TUNING
from sklearn import metrics

# ================================== DATASET ========================================>

df = pd.read_csv("car data .csv")

# =================================== MAKING DATASET ==========================================>

final_dataset = df[
    ['Year', 'Selling_Price', 'Present_Price', 'Kms_Driven', 'Fuel_Type', 'Seller_Type', 'Transmission', 'Owner']]
final_dataset['Current Year'] = 2020
final_dataset['no_year'] = final_dataset['Current Year'] - final_dataset['Year']
final_dataset.drop(['Year'], axis=1, inplace=True)
final_dataset = pd.get_dummies(final_dataset, drop_first=True)
final_dataset = final_dataset.drop(['Current Year'], axis=1)

# ============================= VISUALIZATION ================================================>

# import seaborn as sns
# #get correlations of each features in dataset
# corrmat = df.corr()
# top_corr_features = corrmat.index
# plt.figure(figsize=(20,20))
# #plot heat map
# g = sns.heatmap(df[top_corr_features].corr(),annot=True,cmap="RdYlGn")


# ====================================   ================================================>

X = final_dataset.iloc[:, 1:]
y = final_dataset.iloc[:, 0]

# ================================== FEATURE IMPORTANCE ===================================>

model = ExtraTreesRegressor()
model.fit(X, y)

# ========================= plot graph of feature importance for better visualization =========>

# feat_importances = pd.Series(model.feature_importances_, index=X.columns)
# feat_importances.nlargest(5).plot(kind='barh')
# plt.show()

# ====================================================>

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# ================================ MODEL ==========================================================>

regressor = RandomForestRegressor()

# ============================ HYPER PARAMETER TUNING ================================ ===============>

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
# max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10, 15, 100]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 5, 10]

# ===================================# Create the random grid ================================>

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

# ======================================               ===============================================>

# Use the random grid to search for best hyper parameters
# First create the base model to tune
rf = RandomForestRegressor()

# ========================================            ===========================================================>

# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations
rf_random = RandomizedSearchCV(estimator=rf,
                               param_distributions=random_grid,
                               scoring='neg_mean_squared_error',
                               n_iter=10, cv=5, verbose=2,
                               random_state=42, n_jobs=1)

rf_random.fit(X_train,y_train)

# =================================  BEST PARAMETER & SCORE ===============================>

best = rf_random.best_params_, rf_random.best_score_
# print(best)

# ==================================== Predictions ===========================================>

predictions=rf_random.predict(X_test)

# ==========================================>

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

# =========================================== Model Save ===================================================>

file = open('car_model_save.pkl', 'wb')
# dump information to that file
pickle.dump(rf_random, file)