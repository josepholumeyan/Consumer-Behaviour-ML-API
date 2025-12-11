from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV,KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import xgboost as xgb

import math
import matplotlib.pyplot as plt
import pandas as pd


class model_training:
    def __init__(self,model_name=""):
        self.model_name = model_name
        self.model= None

    def split(self,x,y,testsize=0.2,randomstate=42):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
        x, y, test_size=testsize, random_state=randomstate)
        return self

    def fit_linear(self):
        if self.model is None:
            try:
                self.model=LinearRegression()
                self.model.fit(self.X_train,self.y_train)
                return self
            except AttributeError:
                print("Split the data before fitting")
        else:
            print("you've already fitted this object to another model")

    def  fit_quad(self):
        if self.model is None:  
            try:
                self.quadratic = PolynomialFeatures(degree=2)
                quad_features = self.quadratic.fit_transform(self.X_train)
                self.model=LinearRegression()
                self.model.fit(quad_features, self.y_train)
                return self
            except AttributeError:
                print("Split the data before fitting")
        else:
            print("you've already fitted this object to another model")

    def fit_rf(self,ran_state=42):
        if self.model is None:
            if hasattr(self, "X_train") and hasattr(self, "y_train"):
                rf=RandomForestRegressor(random_state=ran_state)
                param_grid = {
                    'n_estimators': [120,150,200],
                    'max_depth': [None, 40,60,80],
                    'min_samples_split': [2, 5],
                    'min_samples_leaf': [1, 2],
                    'max_features': ['sqrt']
                    # where to start with grid search
                    # 'n_estimators': [80, 120],
                    # 'max_depth': [None, 10, 20],
                    # 'min_samples_split': [2, 5, 10],
                    # 'min_samples_leaf': [1, 2, 4],
                    # 'max_features': ['log2', 'sqrt']
                    }
                kf=KFold(n_splits=5, shuffle=True, random_state=ran_state)
                grid_search = GridSearchCV(estimator=rf, 
                                           param_grid=param_grid, 
                                           cv=kf, 
                                           n_jobs=-1, 
                                           scoring='r2')
                print("Fitting Random Forest with Grid Search...")
                grid_search.fit(self.X_train, self.y_train)
                print("Grid Search completed.")
                results=pd.DataFrame(grid_search.cv_results_)
                results.sort_values('mean_test_score', ascending=False, inplace=True)
                results.to_csv(f"rf_grid_search_results_{self.model_name}.csv", index=False)

                self.model = grid_search.best_estimator_

                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation R²:", round(grid_search.best_score_,3))
                print(results[['params', 'mean_test_score', 'std_test_score', 'rank_test_score']])
                
                return self
            else:
                print("Split the data before fitting")
        else:
            print("you've already fitted this object to another model")
    
    def fit_xgboost(self,ran_state=42):
        if self.model is None:
            if hasattr(self, "X_train") and hasattr(self, "y_train"):
                xgb_model = xgb.XGBRegressor(random_state=ran_state,
                                            objective='reg:squarederror',
                                            tree_method='hist',
                                            eval_metric='rmse'
                                             )
                
                param_grid = {
                    'n_estimators': [100, 150, 200],
                    'max_depth': [3, 6, 9],
                    'learning_rate': [0.01, 0.1, 0.2],
                    'subsample': [0.7, 1.0],
                    'colsample_bytree': [0.7, 1.0]
                }

                print("Fitting XGBoost with Grid Search...")
                grid_search = GridSearchCV(estimator=xgb_model, 
                                           param_grid=param_grid, 
                                           cv=3, 
                                           n_jobs=-1, 
                                           scoring='r2',
                                           verbose=1)
                
                grid_search.fit(self.X_train, self.y_train,eval_set=[(self.X_test, self.y_test)],verbose=False)
                print("Grid Search completed.")
                self.model = grid_search.best_estimator_
                print(f"Best parameters: {grid_search.best_params_}")
                print(f"Best cross-validation R²:", round(grid_search.best_score_,3))

                results=pd.DataFrame(grid_search.cv_results_)
                results.sort_values('mean_test_score', ascending=False, inplace=True)   
                results.to_csv(f"xgb_grid_search_results_{self.model_name}.csv", index=False)
                return self
            else:
                print("Split the data before fitting")
        else:
            print("you've already fitted this object to another model")

    def pred(self,x=None):
        if self.model is None:
            print("No model to make a prediction. Try fitting a model first.")
        elif x is not None:
            if isinstance(self.model, LinearRegression) and hasattr(self, "quadratic"):
                x_transformed = self.quadratic.transform(x)
                return self.model.predict(x_transformed)
            else:
                return self.model.predict(x)
        else:
            if isinstance(self.model, LinearRegression) and hasattr(self, "quadratic"):
                X_test_transformed = self.quadratic.transform(self.X_test)
                return self.model.predict(X_test_transformed)
            else:
                return self.model.predict(self.X_test)

    def eval_model(self):
        if self.model is None:
            print("No model fitted yet.")
            return
        y_pred = self.pred()
        mae = mean_absolute_error(self.y_test, y_pred)
        rmse = math.sqrt(mean_squared_error(self.y_test, y_pred))
        r2 = r2_score(self.y_test, y_pred)
        print(f"\n{self.model_name} Performance:")
        print(f"MAE:  {mae:.2f}")
        print(f"RMSE: {rmse:.2f}")
        print(f"R²:   {r2:.3f}")
        return {"Model": self, "MAE": mae, "RMSE": rmse, "R2": r2}
    
    def check_relia(self,preprocessor,numeric_features=None):
        predictions = self.model.predict(self.X_test)
        print("R²:", r2_score(self.y_test, predictions))

        if numeric_features is None:
            numeric_features = preprocessor.transformers_[0][2]

        cat_features =[]

        for name, transformer, features in preprocessor.transformers_:
            if name == "cat":
                ohe = transformer.named_steps['onehot']
                cat_features = list(ohe.get_feature_names_out(features))
                break

        all_features = list(numeric_features) + cat_features
        importances = self.model.feature_importances_

        if len(all_features) != len(importances):
            print("Warning: Number of features does not match number of importances.")
            return

        feat_imp = pd.DataFrame({
            'Feature': all_features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)


        plt.barh(feat_imp['Feature'], feat_imp['Importance'])
        plt.gca().invert_yaxis()
        plt.title("Feature Importance (Random Forest)")
        plt.xlabel("Importance Score")
        plt.tight_layout()
        feat_imp.to_csv(f"Feature_importance_{self.model_name}.csv", index=False)
        print(f"Feature importance saved to 'Feature_importance_{self.model_name}.csv'")
        plt.show()


class Pipelines:
    def __init__(self, model, preprocessor):
        self.model = model
        self.preprocessor = preprocessor
        self.numeric_columns =[]
        self.categorical_columns =[]
        for name, transformer, features in preprocessor.transformers_:
            if name == "num":
                self.numeric_columns=list(features)
            elif name == "cat":
                self.categorical_columns=list(features)
        
        self.expected_columns = self.numeric_columns + self.categorical_columns
    
    def predict(self, X):
        X_processed = self.preprocessor.transform(X)
        return self.model.pred(X_processed)
    

def build_preprocessor(df, target):
    if df.empty:
        raise ValueError("The input dataframe is empty.")
    elif target not in df.columns:
        raise ValueError(f"The target column '{target}' is not in the dataframe.")
    
    categorical_features = df.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if target in numerical_features:
        numerical_features.remove(target)
    if target in categorical_features:
        categorical_features.remove(target)

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    preprocessor.fit_transform(df.drop(columns=[target]))

    return preprocessor