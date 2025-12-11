"""
Author: Joseph Olumeyan
Date: October 2025
Description: Helper functions and classes for data analysis and visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score



def load_data(file_path):
    return pd.read_csv(file_path)

def add_equality_line(x1,y1):
    x = [x1, y1]
    y = [x1, y1]
    plt.plot(x, y, color='black', alpha=0.5, linestyle='--')

def square_the_plot(x, y):
    if x.min() < y.min():
        LL = x.min() - (0.1 * x.min())
    else:
        LL = y.min() - (0.1 * y.min())
    if x.max() > y.max():
        HL = x.max() + (0.1 * x.max())
    else:
        HL = y.max() + (0.1 * y.max())
    plt.xlim(LL, HL)
    plt.ylim(LL, HL)
    ax = plt.gca()
    ax.set_aspect(1)
    return LL, HL


def add_labels(df, x_col, y_col, label_col):
    for i, row in df.iterrows():
        x = row[x_col]
        y = row[y_col]
        gap = "  "
        label = gap + str(row[label_col])
        plt.text(x, y, label, va='center', ha='left', fontsize=4.5)
        if i % 1000 == 0:
            print(f"Added labels for {i} points...")
          
# def Explore_plot(df, x_col, y_col,s=40):
#     sns.scatterplot(data=df, x=x_col, y=y_col, s=s, alpha=0.5)
#     LL, HL = square_the_plot(df[x_col], df[y_col])
#     add_equality_line(LL, HL)
#     add_labels(df, x_col, y_col, "currency")
#     plt.show()


def scatter_plot(x_col, y_col, df,xlabel,ylabel,title, figsize=(4, 3),s=40,addLabels=False,):
    plt.figure(figsize=figsize) 
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    
    if addLabels:  
        sns.scatterplot(data=df, x=x_col, y=y_col, s=s)
        print("Adding labels to scatter plot...")
        add_labels(df, x_col, y_col, "StockCode")
        plt.show()
    else:
        print("No labels added to scatter plot. generating scatter plot...")
        sns.scatterplot(data=df, x=x_col, y=y_col, s=s)
        plt.show()

def box_plot(df, x_col, y_col, palette="muted",plt_title=None):
    sns.boxplot(data=df, x=x_col, y=y_col, palette=palette)
    if plt_title is None:
        plt.title(f"Boxplot of {y_col} by {x_col}")
    else:
        plt.title(plt_title)
    plt.show()

def hist_plot(df, col, bins=20, kde=True,plt_title=None):
    sns.histplot(df[col], bins=bins, kde=kde)
    if plt_title is None:
        plt.title(f"Distribution of {col}")
    else:
        plt.title(plt_title)
    plt.show()

def bar_plot(df, x_col, y_col, palette="muted"):
    sns.barplot(data=df, x=x_col, y=y_col, palette=palette)
    plt.title(f"Average {y_col} by {x_col}")
    plt.show()

def heatmap(df):
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()

def add_lag_features(df, group_col, date_col, lag_cols, n_lags):
    df=df.sort_values([group_col, date_col])
    for col in lag_cols:
        for lag in range(1, n_lags + 1):
            df[f"{col}_t-{lag}"] = df.groupby(group_col)[col].shift(lag).fillna(0)
    return df

def add_average_features(df, group_col, avg_cols):
    avg=df.groupby(group_col)[avg_cols].mean().reset_index()
    avg.rename(columns={col: f"{col}_avg" for col in avg_cols}, inplace=True)
    df = df.merge(avg, on=group_col, how="left")
    return df

class LinearModel:
    def __init__(self, model_name=""):
        self.model_name = model_name
    
    def fit(self, x, y):
        x = pd.DataFrame(x)
        linear_model = LinearRegression().fit(x, y)
        y_pred = linear_model.predict(x)
        self.slope = linear_model.coef_[0]
        self.intercept = linear_model.intercept_
        self.rsquared = r2_score(y, y_pred)
        
    def predict(self, x):
        return self.slope * x + self.intercept

    def plot_model(self, x_min, x_max, color="black"):
        y_min = self.predict(x_min)
        y_max = self.predict(x_max)
        plt.plot([x_min, x_max], [y_min, y_max], color=color)
        
    def print_model_info(self):
        m = self.slope
        b = self.intercept
        rsquared = self.rsquared
        model_name = self.model_name
        print(f'LinearModel({model_name}):')
        print(f'Parameters: slope = {m:.2f}, intercept = {b:.2f}')
        print(f'Equation: y = {m:.2f}x + {b:.2f}')
        print(f'Goodness of Fit (R²): {rsquared:.3f}')

class QuadraticModel:
    def fit(self, x, y):
        x = pd.DataFrame(x)
        quadratic = PolynomialFeatures(degree=2)
        quad_features = quadratic.fit_transform(x)
        quad_model = LinearRegression().fit(quad_features, y)
        y_pred = quad_model.predict(quad_features)
        self.a = quad_model.coef_[2]
        self.b = quad_model.coef_[1]
        self.c = quad_model.intercept_
        self.rsquared = r2_score(y, y_pred)
        
    def predict(self, x):
        return self.a*x**2 + self.b*x + self.c
       
    def plot_model(self, xmin, xmax):
        xvals = range(xmin, xmax+1)
        yvals = [self.predict(x) for x in xvals]
        plt.plot(xvals, yvals, color='black')
        
    def print_model_info(self):
        a = self.a
        b = self.b
        c = self.c
        rsquared = self.rsquared
        print('QuadraticModel')
        print(f'Parameters: a = {a:.2f}, b = {b:.2f}, c = {c:.2f}')
        print(f'Equation: y = {self.a:.2f}x² + {self.b:.2f}x + {self.c:.2f}')
        print(f'Goodness of Fit (R²): {rsquared:.3f}')
