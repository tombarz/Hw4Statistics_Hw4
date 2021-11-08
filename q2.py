# This is a sample Python script.
import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

np.set_printoptions(suppress=True)
path_age_and_time = "Age_And_Time.csv"
age_and_time = pd.read_csv(path_age_and_time)
path_countries = "countries.csv"
countries = pd.read_csv(path_countries)
def no_cut_b_LS(x, y):
    x_y = x*y
    b_LS = (sum(x_y)/sum(x*x))

    return b_LS
def R_squared_no_cut(x_b, y):
    return 1 - sum((y- x_b)**2)/sum(y*y)

def Q2_A(x,y):
    plt.figure(0)
    b = no_cut_b_LS(x, y)
    plt.scatter(x,y)
    x_b = x*b
    plt.plot(x,x_b)
    r_squared = R_squared_no_cut(x_b, y)
    print("r^2 is: "+ str(r_squared))
def log_t(x,y):
    plt.figure(1)
    y_log = np.log(y)
    plt.scatter(x, y_log)
    b = no_cut_b_LS(x, y_log)
    x_b = x * b
    plt.plot(x, x_b)
    r_squared_log = R_squared_no_cut(x_b, y_log)
    print("r^2 of log(y) is: " + str(r_squared_log))


def sqrt_t(x,y):
    plt.figure(2)
    y_sqrt = np.sqrt(y)
    plt.scatter(x, y_sqrt)
    b = no_cut_b_LS(x, y_sqrt)
    x_b = x * b
    plt.plot(x, x_b)
    r_squared_sqrt = R_squared_no_cut(x_b, y_sqrt)
    print("r^2 of sqrt(y) is: " + str(r_squared_sqrt))


def Q2_b():
    filtered_df = age_and_time[age_and_time['Time'] <= 30]
    x_filtered = filtered_df['Age'].to_numpy()
    y_filtered = filtered_df['Time'].to_numpy()
    log_t(x_filtered,y_filtered)
    sqrt_t(x_filtered,y_filtered)


def one_variable_model(x, y, i):
    plt.figure(i)
    plt.ticklabel_format(useOffset=False, style='plain')
    col1 = x / x
    X = np.array([col1, x]).T
    b = ((np.linalg.inv(X.T.dot(X))).dot(X.T.dot(y)))
    plt.scatter(x, y)
    plt.plot(x, (b[0] + b[1] * x))
    print("r squared is:" + str(R_squared((b[0] + b[1] * x), y)))


def two_variables_model(x1, x2, y, i):
    plt.figure(i)
    col1 = x1 / x1
    X = np.array([col1, x1, x2]).T
    b = (np.linalg.inv(X.T.dot(X)).dot(X.T.dot(y)))
    ax = plt.axes(projection='3d')
    ax.scatter3D(x1,x2,y)
    ax.plot_trisurf(x1,x2,b[0]+(b[1]*x1)+(b[2]*x2))
    print("r squared is:" + str(R_squared((b[0] + (b[1] * x1) + (b[2] * x2)), y)))


def R_squared(y_hat,y):
    y_mean_array = np.array([])
    y_mean = np.mean(y)
    for i in range(y.shape[0]):
        y_mean_array = np.append(y_mean_array,(y_mean))
    return 1 - sum((y-y_hat)**2)/sum((y-y_mean_array)**2)

def Q2_c():
    x1 = countries['income'].to_numpy()
    x2 = countries['education'].to_numpy()
    y = countries['life_expectancy'].to_numpy()
    #y=b0+b1x1
    one_variable_model(x1, y, 3)
    #Y=b0+b1x2
    one_variable_model(x2, y, 4)
    #Y=b0+b1x1+b2x2
    two_variables_model(x1, x2, y, 5)
    #log on x1
    one_variable_model(np.log(x1), y, 6)
    two_variables_model(np.log(x1), x2, y, 8)


if __name__ == '__main__':
    x = age_and_time['Age'].to_numpy()
    y = age_and_time['Time'].to_numpy()
    Q2_A(x,y)
    Q2_b()
    Q2_c()
    plt.show()
