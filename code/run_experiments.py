#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  6 15:18:58 2022

@author: makraus
"""

from sklearn.linear_model import LinearRegression
from igann_interactions import IGANN
from pygam.pygam import LinearGAM
import load_regression_datasets
import load_classification_datasets
from sklearn.metrics import mean_squared_error

small_reg = False
large_reg = True

if small_reg:
    # Small Regression
    for i in range(1,10):
        for j in range(30):
            X_train, X_test, y_train, y_test = load_regression_datasets.get_dataset(str(i), j)
            
            m = IGANN(task='regression', verbose=0)
            
            m.fit(X_train, y_train)
            
            igann_train_loss = mean_squared_error(y_train, m.predict(X_train))
            igann_test_loss = mean_squared_error(y_test, m.predict(X_test))
            
            gam = LinearGAM()
            gam.fit(X_train, y_train)
            
            gam_train_loss = mean_squared_error(y_train, gam.predict(X_train))
            gam_test_loss = mean_squared_error(y_test, gam.predict(X_test))
            
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            lr_train_loss = mean_squared_error(y_train, lr.predict(X_train))
            lr_test_loss = mean_squared_error(y_test, lr.predict(X_test)) 
            
            write_str = f'Split = {j}; {igann_train_loss}; {igann_test_loss};' + \
                        f'{gam_train_loss}; {gam_test_loss};' + \
                        f'{lr_train_loss}; {lr_test_loss}\n'
            
            with open(f'../results/Regr_DS_{i}_results.txt', 'a') as fd:
                fd.write(write_str)

if large_reg:    
    # Large Regression
    for i in range(10, 13):
        for j in range(30):
            X_train, X_test, y_train, y_test = load_regression_datasets.get_dataset(str(i), j)
            
            m = IGANN(task='regression', verbose=1)
            
            m.fit(X_train, y_train)
            
            igann_train_loss = mean_squared_error(y_train, m.predict(X_train))
            igann_test_loss = mean_squared_error(y_test, m.predict(X_test))
            
            gam = LinearGAM()
            gam.fit(X_train, y_train)
            
            gam_train_loss = mean_squared_error(y_train, gam.predict(X_train))
            gam_test_loss = mean_squared_error(y_test, gam.predict(X_test))
            
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            
            lr_train_loss = mean_squared_error(y_train, lr.predict(X_train))
            lr_test_loss = mean_squared_error(y_test, lr.predict(X_test)) 
            
            write_str = f'Split = {j}; {igann_train_loss}; {igann_test_loss};' + \
                        f'{gam_train_loss}; {gam_test_loss};' + \
                        f'{lr_train_loss}; {lr_test_loss}\n'
            
            with open(f'../results/Regr_DS_{i}_results.txt', 'a') as fd:
                fd.write(write_str)