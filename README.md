# IGANN - Interpretable Generalized Additive Neural Networks

--- This project is under active development ---

IGANN is a novel machine learning model from the family of generalized additive models (GAMs). This GAM is special in the sense that it uses a combination of extreme learning machines and gradient boosting.

Some of the main features are:
- Training is very fast and can be performed on CPU and GPU  
- The rate of change of so-called shape functions can be influenced through hyperparameter ELM_scale  
- The initial model is simply linear, thus IGANN generally also performs well on small datasets  

Main developers of this project are:

Mathias Kraus, FAU Erlangen-Nürnberg  
Daniel Tschernutter, ETH Zurich  
Sven Weinzierl, FAU Erlangen-Nürnberg  
Patrick Zschech, FAU Erlangen-Nürnberg  
Nico Hambauer, FAU Erlangen-Nürnberg  
Sven Kruschel, FAU Erlangen-Nürnberg  
Lasse Bohlen, FAU Erlangen-Nürnberg  
Julian Rosenberger, FAU Erlangen-Nürnberg  

# Dependencies

The project depends on PyTorch (tested with version 1.13.0).

# Usage

IGANN can in general be used similar to sklearn models. The methods to interact with IGANN are the following:
- .fit(X, y) for training IGANN on (X, y) dataset
- .predict_raw(X) to compute simple prediction for regression or raw logits for classification tasks. Per default values greater than 0 could be interpreted as belonging to class 1 and values smaller than 0 as belonging to class -1. 
- .predict(X) to compute simple prediction for regression or class prediction in {-1, 1} for classification tasks. If the IGANN parameter optimize_threshold is set to True, the threshold for the class prediction is optimized on the training data and hence deviates from the decision boundary of 0.
- .predict_proba(X) to compute probability estimates
- .plot_single() to show shape functions
- .plot_learning() to show the learning curve on train and validation set
