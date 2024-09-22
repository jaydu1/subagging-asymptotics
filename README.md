# Precise asymptotics of bagging of regularized M-estimators


## Scripts for computing theoretical and empirical risks

- Lasso    
    - Risk of lasso and optimal lasso ensemble (Figures 4, 5 and 10):
        - `run_lasso_opt.py`    
    - Risk of full lasso ensemble (Figures 6 and 11):
        - `run_lasso_equiv.py`
    - Risk of optimal lasso ensemble (Figure 7):
        - `run_lasso_opt_2.py`
    - Fixed-point quantities of lassoless (Figure 8):
        - `run_lassoless.py`
    - Empirical risk of lassoless ensemble (Figure 9):
        - `run_lasso_emp.py`
- Huber
    - Risk of full unregularized Huber ensemble (Figure 12):
        - `run_huber.py`
    - Risk of l1-regularized Huber and optimal l1-regularized Huber ensemble (Figures 3):
        - `run_huber_l1_opt.py`
    - Risk of full l1-regularized Huber ensemble (Figures 2, 8, 13 and 14):
        - `run_huber_l1_emp.py`
        - `run_huber_l1_equiv.py`
- Utility functions
    - `compute_risk.py`
    - `generate_data.py`   
- Visualization
    - The figures can be reproduced with the Jupyter Notebook `Plot.ipynb`.



## Computation details

All the experiments are run on Ubuntu 22.04.4 LTS using 12 cores and 128 GB of RAM.

The estimated time to run all experiments is roughly 12 hours.

## Dependencies

Package | Version
--- | ---
h5py | 3.1.0
joblib | 1.4.0
matplotlib | 3.4.3
numpy | 1.20.3
pandas | 1.3.3
python | 3.8.12
scikit-learn | 1.3.2
sklearn_ensemble_cv | 0.2.3
scipy | 1.10.1
statsmodels | 0.13.5
tqdm | 4.62.3