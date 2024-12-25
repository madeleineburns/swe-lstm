# Comparing Snow Water Equivalent Estimations from Long Short-Term Memory Networks and Physics-Based Models in the Western United States

This repository contains the code necessary to train an LSTM model to estimate snow water equivalent (SWE) in the western U.S. This repository also contains the code to test the model on a representative set of national test sites, evaluate its performance in comparison to two physics-based models (ParFlow-CLM and the UA SWE model), and investigate the strengths and limitations of the all three models on the testing data. A brief description of the files in this repository is included here. 

### Model scripts
`run_lstm.py`: script for training the LSTM on a user-specified set of training data, generating predictions for the testing dataset, and quantifying model performance on the testing dataset.<br>
`run_pfclm.py`: script for running ParFlow-CLM for all of the years in the testing dataset and quantifying model performance.<br>

### Analysis notebooks
`lstm_analysis.ipynb`: Jupyter notebook for analysis of LSTM model predictions on the testing dataset. <br>
`lstm_regime_model.ipynb`: Jupyter notebook for analysis of the predictions of multiple LSTM models trained on varying percentages of data from each snowpack regime.<br>
`lstm_state_model.ipynb`: Jupyter notebook for training an LSTM model on data from a single year and analyzing the model's transferability.<br>
`manage_data.ipynb`: Jupyter notebook with assorted functions for managing or analyzing training data, testing data, and performance statistics.<br>
`model_comparisons.ipynb`: Jupyter notebook for the analysis of estimations from the LSTM model, ParFlow-CLM, and the UA SWE model. <br>
`parflow_analysis.ipynb`: Jupyter notebook for the analysis of ParFlow-CLM predictions. <br>

### Supporting files
`_data.py`: function library script supporting data collection, cleaning, and analysis. <br>
`_lstm.py`: function library script supporting the development, training, and testing of the LSTM model.<br> 
`national_test_sites.txt`: text file listing the sites included in the testing data. <br>
`national_test_years.txt`: text file listing the full set of testing data (both sites and years). <br>
