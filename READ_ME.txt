Data_Gen folder contains all code related to generating the Data_Gen

HNN.py is the main training file and is run from the terminal with a config file
like hnn_config.yml

HNN_helper, ODE_pinn_helper and architectures are helper files used by HNN.py

HNNconfigs folder contains alot of different configs that has been trained
HNNruns contains alot of different tensorboard logfiles for each config file

models folder contains saved parameter for the final models such that their performance can be validated
figs folder contains variuos plots that is generated with the files in the plotting_etc folder. 