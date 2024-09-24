# Federated LGBM on Penguins Dataset

![Alt text](images/penguins.png "three penguins")
Artwork by @allison_horst

# Description
An federated solution for the LightGBM model. Most of the code is from the Flower XGBoost example, but modified to support LightGBM and multi-class data for this penguins dataset.

# How to Run
1. Download and extract dataset and put penguins_size.csv in data/
2. `docker build -t federated-lgbm-penguins .`
3. `docker run federated-lgbm-penguins`

# Technolgies Used
Python 3.8, Flsower 1.4, Pytorch, Pandas, LightGBM

# Citations
Gorman KB, Williams TD, Fraser WR (2014). Ecological sexual dimorphism and environmental variability within a community of Antarctic penguins (genus Pygoscelis). PLoS ONE 9(3):e90081. https://doi.org/10.1371/journal.pone.0090081

https://www.kaggle.com/datasets/parulpandey/palmer-archipelago-antarctica-penguin-data

https://github.com/adap/flower/blob/32912119cd1f1376234c58c3bf8f9fb9c213e706/examples/quickstart_xgboost_horizontal/code_horizontal.ipynb

# License
This project is licensed under the Apache License - see the LICENSE file for details.