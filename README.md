# Kaggle-Santander
The repo containing my codes for the Kaggle Santander Customer Satisfaction competition 
https://www.kaggle.com/c/santander-customer-satisfaction

The file santander_cv.py is the single python file containing my code for model evaluation using K fold cross validation as well as the code for making predictions.

This competiton was the most fiercely competed one on Kaggle with over 5000 participating teams. I managed to finish in the top 10%. My final model was a weighted geometric mean blend of the highest scoring public script (10%) and my own 500-XGB-Bag model (90%). The 500-XGB-Bag Model was a bootsrap aggregate of 500 XG Boost classifiers (And no that does not sound crazy, not by Kaggle standards at least). My model scored an AUC of 0.827 on the test data as opposed the winning model score of 0.829.  

The codes are old (April 2016) and hence are outdated from a Scikit Learn API perspective. The KFold and PCA features in the latest version of Scikit learn follow a different syntax.
