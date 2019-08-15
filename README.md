# bayes_and_knn_classifiers
Bespoke implemententation the K-Nearest Neighbour and Naïve Bayes algorithms using only native Python libraries and numpy. 

Evaluated on the famous Pima Indians Diabetes data set https://www.kaggle.com/uciml/pima-indians-diabetes-database using 
stratified 10-fold cross-validation (implementation of cross-validation included).

To run the classifier:
`python bayes_knn.py training.txt testing.txt { algorithm } `

for Naive Bayes `{ algorithm } ` = NB 

for KNN `{ algorithm } ` = 1NN or 2NN or 3NN or 4NN or 5NN

e.g. using the data set provided: 
`python3 bayes_knn.py pima.csv pima_test.csv NB`
