# Text Classification with Bag of Words Model
This repository contains a notebooks where various classification models are trained using the Bag of Words (BoW) representation of text data. 
The notebooks demonstrate the entire workflow from text preprocessing and BoW model creation to training and evaluating the multiple classifiers.

## Features
* ***Dataset:*** I stumbled on the dataset from a <a href='https://www.superdatascience.com/'>Superdatascience</a> course; Machine Learning A-Z: AI, Python and R.
* ***Text Preprocessing:*** Tokenization, stop word removal, and vocabulary creation.
* ***Bag of Words Model:*** Implementation of BoW to convert text data into numerical vectors.
* **Classification Models:** Training and evaluation of various classifiers including:
  - Logistic Regression - <a href='./notebooks/logistic_regression.ipynb'>logistic_regression.ipynb</a>
  - Support Vector Machine (SVM) - <a href='./notebooks/svm.ipynb'>svm.ipynb</a>
  - Naive Bayes - <a href='./notebooks/naive_bayes.ipynb'>naive_bayes.ipynb</a>
  - Random Forest - <a href='./notebooks/random_forest.ipynb'>random_forest.ipynb</a>
  - k-Nearest Neighbors (k-NN) - <a href='./notebooks/knn.ipynb'>knn.ipynb</a>
  - Decision Tree - <a href='./notebooks/decision_tree.ipynb'>decision_tree.ipynb</a>
* ***Performance Metrics:*** Accuracy, precision, recall, F1-score, and K-Fold Cross Validation.

## Model Comparison
The models are evaluated by their accuracies:
```
                 TP + TN
Accuracy =  -----------------
            TP + TN + FP + FN
```
​where:

* TP (True Positives): The number of positive instances correctly classified as positive.
* TN (True Negatives): The number of negative instances correctly classified as negative.
* FP (False Positives): The number of negative instances incorrectly classified as positive.
* FN (False Negatives): The number of positive instances incorrectly classified as negative.

The models are then evaluated using K-Fold Cross Validation to compute their average accuracies and the standard deviation across a number of folds (10)


From the evaluation of the trained classification models on the data, the performances are:
| Model                  | Accuracy(%) | K-Fold Avg Accuracy | K-Fold Standard Deviation |
|------------------------|-------------|---------------------|---------------------------|
| Decision Tree          |  83.5       |  76.38              |  ± 3.77                   |
| k-NN                   |  76.5       |  70.75              |  ± 3.72                   |
| Logistic Regression    |  81.5       |  82.00              |  ± 4.72                   |
| Naive Bayes            |  67.0       |  67.88              |  ± 4.51                   |
| Random Forest          |  83.0       |  80.62              |  ± 4.65                   |
| SVM                    |  81.5       |  81.62              |  ± 3.79                   |

Considering both average accuracy and standard deviation, the Decision Tree model can be said to have performed best. It has a relatively high 
average accuracy (83.5%) and a low standard deviation (± 3.77), indicating both good overall performance and stability across folds.

While Logistic Regression and Random Forest also have high average accuracies, the **Decision Tree** model edges them out due to its lower standard deviation, indicating more consistent performance.


## Dependencies
* Python
* scikit-learn
* pandas
* numpy
* seaborn
* nltk (for text preprocessing)
