# PENGUIN CLASSIFICATION USING KNN

## ABOUT THE PROJECT

This project uses Machine Learning to classify penguins into different species using the K-Nearest Neighbors (KNN) algorithm.

## DATASET

The dataset contains penguin measurements such as:

* Bill Length
* Bill Depth
* Flipper Length
* Body Mass

The target is to predict the penguin species.

## PENGUIN SPECIES

### Adelie Penguin
![Adelie Penguin](images/adelie.jpg)

### Chinstrap Penguin
![Chinstrap Penguin](images/chinstrap.jpg)

### Gentoo Penguin
![Gentoo Penguin](images/gentoo.jpg)

## TECHNOLOGIES USED

* Python
* Pandas
* Matplotlib
* Scikit-learn
* KNN Algorithm

## PROJECT STEPS

1. Load the penguin dataset.
2. Clean the data by removing missing values.
3. Visualize the data using a histogram and scatter plot.
4. Split the data into training and testing sets.
5. Train the KNN classification model.
6. Calculate the model accuracy.
7. Enter new penguin measurements and predict the species.

## OUTPUT

The program displays:

* Histogram of flipper length
* Scatter plot of flipper length and body mass
* Classification accuracy
* Predicted penguin species

## HOW TO RUN

Install the required libraries:

```bash
pip install pandas matplotlib scikit-learn
```

Then run:

```bash
python penguin_classification.py
```
