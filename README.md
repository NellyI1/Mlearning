# Customer Segmentation with K-Means Clustering

## Overview
Welcome to the Customer Segmentation project! This project uses K-Means clustering to analyze and identify different customer segments in the dataset to gain insights that can help to achieve effective marketing strategies.

## Dataset
Customer segmentation data stored in CSV files. The main files are:

- Test.csv: Contains customer test data.
- Train.csv: Contains training data for analysis.
- Sample_submission.csv: A sample format for how to submit results.

The dataset includes features like:
- Age: The age of the customer.
- Income: The customer's annual income.
- Family_Size: The number of people in the customer's family.

## How the code is organized
The IDE used is Pycharm and the code has three main functions:

1. Load Data: This function pulls in all CSV files from a specified folder and combines them into one DataFrame. It checks each file and confirms successful loading.

2. Preprocess Data: This function cleans the data by selecting numeric columns and filling in any missing values with the average for that column. It also standardizes the data to help the K-Means algorithm work better.

3. Apply K-Means: This function runs the K-Means clustering algorithm on the cleaned and standardized data, identifying different customer segments.

After applying the K-means clustering, the results are saved to a new CSV file named `clustered_data.csv`. There is a scatter plot showing the different customer clusters visually.

## Testing the Code
To make sure everything works as expected, unit tests are included. These tests check if the data loads correctly, if the preprocessing works, and if K-Means can successfully label the data.

### How to Run Tests
To run the tests and save the results, use the following command in your terminal:

```bash
python -m unittest discover -s tests > unit_test_results.txt

See result of Unit test
Ran 3 tests in 0.024s

OK
Numeric Data Types:
 Age            float64
Income         float64
Family_Size      int64
dtype: object
NaN values filled with mean.
No remaining NaN values in the numeric data.
###
