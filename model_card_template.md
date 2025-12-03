# Model Card

## Model Details

This model is designed to predict whether an individual's income exceeds $50,000 annually based off of their demographic (age, education, marital status) and employment status. It is a Random Forest Classifier implemented with the default hyperparameters from scikit-learn version 1.5.1.

## Intended Use

This model is made for educational purposes and made to demonstrate machine learning techniques on structured tabular data (csv file).

## Training Data

This model was trained using 80% of the Census Income dataset, extracted by Barry Becker, through the 1994 U.S Census database.

## Evaluation Data

This model was evaluated using the other 20% that it was not previously trained on. This is to ensure the model is tested on data that it has not previously seen.

## Metrics

This model was evaluated using three standard classification metrics:

Precision: 0.7419
Recall: 0.6384
F1 Score: 0.6863

For more information on other performance metrics, please refer to the slice_output.txt.

## Ethical Considerations

If this model is used for other datasets, please ensure the data does not violate any personal information laws. The 1994 U.S Census is publically available at the UCI Machine Learning Repository and does not contain any data that may cause ethical concerns.

## Caveats and Recommendations

This model was created for educational purposes. Please take caution when applying it to other datasets as there may be biases.