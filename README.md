# Insurance Charges Prediction

This project trains a linear regression model on an insurance dataset to predict medical charges based on features like age, sex, BMI, children, smoking status, and region.

## Dataset
- `insurance.csv`: Main dataset used for training.
- `validation_dataset.csv`: Separate dataset for making predictions.

## Features Used
- Age
- Sex (encoded)
- BMI
- Number of children
- Smoker (encoded)
- Region (encoded)

## Workflow
1. **Data Cleaning**:
   - Dollar signs removed from charges.
   - Standardized text values in categorical columns.
   - Removed rows with invalid or missing data.

2. **Model Training**:
   - Linear regression model trained using `scikit-learn`.
   - One-hot encoding for categorical features.

3. **Evaluation**:
   - R² score calculated on training data.

4. **Prediction**:
   - Model predicts on validation dataset.
   - Ensures no predicted charge is below 1000.
   - Results appended to the validation dataframe.

## Output
- Console output of R² score and predicted charges.
- `validation_data` includes a `predicted_charges` column.

## Requirements
- pandas
- numpy
- scikit-learn
