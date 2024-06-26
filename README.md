
# Rohlik.cz Daily Orders Forecasting

This repository contains the code and documentation for forecasting daily orders for Rohlik.cz using the `modeltime` framework. This project is a submission for the Kaggle forecasting competition hosted by Rohlik.cz.

More details regarding the competition can be found [here](https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge)

## Project Overview

The goal of this project is to forecast the daily orders of Rohlik.cz for the next 60 days. Accurate forecasting will help Rohlik.cz manage inventory, optimize delivery schedules, and improve customer satisfaction.

## Dataset

The dataset used for this project includes historical daily order data provided by Rohlik.cz.

There are 5 files provided:
   1. train.csv - the training set containing the historical orders data and selected features described below
   2. test.csv - the test set
   3. solution_example.csv - a sample submission file in the correct format
   4. train_calendar.csv - a calendar for the training set containing data about holidays or warehouse specific events, some columns are already in the train data but there are additional rows in this file for dates where some warehouses could be closed due to public holiday or Sunday (and therefore they are not in the train set)
   5. test_calendar.csv - a calendar for the test set

## Methodology

### Modeltime Framework

We use the `modeltime` framework for time series forecasting in R. `modeltime` provides a suite of tools for building, tuning, and ensembling time series models. Key features include:

- Integration with `tidymodels` for consistent model building and evaluation.
- Support for a wide range of forecasting models, including ARIMA, exponential smoothing, and machine learning models.
- Easy model comparison and ensembling.

### Workflow

1. **Data Preprocessing**: Clean and transform the historical order data.
2. **Feature Engineering**: Create relevant features that can help improve model accuracy.
3. **Model Building**: Train multiple forecasting models using the `modeltime` framework.
4. **Model Evaluation**: Compare model performance using cross-validation.
5. **Ensembling**: Combine the best-performing models to create a robust ensemble model.
6. **Forecasting**: Generate forecasts for the next 60 days and evaluate their accuracy.

## Dependencies

This project requires the following R packages:

- `tidyverse`
- `modeltime`
- `tidymodels`
- `lubridate`
- `timetk`
- `ggplot2`

## License

This project is licensed under the GPL-3.0 License. See the `LICENSE` file for details.

## Acknowledgements

- [Rohlik.cz](https://www.rohlik.cz) for providing the data and hosting the competition.
- The `modeltime` and `tidymodels` teams for developing such powerful tools for time series forecasting.

## Contact

For any questions or feedback, please contact me at kakas@cleandata.sk.
