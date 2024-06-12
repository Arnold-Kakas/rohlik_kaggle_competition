
# Rohlik.cz Daily Orders Forecasting

This repository contains the code and documentation for forecasting daily orders for Rohlik.cz using the `modeltime` framework. This project is a submission for the Kaggle forecasting competition hosted by Rohlik.cz.

More details regarding the competition can be found (here)[https://www.kaggle.com/competitions/rohlik-orders-forecasting-challenge]

## Project Overview

The goal of this project is to forecast the daily orders of Rohlik.cz for the next 60 days. Accurate forecasting will help Rohlik.cz manage inventory, optimize delivery schedules, and improve customer satisfaction.

## Dataset

The dataset used for this project includes historical daily order data provided by Rohlik.cz. This data will be preprocessed and used to train various forecasting models.

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
- `dplyr`
- `ggplot2`

You can install these packages using the following commands:

```R
install.packages(c("tidyverse", "modeltime", "tidymodels", "lubridate", "timetk", "dplyr", "ggplot2"))
```

## Usage

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/rohlik-forecasting.git
   cd rohlik-forecasting
   ```

2. Open the R project in RStudio.

3. Run the preprocessing script to clean and transform the data:
   ```R
   source("scripts/preprocessing.R")
   ```

4. Train the models and generate forecasts:
   ```R
   source("scripts/modeling.R")
   ```

5. Evaluate the models and visualize the results:
   ```R
   source("scripts/evaluation.R")
   ```

## Results

The final model ensemble achieved [mention your results here, e.g., RMSE, MAE, MAPE] on the validation set. Detailed results and visualizations can be found in the `results` directory.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [Rohlik.cz](https://www.rohlik.cz) for providing the data and hosting the competition.
- The `modeltime` and `tidymodels` teams for developing such powerful tools for time series forecasting.

## Contact

For any questions or feedback, please contact [your name] at [your email].
