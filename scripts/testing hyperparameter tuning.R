library(tidymodels)
library(modeltime)
library(timetk)

# Load example dataset
data <- m4_monthly %>% filter(id == "M750")

# Split data into training and testing sets
splits <- initial_time_split(data, prop = 0.8)

# Define recipe
recipe_spec <- recipe(value ~ date, training(splits)) %>%
  step_timeseries_signature(date) %>%
  step_rm(matches("(.iso$|xts$|hour|minute|second|am.pm)")) %>%
  step_rm(date) |> 
  step_normalize(matches("(index.num|year)")) %>%
  step_dummy(all_nominal_predictors())


model_spec_xgb <- boost_tree(mtry = tune(), trees = tune(), min_n = tune(), learn_rate = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("regression")

# Combine recipes and models into workflows
workflow_xgb <- workflow() %>%
  add_recipe(recipe_spec) %>%
  add_model(model_spec_xgb)

# Define grid for hyperparameter tuning
grid_xgb <- grid_regular(mtry(range = c(2, 5)), trees(range = c(50, 500)), min_n(range = c(2, 10)), learn_rate(range = c(0.01, 0.3)), levels = 5)

# Perform hyperparameter tuning
tune_results_xgb <- tune_grid(
  workflow_xgb,
  resamples = time_series_cv(training(splits), initial = "15 years", assess = "1 year"),
  grid = grid_xgb
)

# Select best models
best_model_xgb <- tune_results_xgb %>%
  select_best(metric = "rmse")

# Finalize the workflows

final_workflow_xgb <- workflow_xgb %>%
  finalize_workflow(best_model_xgb)

# Fit the final models on the training data

final_fit_xgb <- final_workflow_xgb %>%
  fit(training(splits))

# Make predictions on the testing data

predictions_xgb <- final_fit_xgb %>%
  modeltime_table() %>%
  modeltime_forecast(new_data = testing(splits))

# Plot the results

predictions_xgb %>%
  plot_modeltime_forecast()
