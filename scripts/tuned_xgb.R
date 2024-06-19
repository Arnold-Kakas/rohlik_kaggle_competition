library(lubridate)
library(tidyverse)
library(tidymodels)
library(timetk)
library(tsibble)
library(feasts)
library(modeltime)
library(modeltime.resample)
library(timetk)

# Read and preprocess data ----
sub_template <- read_csv("data/test.csv")

# Feature engineering ----
lag_roll_transformer <- function(data){
  data |> 
    group_by(warehouse) |> 
    arrange(warehouse, date) |> 
    select(!contains("lag")) |> 
    tk_augment_lags(.value = orders, 
                    .lags = c(2, 6, 7, 8, 14, 28)) %>%  
    tk_augment_slidify(.value = contains("lag"), 
                       .period = c(3, 5, 7), 
                       .f = ~mean(., na.rm = TRUE),
                       .partial = TRUE,
                       .align = "center") |> 
    # tk_augment_fourier(.date_var = date,
    #                    .periods = c(2, 7, 14, 28),
    #                    .K = 1
    #) |> 
    ungroup()
}

df <- read_rds("data/expanded_data.RDS") |> 
  mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0))) |> 
  mutate(orders = ifelse(is.na(orders) & date < ymd("2024-03-16"), 0, orders)) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |>  
  select(-c(user_activity_1, user_activity_2))|> 
  tk_augment_timeseries_signature(.date_var = date) |> 
  select(-c(index.num,
            diff,
            year.iso,
            half,
            month.xts,
            month.lbl,
            hour,
            minute,
            second,
            hour12,
            am.pm,
            wday.xts,
            wday.lbl,
            week.iso,
            week2,
            week3,
            week4,
            mday7)) |>  
  group_by(warehouse) |> 
  arrange(warehouse, date) |> 
  mutate(bridge_day = ifelse(holiday == 0 & wday == 2 & lead(holiday, 1) == 1,
                             1,
                             ifelse(holiday == 0 & wday == 6 & lag(holiday, 1) == 1,
                                    1,
                                    0)),
         day_before_holiday = ifelse(holiday == 0 & lead(holiday, 1), 1, 0),
         day_after_holiday = ifelse(holiday == 0 & lag(holiday, 1), 1, 0)) |> 
  ungroup() |> 
  filter(date < ymd("2024-03-16"))

rohlik_test <- read_csv("data/test.csv") |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  select(-id) |> 
  tk_augment_timeseries_signature(.date_var = date) |> 
  select(-c(index.num,
            diff,
            year.iso,
            half,
            month.xts,
            month.lbl,
            hour,
            minute,
            second,
            hour12,
            am.pm,
            wday.xts,
            wday.lbl,
            week.iso,
            week2,
            week3,
            week4,
            mday7)) |> 
  group_by(warehouse) |> 
  arrange(warehouse, date) |> 
  mutate(bridge_day = ifelse(holiday == 0 & wday == 2 & lead(holiday, 1) == 1,
                             1,
                             ifelse(holiday == 0 & wday == 6 & lag(holiday, 1) == 1,
                                    1,
                                    0)),
         day_before_holiday = ifelse(holiday == 0 & lead(holiday, 1), 1, 0),
         day_after_holiday = ifelse(holiday == 0 & lag(holiday, 1), 1, 0)) |> 
  ungroup() |> 
  mutate(day_before_holiday = coalesce(day_before_holiday,0),
         day_after_holiday = coalesce(day_after_holiday,0))

# Identify missing columns
missing_cols <- setdiff(names(df), names(rohlik_test))

# Add missing columns to dfB with value 0
rohlik_test[missing_cols] <- 0

rohlik_test <- rohlik_test |> 
  select(-orders)


df_extended <- bind_rows(df, rohlik_test)
# create training and testing datasets
# df_w_features <- df |> 
#   drop_na()
# 
# test_rohlik <- df |> 
#   filter(date > ymd("2024-03-15"))

df_rolling <- df_extended |> 
  arrange(warehouse, date) #|> 
#lag_roll_transformer()

train_data <- df_rolling |> 
  drop_na()


future_data <- df_rolling %>%
  filter(is.na(orders))

splits <- train_data |> 
  time_series_split(
    date_var = date,
    assess = 60,
    cumulative = TRUE
  )

# workflow preparation ----
# cross-validation folds
set.seed(123)
folds <- vfold_cv(training(splits),
                  strata = warehouse,
                  v = 6)

# recipe
rec_obj <- recipe(orders ~ ., training(splits), skip = TRUE) |> 
  step_rm(date) |> 
  #step_naomit(contains("lag"), skip = TRUE) |> 
  step_novel() |> 
  step_zv(all_predictors()) |> 
  #step_normalize(contains("lag")) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# model
xgb_model <- boost_tree(
  trees = tune(), loss_reduction = tune(),
  tree_depth = tune(), min_n = tune(),
  mtry = tune(),
  learn_rate = tune()
) |> 
  set_engine("xgboost") |> 
  set_mode("regression")

workflow_xgb <- workflow() |> 
  add_model(xgb_model) |> 
  add_recipe(rec_obj,
             blueprint = hardhat::default_recipe_blueprint(allow_novel_levels = TRUE))

# stage 1 tuning ----
grid <- grid_latin_hypercube(
  mtry(range = c(1, length(df) - 1)),
  trees(),
  min_n(),
  tree_depth(),
  learn_rate(),
  loss_reduction(),
  size = 20
)

xgb_tune <-
  workflow_xgb |>
  tune_grid(
    resamples = folds,
    grid = grid,
    metrics = metric_set(mae, rmse, rsq),
    control = control_grid(
      save_pred = TRUE, 
      verbose = TRUE
    )
  )

plot <- workflowsets::autoplot(xgb_tune)

plotly::ggplotly(plot)
plot

# stage 2 tuning ----
grid_stage_2 <- grid_latin_hypercube(
  mtry(range = c(1, length(df) - 1)),
  trees(),
  min_n(),
  tree_depth(),
  learn_rate(range = c(-2.4, -1)),
  loss_reduction(),
  size = 20
)

xgb_tune_stage_2 <-
  workflow_xgb |>
  tune_grid(
    resamples = folds,
    grid = grid_stage_2,
    metrics = metric_set(mae, rmse, rsq),
    control = control_grid(
      save_pred = TRUE, 
      verbose = TRUE
    )
  )

plot_stage_2 <- workflowsets::autoplot(xgb_tune_stage_2)
plot_stage_2
plotly::ggplotly(plot_stage_2)

# finalization and recursive model ----
best_model <- select_best(xgb_tune_stage_2, metric = "mae")

write_rds(best_model, "data/best_model_params_xgb.RDS")
best_model <- read_rds("data/best_model_params_xgb.RDS")

finalized_model <- finalize_model(xgb_model, best_model)

finalized_workflow <- workflow_xgb |>
  finalize_workflow(best_model) |>
  fit(training(splits))

recursive_forecast <- function(model, initial_data, to_forecast, warehouse, transformer_func) {
  # Filter the initial data and future data for the specific warehouse
  data <- initial_data %>% filter(warehouse == !!warehouse)
  
  to_forecast <- to_forecast |> 
    filter(warehouse == !!warehouse) |> 
    arrange(date)
  
  predictions <- data.frame()
  # Loop through each future date to make predictions
  for (i in 1:nrow(to_forecast)) {
    # Extract the current date from future_data
    current_date <- to_forecast$date[i]
    
    data <- bind_rows(to_forecast |> filter(date == current_date), data |> arrange(desc(date))) |>
      transformer_func() |> 
      arrange(desc(date))
    
    # Prepare the data for prediction
    print(nrow(data))
    print(data[1, ] |> select(date, orders, contains("lag")))
    
    data_warehouse <- data |> 
      filter(date == current_date)
    
    # Make a prediction for the next time step
    next_prediction <- predict(finalized_workflow, new_data = data_warehouse)$.pred
    
    # Create a new row with the current date, warehouse, and prediction
    new_row <- data_warehouse |> 
      mutate(orders = round(next_prediction, 0),
             .pred = round(next_prediction, 0))
    
    # Store the prediction
    predictions <- bind_rows(predictions, new_row)
    
    # Add the new prediction to the dataset
    data <- data |> 
      filter(date != current_date) %>% # not native due to "."
      bind_rows(., new_row) |> 
      arrange(desc(date))
    
    print(i)
    print(current_date)
  }
  
  return(predictions)
}

# List of unique warehouses
warehouses <- unique(future_data$warehouse)

# Run the recursive forecast for each warehouse
# all_predictions <- lapply(warehouses, function(wh) {
#   recursive_forecast(finalized_workflow, train_data, future_data, wh, lag_roll_transformer)
# })

# Combine all predictions into a single dataframe
#final_predictions <- bind_rows(all_predictions)

forecast <-  predict(finalized_workflow, future_data) |>
  bind_cols(future_data |> select(warehouse, date)) |> 
  rename(orders = .pred) |>
  mutate(id = paste0(warehouse, "_", as.character(date)),
         orders = round(orders, 0)) |> 
  select(id, orders)

# submission ----
submission <- final_predictions |> 
  mutate(id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

write_csv(forecast, "data/tuned_xgb.csv")


