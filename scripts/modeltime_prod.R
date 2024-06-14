library(lubridate)
library(tidyverse)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(timetk)
library(lightgbm) # lgbm
library(bonsai) # lgbm

# adds lags 1, 3, 7, 14, 28 and 56
lag_transformer <- function(data){
  data  |> 
    tk_augment_lags(orders, .lags = c(1, 3, 7, 14, 28))
}

xgb_process <- function(warehouse_name, df, rohlik_test) {
  warehouse_df <- df |> 
    filter(warehouse == warehouse_name) |> 
    lag_transformer()
  
  warehouse_test <- rohlik_test |> 
    filter(warehouse == warehouse_name)
  
 splits <- warehouse_df |> 
    time_series_split(
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  rec_obj <- recipe(orders ~ ., training(splits)) |> 
    step_mutate_at(warehouse, fn = droplevels) |> 
    step_timeseries_signature(date) |> 
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts"), date) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  
  xgb_model <- boost_tree() |> 
    set_engine("xgboost") |> 
    set_mode("regression")
  
  workflow_xgb <- workflow() |> 
    add_model(xgb_model) |> 
    add_recipe(rec_obj) |> 
    fit(training(splits))
  
  model_tbl <- modeltime_table(
    workflow_xgb
  )
  
  calib_tbl <- model_tbl |> 
    modeltime_calibrate(
      new_data = testing(splits), 
      id       = "warehouse"
    )

  refit_tbl <- calib_tbl |> 
    modeltime_refit(data = warehouse_df)
  
  forecast <- refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    )
  
  return(forecast)
}

lgbm_process <- function(warehouse_name, df, rohlik_test) {
  warehouse_df <- df |> 
    filter(warehouse == warehouse_name)
  
  warehouse_test <- rohlik_test |> 
    filter(warehouse == warehouse_name)
  
  splits <- warehouse_df |> 
    time_series_split(
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  rec_obj <- recipe(orders ~ ., training(splits)) |> 
    step_mutate_at(warehouse, fn = droplevels) |> 
    step_timeseries_signature(date) |> 
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts"), date) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = TRUE)
  
  xgb_model <- boost_tree() |> 
    set_engine("lightgbm") |> 
    set_mode("regression")
  
  workflow_xgb <- workflow() |> 
    add_model(xgb_model) |> 
    add_recipe(rec_obj) |> 
    fit(training(splits))
  
  model_tbl <- modeltime_table(
    workflow_xgb
  )
  
  calib_tbl <- model_tbl |> 
    modeltime_calibrate(
      new_data = testing(splits), 
      id       = "warehouse"
    )
  
  refit_tbl <- calib_tbl |> 
    modeltime_refit(data = warehouse_df)
  
  forecast <- refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    )
  
  return(forecast)
}

# Read and preprocess data
sub_template <- read_csv("data/test.csv")

rohlik_test <- read_csv("data/test.csv") |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  select(-id) |>
  mutate(warehouse = as_factor(warehouse))

df <- read_csv("data/train.csv") |> 
  mutate(orders = ifelse(is.na(orders), 0, orders)) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  filter(date <= ymd("2024-03-15")) |> 
  mutate(warehouse = as_factor(warehouse)) |> 
  select(-c(id, user_activity_1, user_activity_2))

df0 <- read_rds("data/full_data_cleaned.RDS") |> 
  mutate(orders = ifelse(is.na(orders), 0, orders)) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  select(-c(orders_lag_1d, orders_lag_3d, orders_lag_7d, orders_lag_14d, orders_lag_28d)) |> 
  filter(date <= ymd("2024-03-15")) |> 
  mutate(warehouse = as_factor(warehouse))

# Identify missing columns
missing_cols <- setdiff(names(df), names(rohlik_test))

# Add missing columns to dfB with value 0
rohlik_test[missing_cols] <- 0

# Get unique warehouse names
warehouse_names <- unique(df$warehouse)


xgb_Brno_1 <- xgb_process("Brno_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_1 <- xgb_process("Prague_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_2 <- xgb_process("Prague_2", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_3 <- xgb_process("Prague_3", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Budapest_1 <- xgb_process("Budapest_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Munich_1 <- xgb_process("Munich_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Frankfurt_1 <- xgb_process("Frankfurt_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_pred <- 
  bind_rows(
    xgb_Brno_1,
    xgb_Munich_1,
    xgb_Frankfurt_1,
    xgb_Budapest_1,
    xgb_Prague_1,
    xgb_Prague_2,
    xgb_Prague_3
  )


lgbm_Brno_1 <- lgbm_process("Brno_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_1 <- lgbm_process("Prague_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_2 <- lgbm_process("Prague_2", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_3 <- lgbm_process("Prague_3", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Budapest_1 <- lgbm_process("Budapest_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Munich_1 <- lgbm_process("Munich_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Frankfurt_1 <- lgbm_process("Frankfurt_1", df, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_pred <- 
  bind_rows(
    lgbm_Brno_1,
    lgbm_Munich_1,
    lgbm_Frankfurt_1,
    lgbm_Budapest_1,
    lgbm_Prague_1,
    lgbm_Prague_2,
    lgbm_Prague_3
  )

write_csv(lgbm_pred, "data/modeltime_submission.csv")

