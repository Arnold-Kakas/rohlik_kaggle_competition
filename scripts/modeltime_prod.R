library(lubridate)
library(tidyverse)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(timetk)
library(lightgbm) # lgbm
library(bonsai) # lgbm

# Read and preprocess data ----
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

# Identify missing columns
missing_cols <- setdiff(names(df), names(rohlik_test))

# Add missing columns to dfB with value 0
rohlik_test[missing_cols] <- 0

# Get unique warehouse names
warehouse_names <- unique(df$warehouse)

# adds lags 1, 3, 7, 14, 28 and 56
lag_transformer <- function(data){
  data |> 
    group_by(warehouse) |> 
    tk_augment_lags(orders, .lags = c(1, 3, 7, 14, 28, 42, 56)) |> 
    ungroup()
}

xgb_process <- function(warehouse_name, df, rohlik_test) {
  warehouse_df <- df |> 
    filter(warehouse == warehouse_name)
  
  warehouse_test <- rohlik_test |> 
    filter(warehouse == warehouse_name)
  
  splits <- warehouse_df |> 
    time_series_split(
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  xgb_params <- read_rds("data/best_model_params_xgb.RDS")
  lgbm_params <- read_rds("data/best_model_params.RDS")
  
  rec_obj <- recipe(orders ~ ., training(splits)) |> 
    step_mutate_at(warehouse, fn = droplevels) |> 
    step_timeseries_signature(date) |> 
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts"), date) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = FALSE)
  
  xgb_model <- boost_tree() |> 
    set_engine("xgboost") |> 
    set_mode("regression") %>%
    finalize_model(., xgb_params)
  
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
    set_mode("regression") %>%
    finalize_model(., lgbm_params)
  
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

xgb_recursive_process <- function(warehouse_name, df) {
  warehouse_df <- df  |> 
    select(warehouse, orders, date, holiday) |> 
    filter(warehouse == warehouse_name)
    
  warehouse_df_extended <- warehouse_df |> 
    group_by(warehouse) |> 
    future_frame(
    .date_var = date,
    .length_out = 62,
    .bind_data  = TRUE
  ) |> 
    ungroup()
  
  warehouse_df_lagged <- warehouse_df_extended |> 
    group_by(warehouse) |> 
    tk_augment_lags(orders, .lags = c(1, 3, 7, 14, 28, 42, 56)) |> 
    ungroup()
  
  future_data <- warehouse_df_lagged |> 
    filter(is.na(orders))
  
  training_data <- warehouse_df_lagged |> 
    drop_na()
  
  model_xgb_recursive <- boost_tree(
    mode = "regression"
  ) |>
    set_engine("xgboost") |> 
    fit(
      orders ~ . 
      + month(date, label = TRUE)
      - date,
      data = training_data
    ) |> 
    recursive(
      id         = "warehouse",
      transform  = lag_transformer,
      train_tail = panel_tail(training_data, warehouse, 62)
    )
  
  model_xgb_recursive
  
  model_tbl <- modeltime_table(
    model_xgb_recursive
  )
  
  forecast <- model_tbl |> 
    modeltime_forecast(
      new_data    = future_data,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    )
  
  return(forecast)
}


xgb_tuned_process <- function(warehouse_name, df, rohlik_test) {
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
  
  xgb_model <- boost_tree(
    trees = tune(), loss_reduction = tune(),
    tree_depth = tune(), min_n = tune(),
    mtry = tune(), sample_size = tune(),
    learn_rate = tune()
  ) |> 
    set_engine("xgboost") |> 
    set_mode("regression")
  
  workflow_xgb <- workflow() |> 
    add_model(xgb_model) |> 
    add_recipe(rec_obj)
  
  cv_folds <- time_series_cv(
    data        =  training(splits),
    assess      = "2 months",
    initial     = "18 months",
    skip        = "2 months",
    slice_limit = 4
  )
  
  workflow <- workflow() |> 
    add_model(xgb_model) |> 
    add_recipe(rec_obj)
  
  xgb_params <- parameters(
    trees(), learn_rate(), loss_reduction(),
    tree_depth(), min_n(),
    sample_size = sample_prop(),
    finalize(mtry(), training(splits))
  )
  
  xgb_params <- xgb_params |> update(trees = trees(c(200, 500)))

  xgb_tune <-
    workflow |>
    tune_bayes(
      resamples = cv_folds,
      param_info = xgb_params,
      iter = 30,
      metrics = metric_set(rmse, mape),
      control = control_bayes(
        no_improve = 20,
        save_pred = TRUE, verbose = TRUE
      )
    )
  
  best_model <- select_best(xgb_tune, metric = "rmse")
  finalized_model <- finalize_model(xgb_model, best_model)
  updated_workflow <- workflow |> update_model(finalized_model)
  xgb_fit <- fit(updated_workflow, data = training(splits))
  
  final_model <- last_fit(updated_workflow, split = splits)
  
  
  forecast <-  predict(xgb_fit, warehouse_test) |>
    bind_cols(warehouse_test)
  
  return(forecast)
}

lgbm_tuned_process <- function(warehouse_name, df, rohlik_test) {
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
  
  lgbm_model <- boost_tree(
    trees = tune(), loss_reduction = tune(),
    tree_depth = tune(), min_n = tune(),
    mtry = tune(),
    learn_rate = tune()
  ) |> 
    set_engine("lightgbm") |> 
    set_mode("regression")
  
  workflow_lgbm <- workflow() |> 
    add_model(lgbm_model) |> 
    add_recipe(rec_obj)
  
  cv_folds <- time_series_cv(
    data        =  training(splits),
    assess      = "2 months",
    initial     = "18 months",
    skip        = "2 months",
    slice_limit = 4
  )
  
  workflow <- workflow() |> 
    add_model(lgbm_model) |> 
    add_recipe(rec_obj)
  
  lgbm_params <- parameters(
    trees(), learn_rate(), loss_reduction(),
    tree_depth(), min_n(),
    finalize(mtry(), training(splits))
  )
  
  lgbm_params <- lgbm_params |> update(trees = trees(c(200, 500)))
  
  lgbm_tune <-
    workflow |>
    tune_bayes(
      resamples = cv_folds,
      param_info = lgbm_params,
      iter = 30,
      metrics = metric_set(rmse, mape),
      control = control_bayes(
        no_improve = 20,
        save_pred = TRUE, verbose = TRUE
      )
    )
  
  best_model <- select_best(lgbm_tune, metric = "rmse")
  finalized_model <- finalize_model(lgbm_model, best_model)
  updated_workflow <- workflow |> update_model(finalized_model)
  lgbm_fit <- fit(updated_workflow, data = training(splits))
  
  final_model <- last_fit(updated_workflow, split = splits)
  
  
  forecast <-  predict(lgbm_fit, warehouse_test) |>
    bind_cols(warehouse_test)
  
  return(forecast)
}


# xgb ----
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

write_csv(xgb_pred, "data/xgb_pred.csv")

# lgbm ----
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

write_csv(lgbm_pred, "data/lgbm_pred.csv")


# xgb_recursive ----
xgb_recursive_Brno_1 <- xgb_recursive_process("Brno_1", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  group_by(warehouse) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  ungroup() |> 
  select(id, orders)

xgb_recursive_Prague_1 <- xgb_recursive_process("Prague_1", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_Prague_2 <- xgb_recursive_process("Prague_2", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_Prague_3 <- xgb_recursive_process("Prague_3", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_Budapest_1 <- xgb_recursive_process("Budapest_1", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_Munich_1 <- xgb_recursive_process("Munich_1", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_Frankfurt_1 <- xgb_recursive_process("Frankfurt_1", df) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_recursive_pred <- 
  bind_rows(
    xgb_recursive_Brno_1,
    xgb_recursive_Munich_1,
    xgb_recursive_Frankfurt_1,
    xgb_recursive_Budapest_1,
    xgb_recursive_Prague_1,
    xgb_recursive_Prague_2,
    xgb_recursive_Prague_3
  ) |> 
  filter(id %in% xgb_pred$id)

# xgb_tuned ----
xgb_tuned_Brno_1 <- xgb_tuned_process("Brno_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Prague_1 <- xgb_tuned_process("Prague_1", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Prague_2 <- xgb_tuned_process("Prague_2", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Prague_3 <- xgb_tuned_process("Prague_3", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Budapest_1 <- xgb_tuned_process("Budapest_1", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Munich_1 <- xgb_tuned_process("Munich_1", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_Frankfurt_1 <- xgb_tuned_process("Frankfurt_1", df, rohlik_test) |> 
mutate(orders = round(.pred,0),
       id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

xgb_tuned_pred <- 
  bind_rows(
    xgb_tuned_Brno_1,
    xgb_tuned_Munich_1,
    xgb_tuned_Frankfurt_1,
    xgb_tuned_Budapest_1,
    xgb_tuned_Prague_1,
    xgb_tuned_Prague_2,
    xgb_tuned_Prague_3
  )

write_csv(xgb_tuned_pred, "data/xgb_tuned_pred.csv")

# lgbm_tuned ----
lgbm_tuned_Brno_1 <- lgbm_tuned_process("Brno_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Prague_1 <- lgbm_tuned_process("Prague_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Prague_2 <- lgbm_tuned_process("Prague_2", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Prague_3 <- lgbm_tuned_process("Prague_3", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Budapest_1 <- lgbm_tuned_process("Budapest_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Munich_1 <- lgbm_tuned_process("Munich_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_Frankfurt_1 <- lgbm_tuned_process("Frankfurt_1", df, rohlik_test) |> 
  mutate(orders = round(.pred,0),
         id = paste0(warehouse, "_", as.character(date))) |> 
  select(id, orders)

lgbm_tuned_pred <- 
  bind_rows(
    lgbm_tuned_Brno_1,
    lgbm_tuned_Munich_1,
    lgbm_tuned_Frankfurt_1,
    lgbm_tuned_Budapest_1,
    lgbm_tuned_Prague_1,
    lgbm_tuned_Prague_2,
    lgbm_tuned_Prague_3
  )

write_csv(lgbm_tuned_pred, "data/lgbm_tuned_pred.csv")

tuned_sub <- rohlik_test |> 
  select(date, warehouse) |> 
  mutate(id = paste0(warehouse, "_", as.character(date))) |> 
  select(-warehouse, -date) |> 
  left_join(xgb_tuned_pred, join_by(id == id), keep = FALSE) |> 
  select(orders, id)

ensemble_pred <- xgb_pred |> 
  left_join(lgbm_pred, join_by(id == id), keep = FALSE) |> 
  left_join(xgb_tuned_pred, join_by(id == id), keep = FALSE) |> 
  mutate(orders = ifelse(is.na(orders),
                         round((orders.x + 2*orders.y)/3,0),
                         round((orders.x + 2*orders.y + 2*orders)/5,0))) |> 
  select(id, orders)

write_csv(lgbm_tuned_pred, "data/lgbm_tuned_pred.csv")

# skusit este tento ensemble
# potom prejst na parameters tuning
# pridat do dat ci islo o bridge day