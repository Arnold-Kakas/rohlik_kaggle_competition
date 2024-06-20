library(lubridate)
library(tidyverse)
library(tidymodels)
library(modeltime)
library(modeltime.resample)
library(timetk)
library(lightgbm) # lgbm
library(bonsai) # lgbm

# Read and preprocess data ----
sub_template <- read_csv("data/test.csv")

# rohlik_test <- read_csv("data/test.csv") |> 
#   mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
#   select(-id) |>
#   mutate(warehouse = as_factor(warehouse))
# 
# df <- read_csv("data/train.csv") |> 
#   mutate(orders = ifelse(is.na(orders), 0, orders)) |> 
#   mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
#   filter(date <= ymd("2024-03-15")) |> 
#   mutate(warehouse = as_factor(warehouse)) |> 
#   select(-c(id, user_activity_1, user_activity_2))

df <- read_rds("data/expanded_data.RDS") |> 
  mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0))) |> 
  mutate(orders = ifelse(is.na(orders) & date < ymd("2024-03-16"), 0, orders)) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |>  
  select(-c(shutdown,
            mini_shutdown,
            blackout,
            mov_change,
            frankfurt_shutdown))|> #user_activity_1,user_activity_2
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
  ungroup()

rohlik_test <- df |> 
  filter(date >= ymd("2024-03-16")) |> 
  select(-orders)

rohlik_train <- df |> 
  filter(date < ymd("2024-03-16"))

# Identify missing columns
#missing_cols <- setdiff(names(df), names(rohlik_test))

# Add missing columns to dfB with value 0
#rohlik_test[missing_cols] <- 0

# Get unique warehouse names
#warehouse_names <- unique(df$warehouse)

xgb_process <- function(warehouse_name, df, rohlik_test) {
  warehouse_df <- df |> 
    filter(warehouse == warehouse_name)
  
  warehouse_test <- rohlik_test |> 
    filter(warehouse == warehouse_name)
  
  # warehouse_df <- rohlik_train |> 
  #   filter(warehouse == "Brno_1")
  # 
  # warehouse_test <- rohlik_test |> 
  #   filter(warehouse == "Brno_1")
  
  splits <- warehouse_df |> 
    time_series_split(
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  xgb_params <- read_rds("data/best_model_params_xgb.RDS")
  
  rec_obj <- recipe(orders ~ ., training(splits)) |> 
    #step_mutate_at(warehouse, fn = droplevels) |> 
    #step_timeseries_signature(date) |> 
    step_rm(date) |> 
    # step_rm(contains("am.pm"), contains("hour"), contains("minute"),
    #         contains("second"), contains("xts"), date) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_center(all_numeric_predictors()) |> 
    step_scale(all_numeric_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = FALSE)
  
  xgb_model <- boost_tree() |> 
    set_engine("xgboost") |> 
    set_mode("regression") #%>%
    #finalize_model(., xgb_params)
  
  workflow_xgb <- workflow() |> 
    add_model(xgb_model) |> 
    add_recipe(rec_obj) |> 
    fit(training(splits))
  
  model_tbl <- modeltime_table(
    workflow_xgb
  )
  model_tbl
  
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
  
  lgbm_params <- read_rds("data/best_model_params.RDS")
  
  rec_obj <- recipe(orders ~ ., training(splits)) |> 
    #step_mutate_at(warehouse, fn = droplevels) |> 
    #step_timeseries_signature(date) |> 
    step_rm(date) |> 
    # step_rm(contains("am.pm"), contains("hour"), contains("minute"),
    #         contains("second"), contains("xts"), date) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_center(all_numeric_predictors()) |> 
    step_scale(all_numeric_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = FALSE)
  
  xgb_model <- boost_tree() |> 
    set_engine("lightgbm") |> 
    set_mode("regression") #%>%
    #finalize_model(., lgbm_params)
  
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

# xgb ----
xgb_Brno_1 <- xgb_process("Brno_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_1 <- xgb_process("Prague_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_2 <- xgb_process("Prague_2", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Prague_3 <- xgb_process("Prague_3", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Budapest_1 <- xgb_process("Budapest_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Munich_1 <- xgb_process("Munich_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

xgb_Frankfurt_1 <- xgb_process("Frankfurt_1", rohlik_train, rohlik_test) |> 
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
lgbm_Brno_1 <- lgbm_process("Brno_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_1 <- lgbm_process("Prague_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_2 <- lgbm_process("Prague_2", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Prague_3 <- lgbm_process("Prague_3", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Budapest_1 <- lgbm_process("Budapest_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Munich_1 <- lgbm_process("Munich_1", rohlik_train, rohlik_test) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(orders = round(.value,0),
         id = paste0(warehouse, "_", as.character(.index))) |> 
  select(id, orders)

lgbm_Frankfurt_1 <- lgbm_process("Frankfurt_1", rohlik_train, rohlik_test) |> 
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

