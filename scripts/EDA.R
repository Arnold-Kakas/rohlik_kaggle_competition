library(timetk)
library(ggplot2)
library(lubridate)
library(tidyverse)
library(tidymodels)
library(modeltime)
library(lightgbm) # lgbm
library(bonsai) # lgbm

# to do:
# mean precipitation a snow po mesiacoch
# forecast user_activity 1 a 2

calendar_test <- read_csv("data/test_calendar.csv")
calendar_train <- read_csv("data/train_calendar.csv")


calendar <- bind_rows(
  calendar_train,
  calendar_test
  ) |> 
  filter(
    date >= ymd("2020-12-05"),
    date <= ymd("2024-05-15")
  )

full_time_series <- calendar |> 
  select(warehouse, date) |> 
  distinct()

summarize(calendar_train,
          .by = warehouse,
          min_date = min(date),
          max_date = max(date))

rohlik_test <- read_csv("data/test.csv")
rohlik_train <- read_csv("data/train.csv") |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  select(-id)

# rohlik_all <- bind_rows(
#   rohlik_train,
#   rohlik_test
# ) |> 
#   select(-id)

precip_snow <- rohlik_train |> 
  mutate(month_number = as.character(month(date))) |> 
  summarize(.by = c(warehouse, 
                    month_number),
            snow = round(mean(snow, na.rm = TRUE),2),
            precipitation = round(mean(precipitation, na.rm = TRUE),2))

rohlik_test_adj <-  rohlik_test |> 
  mutate(month_number = as.character(month(date))) |> 
  left_join(precip_snow, 
            join_by(warehouse == warehouse, 
                      month_number == month_number), 
            keep = FALSE) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
  select(-id)

missing_cols <- setdiff(names(rohlik_train), names(rohlik_test_adj))

rohlik_test_adj[missing_cols] <- 0


# xgb ua 1 function ----
xgb_user_activity_1 <- function(warehouse_name, train_df, test_df) {
  warehouse_df <- train_df |>
    select(-orders) |> 
    filter(warehouse == !!warehouse_name)

  warehouse_test <- test_df |>
    filter(warehouse == !!warehouse_name)
  
  # warehouse_df <- rohlik_train |> 
  #   filter(warehouse == "Brno_1")
  # 
  # warehouse_test <- rohlik_test_adj |> 
  #   filter(warehouse == "Brno_1")
  
  splits <- warehouse_df |> 
    time_series_split(
      date_var = date,
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  rec_obj <- recipe(user_activity_1 ~ ., training(splits)) |> 
    #step_mutate_at(warehouse, fn = droplevels) |> 
    step_timeseries_signature(date) |> 
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts"), date, user_activity_2) |> 
    step_novel() |> 
    step_center(all_numeric_predictors()) |> 
    step_scale(all_numeric_predictors()) |> 
    step_zv(all_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = FALSE)
  
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
  
  model_tbl
  
  calib_tbl <- model_tbl |> 
    modeltime_calibrate(
      new_data = testing(splits), 
      id       = "warehouse",
      quiet = FALSE
    )
  
  refit_tbl <- calib_tbl |> 
    modeltime_refit(data = warehouse_df)
  
  forecast <- refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    )
  
  refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    ) |> 
    group_by(warehouse) |> 
    plot_modeltime_forecast(
      .facet_ncol  = 1,
      .interactive = FALSE
    ) |> 
    print()
  
  return(forecast)
}

xgb_ua1_Brno_1 <- xgb_user_activity_1("Brno_1", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Prague_1 <- xgb_user_activity_1("Prague_1", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Prague_2 <- xgb_user_activity_1("Prague_2", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Prague_3 <- xgb_user_activity_1("Prague_3", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Budapest_1 <- xgb_user_activity_1("Budapest_1", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Munich_1 <- xgb_user_activity_1("Munich_1", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_Frankfurt_1 <- xgb_user_activity_1("Frankfurt_1", rohlik_train, rohlik_test_adj |> select(-user_activity_1)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_1 = .value)

xgb_ua1_pred <- 
  bind_rows(
    xgb_ua1_Brno_1,
    xgb_ua1_Munich_1,
    xgb_ua1_Frankfurt_1,
    xgb_ua1_Budapest_1,
    xgb_ua1_Prague_1,
    xgb_ua1_Prague_2,
    xgb_ua1_Prague_3
  )

# xgb ua 1 function ----
xgb_user_activity_2 <- function(warehouse_name, train_df, test_df) {
  warehouse_df <- train_df |>
    select(-orders) |> 
    filter(warehouse == !!warehouse_name)
  
  warehouse_test <- test_df |>
    filter(warehouse == !!warehouse_name)
  
  # warehouse_df <- rohlik_train |> 
  #   filter(warehouse == "Brno_1")
  # 
  # warehouse_test <- rohlik_test_adj |> 
  #   filter(warehouse == "Brno_1")
  
  splits <- warehouse_df |> 
    time_series_split(
      date_var = date,
      assess     = "2 months", 
      cumulative = TRUE
    )
  
  rec_obj <- recipe(user_activity_2 ~ ., training(splits)) |> 
    #step_mutate_at(warehouse, fn = droplevels) |> 
    step_timeseries_signature(date) |> 
    step_rm(contains("am.pm"), contains("hour"), contains("minute"),
            contains("second"), contains("xts"), date, user_activity_1) |> 
    step_novel() |> 
    step_zv(all_predictors()) |> 
    step_center(all_numeric_predictors()) |> 
    step_scale(all_numeric_predictors()) |> 
    step_dummy(all_nominal_predictors(), one_hot = FALSE)
  
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
  
  model_tbl
  
  calib_tbl <- model_tbl |> 
    modeltime_calibrate(
      new_data = testing(splits), 
      id       = "warehouse",
      quiet = FALSE
    )
  
  refit_tbl <- calib_tbl |> 
    modeltime_refit(data = warehouse_df)
  
  forecast <- refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    )
  
  refit_tbl |> 
    modeltime_forecast(
      new_data    = warehouse_test,
      actual_data = warehouse_df, 
      conf_by_id  = TRUE
    ) |> 
    group_by(warehouse) |> 
    plot_modeltime_forecast(
      .facet_ncol  = 1,
      .interactive = FALSE
    ) |> 
    print()
  
  return(forecast)
}

xgb_ua2_Brno_1 <- xgb_user_activity_2("Brno_1", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |> 
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Prague_1 <- xgb_user_activity_2("Prague_1", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Prague_2 <- xgb_user_activity_2("Prague_2", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Prague_3 <- xgb_user_activity_2("Prague_3", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Budapest_1 <- xgb_user_activity_2("Budapest_1", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Munich_1 <- xgb_user_activity_2("Munich_1", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_Frankfurt_1 <- xgb_user_activity_2("Frankfurt_1", rohlik_train, rohlik_test_adj |> select(-user_activity_2)) |> 
  filter(.index > ymd("2024-03-15")) |> 
  select(.value, warehouse, .index) |>  
  mutate(.value = round(.value,0)) |> 
  rename(date = .index) |> 
  rename(user_activity_2 = .value)

xgb_ua2_pred <- 
  bind_rows(
    xgb_ua2_Brno_1,
    xgb_ua2_Munich_1,
    xgb_ua2_Frankfurt_1,
    xgb_ua2_Budapest_1,
    xgb_ua2_Prague_1,
    xgb_ua2_Prague_2,
    xgb_ua2_Prague_3
  )

rohlik_test_adj <- rohlik_test_adj |> 
  select(-user_activity_1, -user_activity_2, -orders, -month_number) |> 
  left_join(xgb_ua1_pred, join_by(warehouse == warehouse, date == date), keep = FALSE) |> 
  left_join(xgb_ua2_pred, join_by(warehouse == warehouse, date == date), keep = FALSE)

expanded_data <- bind_rows(rohlik_test_adj,
                           rohlik_train)

write_rds(expanded_data, "data/expanded_data.RDS")


cor(expanded_data |> 
      filter(date < ymd("2024-03-15"))|> 
      select(where(is.numeric)), method = c("pearson"))
