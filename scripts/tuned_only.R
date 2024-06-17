library(lubridate)
library(tidyverse)
library(tidymodels)
library(tsibble)
library(feasts)
library(modeltime)
library(modeltime.resample)
library(modeltime.ensemble)
library(timetk)
library(lightgbm) # lgbm
library(bonsai) # lgbm

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
    tk_augment_fourier(.date_var = date,
                       .periods = c(2, 7, 14, 28),
                       .K = 1
    ) |> 
  ungroup()
}

# df_check <- read_rds("data/expanded_data.RDS") |> 
#   mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0))) |> 
#   mutate(orders = ifelse(is.na(orders) & date < ymd("2024-03-16"), 0, orders)) |> 
#   mutate(across(where(is.character), ~ replace_na(., "No"))) |> 
#   select(-c(user_activity_1, user_activity_2)) |> 
#   group_by(warehouse) |> 
#   tk_augment_timeseries_signature(.date_var = date) |> 
#   select(-c(index.num,
#             diff,
#             year.iso,
#             half,
#             month.xts,
#             month.lbl,
#             hour,
#             minute,
#             second,
#             hour12,
#             am.pm,
#             wday.xts,
#             wday.lbl,
#             week.iso,
#             week2,
#             week3,
#             week4,
#             mday7)) |> 
#   tk_augment_timeseries_signature(.date_var = date) |> 
#   select(-c(index.num,
#             diff,
#             year.iso,
#             half,
#             month.xts,
#             month.lbl,
#             hour,
#             minute,
#             second,
#             hour12,
#             am.pm,
#             wday.xts,
#             wday.lbl,
#             week.iso,
#             week2,
#             week3,
#             week4,
#             mday7)) |> 
#   arrange(warehouse, date) |> 
#   mutate(bridge_day = ifelse(holiday == 0 & wday == 2 & lead(holiday, 1) == 1,
#                              1,
#                              ifelse(holiday == 0 & wday == 6 & lag(holiday, 1) == 1,
#                                     1,
#                                     0)),
#          day_before_holiday = ifelse(holiday == 0 & lead(holiday, 1), 1, 0),
#          day_after_holiday = ifelse(holiday == 0 & lag(holiday, 1), 1, 0)) |> 
#   mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0))) |> 
#   tk_augment_lags(.value = orders, 
#                   .lags = c(2, 6, 7, 8, 14, 28)) %>%  
#   tk_augment_slidify(.value = contains("lag"), 
#                      .period = c(3, 5, 7), 
#                      .f = ~mean(., na.rm = TRUE),
#                      .partial = TRUE,
#                      .align = "center") |> 
#   tk_augment_fourier(.date_var = date,
#                      .periods = c(2, 7, 14, 28),
#                      .K = 1
#                      ) |> 
#   ungroup()
# 
# df_check |> 
#   filter(warehouse == "Brno_1") |> 
#   as_tsibble(key = warehouse,
#              index = date) |> 
#   ACF(difference(orders)) |>
#   autoplot()

# lags 2, 6, 7, 8, 14, 28

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
  arrange(warehouse, date) |> 
  lag_roll_transformer()

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
  step_naomit(contains("lag"), skip = TRUE) |> 
  step_novel() |> 
  step_zv(all_predictors()) |> 
  step_normalize(contains("lag")) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# model
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
  size = 30
)

lgbm_tune <-
  workflow_lgbm |>
  tune_grid(
    resamples = folds,
    grid = grid,
    metrics = metric_set(mae, rmse, rsq),
    control = control_grid(
      save_pred = TRUE, 
      verbose = TRUE
    )
  )

plot <- workflowsets::autoplot(lgbm_tune)

plotly::ggplotly(plot)


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

lgbm_tune_stage_2 <-
  workflow_lgbm |>
  tune_grid(
    resamples = folds,
    grid = grid_stage_2,
    metrics = metric_set(mae, rmse, rsq),
    control = control_grid(
      save_pred = TRUE, 
      verbose = TRUE
    )
  )

plot_stage_2 <- workflowsets::autoplot(lgbm_tune_stage_2)

plotly::ggplotly(plot_stage_2)

# finalization and recursive model ----
best_model <- select_best(lgbm_tune_stage_2, metric = "mae")
finalized_model <- finalize_model(lgbm_model, best_model)

finalized_workflow <- workflow_lgbm |>
  finalize_workflow(best_model) |>
  fit(training(splits))

model_tbl <- modeltime_table(
  finalized_workflow
)

recursive_forecast <- function(model, initial_data, future_data, warehouse, transformer_func) {
  # Filter the initial data and future data for the specific warehouse
  data <- initial_data %>% filter(warehouse == warehouse)
  future_data <- future_data |> 
    filter(warehouse == warehouse)
  
  df <- bind_rows(future_data0, data) |> arrange(date)
  
  predictions <- data.frame()
  
  # Loop through each future date to make predictions
  for (i in 1:nrow(future_data)) {
    # Extract the current date from future_data
    current_date <- future_data$date[i]
    
    # Prepare the data for prediction
    data_warehouse <- df |> filter(date == current_date) |> select(-contains(".pred"))
    
    # Make a prediction for the next time step
    next_prediction <- predict(finalized_workflow, new_data = data_warehouse)$.pred
    
    # Create a new row with the current date, warehouse, and prediction
    new_row <- future_data[i, ] %>%
      mutate(orders = next_prediction,
             .pred = next_prediction)
    
    # Store the prediction
    predictions <- bind_rows(predictions, new_row)
    
    # Add the new prediction to the dataset
    df <- df |> filter(date != current_date) %>% bind_rows(., new_row) |> arrange(date)
    
    # Update lags and rolling averages
    df <- transformer_func(df)
  }
  
  return(predictions)
}

# List of unique warehouses
warehouses <- unique(future_data$warehouse)

# Run the recursive forecast for each warehouse
all_predictions <- lapply(warehouses, function(wh) {
  recursive_forecast(finalized_workflow, train_data, future_data, wh, lag_roll_transformer)
})

# Combine all predictions into a single dataframe
final_predictions <- bind_rows(all_predictions)


# predictions ----

a <- predict(finalized_workflow, new_data = future_data)

testitng_predictions <- modeltime_forecast(
  model_tbl,
  new_data = rohlik_test |> filter(warehouse == "Prague_1"),
  actual_data = df_rolling |> filter(warehouse == "Prague_1"),
  keep_data = TRUE
)

plot_modeltime_forecast(tesitng_predictions)















data <- train_data %>% filter(warehouse == "Frankfurt_1")
future_data0 <- future_data |> 
  filter(warehouse == "Frankfurt_1")

df <- bind_rows(future_data0, data) |> arrange(date)

predictions <- data.frame()

# Loop through each future date to make predictions

  # Extract the current date from future_data
  current_date <- future_data0$date[1]
  
  # Prepare the data for prediction
  data_warehouse <- df |> filter(date == current_date)
  
  # Make a prediction for the next time step
  next_prediction <- predict(finalized_workflow, new_data = data_warehouse)$.pred
  
  # Create a new row with the current date, warehouse, and prediction
  new_row <- future_data0[i, ] %>%
    mutate(orders = next_prediction,
           .pred = next_prediction)
  
  # Store the prediction
  predictions <- bind_rows(predictions, new_row)
  
  # Add the new prediction to the dataset
  df <- df |> filter(date != current_date) %>% bind_rows(., new_row) |> arrange(date)
  
  # Update lags and rolling averages
  df <- lag_roll_transformer(df) #transformer_func(df)
  
  
  print(data$orders_lag2)
  print(current_date)
  print(new_row$.pred)
  print(data$.pred)