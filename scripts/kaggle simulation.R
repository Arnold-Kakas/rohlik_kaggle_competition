library(lubridate)
library(tidyverse)
library(tidymodels)
library(lightgbm) # lgbm
library(bonsai) # lgbm
library(poissonreg)
library(rstanarm)
library(parallelly)

# Load data
df <- read_rds("data/expanded_data.RDS") |> 
  mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0))) |> 
  mutate(orders = ifelse(is.na(orders) & date < ymd("2024-03-16"), 0, orders)) |> 
  mutate(across(where(is.character), ~ replace_na(., "No"))) |>  
  select(-c(shutdown,
            mini_shutdown,
            blackout,
            mov_change,
            frankfurt_shutdown,
            holiday_name))|> #user_activity_1,user_activity_2
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
         day_after_holiday = ifelse(holiday == 0 & lag(holiday, 1), 1, 0),
         is_weekend = if_else(wday %in% c(7, 1), 1, 0)) |> 
  ungroup() |> 
  mutate(across(where(is.numeric) & !contains("orders"), ~ replace_na(., 0)))
  


rohlik_test <- df |> 
  filter(date >= ymd("2024-03-16")) |> 
  select(-orders)

rohlik_train <- df |> 
  filter(date < ymd("2024-03-16"))

# Splitting data
set.seed(123)
data_split <- initial_split(rohlik_train, prop = 0.8, strata = warehouse)
train_data <- training(data_split)
test_data <- testing(data_split)

rec_obj <- recipe(orders ~ ., train_data) |> 
  step_rm(date) |> 
  step_novel() |> 
  step_zv(all_predictors()) |> 
  step_center(all_numeric_predictors()) |> 
  step_scale(all_numeric_predictors()) |> 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

# Model specifications with tuning parameters
xgboost_spec <- boost_tree(
  mode = "regression",
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("xgboost")

lightgbm_spec <- boost_tree(
  mode = "regression",
  trees = tune(),
  tree_depth = tune(),
  learn_rate = tune()
) %>%
  set_engine("lightgbm")

poissonreg_spec <- poisson_reg() |> 
  set_engine("glm")

# Model workflows
xgboost_wf <- workflow() %>%
  add_model(xgboost_spec) %>%
  add_recipe(rec_obj)

lightgbm_wf <- workflow() %>%
  add_model(lightgbm_spec) %>%
  add_recipe(rec_obj)

poissonreg_wf <- workflow() %>%
  add_model(poissonreg_spec) %>%
  add_recipe(rec_obj)

# Cross-validation
set.seed(123)
cv_splits <- vfold_cv(train_data, v = 5)

# Grid search
grid <- grid_latin_hypercube(
  trees(),
  tree_depth(),
  learn_rate(),
  size = 20
)

# Tune models
xgboost_res <- tune_grid(
  xgboost_wf,
  resamples = cv_splits,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

lightgbm_res <- tune_grid(
  lightgbm_wf,
  resamples = cv_splits,
  grid = grid,
  control = control_grid(save_pred = TRUE)
)

# Select best models
best_xgboost <- select_best(xgboost_res)
best_lightgbm <- select_best(lightgbm_res)


# Finalize workflows with best hyperparameters
final_xgboost_wf <- finalize_workflow(xgboost_wf, best_xgboost)
final_lightgbm_wf <- finalize_workflow(lightgbm_wf, best_lightgbm)

# Train final models
final_xgboost_fit <- fit(final_xgboost_wf, data = train_data)
final_lightgbm_fit <- fit(final_lightgbm_wf, data = train_data)
final_poissonreg_fit <- fit(poissonreg_wf, data = train_data)

# Make predictions
xgboost_pred <- predict(final_xgboost_fit, new_data = test_data) %>% pull(.pred)
lightgbm_pred <- predict(final_lightgbm_fit, new_data = test_data) %>% pull(.pred)
poissonreg_pred <- predict(final_poissonreg_fit, new_data = test_data) %>% pull(.pred)


prediction_stage_1 <- bind_cols(xgboost_pred, lightgbm_pred, poissonreg_pred, test_data$orders)

colnames(prediction_stage_1) <- c("xgb", "lgbm", "poiss", "reality")

prediction_stage_1 <- prediction_stage_1 |> 
  mutate(xgb_mape = abs(round(xgb/reality-1, 4)),
         lgbm_mape = abs(round(lgbm/reality-1, 4)),
         poiss_mape = abs(round(poiss/reality-1, 4)))

summarize(prediction_stage_1,
          xgb_mape = mean(xgb_mape),
          lgbm_mape = mean(lgbm_mape),
          poiss_mape = mean(poiss_mape))

# Ensemble model ----

ensemble_spec <- linear_reg() %>% 
  set_engine("lm")

ensemble_wf <- workflow() %>%
  add_model(ensemble_spec) %>%
  add_formula(reality ~ .)

set.seed(123)
data_split_ensemble <- initial_split(prediction_stage_1 |> select(-contains("mape")), prop = 0.8)
train_data_ensemble <- training(data_split_ensemble)
test_data_ensemble <- testing(data_split_ensemble)

final_ensemble_fit <- fit(ensemble_wf, data = train_data_ensemble)

#ensemble_pred_final <- NULL
ensemble_pred_final <- predict(final_ensemble_fit, new_data = test_data_ensemble) %>% pull(.pred)

prediction_analysis <- bind_cols(ensemble_pred_final, test_data_ensemble$reality)

colnames(prediction_analysis) <- c("ensemble", "reality")

prediction_analysis <- prediction_analysis |> 
  mutate(mape = abs(round(ensemble/reality-1, 4)))

summarize(prediction_analysis,
          mape = mean(mape))


# predict rohlik test ----
xgboost_pred_sub <- predict(final_xgboost_fit, new_data = rohlik_test) %>% pull(.pred)
lightgbm_pred_sub <- predict(final_lightgbm_fit, new_data = rohlik_test) %>% pull(.pred)
poissonreg_pred_sub <- predict(final_poissonreg_fit, new_data = rohlik_test) %>% pull(.pred)

prediction_stage_1_sub <- bind_cols(xgboost_pred_sub, lightgbm_pred_sub, poissonreg_pred_sub)

colnames(prediction_stage_1_sub) <- c("xgb", "lgbm", "poiss")

ensemble_pred_final <- predict(final_ensemble_fit, new_data = prediction_stage_1_sub) %>% pull(.pred)

ensemble_pred_final_df <- bind_cols(ensemble_pred_final, rohlik_test$warehouse, rohlik_test$date)
colnames(ensemble_pred_final_df) <- c("orders", "warehouse", "date")
ensemble_pred_final_df <- ensemble_pred_final_df |> 
  mutate(id = paste0(warehouse, "_", date),
         orders = round(orders,0)) |> 
  select(id, orders)

write_csv(ensemble_pred_final_df, "data/ensemble_pred_kaggle.csv")
