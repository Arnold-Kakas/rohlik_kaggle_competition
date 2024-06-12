library(tidyverse)
library(timetk)
library(lubridate)
library(ggplot2)

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

summarize(calendar_test,
          .by = warehouse,
          min_date = min(date),
          max_date = max(date))

rohlik_test <- read_csv("data/test.csv")
rohlik_train <- read_csv("data/train.csv")

rohlik_all <- bind_rows(
  rohlik_train,
  rohlik_test
) |> 
  select(-id)

summarize(rohlik_test,
          .by = warehouse,
          min_date = min(date),
          max_date = max(date))

expanded_data <- full_time_series %>%
  left_join(rohlik_all, by = c("warehouse", "date"))

# Replace NA values with 0 for numeric/integer columns and NA for character columns
expanded_data <- expanded_data %>%
  mutate(across(where(is.numeric), ~replace_na(., 0))) %>%
  mutate(across(where(is.character), ~replace_na(., NA_character_))) |> 
  arrange(warehouse, date) |>  # Ensure data is sorted by warehouse and date
  group_by(warehouse) |> 
  mutate(
    orders_lag_1d = lag(orders, 1),
    orders_lag_3d = lag(orders, 3),
    orders_lag_7d = lag(orders, 7),
    orders_lag_14d = lag(orders, 14),
    orders_lag_28d = lag(orders, 28)
  ) |> 
  ungroup()

expanded_data |>
  filter(warehouse == "Prague_1",
         date <= ymd("2024-03-15")) |> 
  select(date, orders) |> 
  plot_time_series(
    .date_var = date,
    .value = orders,
    .smooth = FALSE,
    .interactive = FALSE
  )

expanded_data |>
  filter(warehouse == "Prague_1",
         date <= ymd("2024-03-15")) |> 
  select(date, orders) |> 
  plot_seasonal_diagnostics(
    .date_var = date,
    .value = orders,
    .interactive = FALSE,
    .geom_color = "steelblue"
)

expanded_data |>
  filter(warehouse == "Prague_1",
         date <= ymd("2024-03-15")) |> 
  tk_augment_timeseries_signature() |> 
  colnames()

# we will remove  "hour", "minute", "second", "hour12", "am.pm", "index.num"

