library(tidyverse)
library(timetk)
library(lubridate)
library(ggplot2)

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
rohlik_train <- read_csv("data/train.csv")

rohlik_all <- bind_rows(
  rohlik_train,
  rohlik_test
) |> 
  select(-id)

summarize(rohlik_train,
          .by = warehouse,
          min_date = min(date),
          max_date = max(date))

rohlik_all |>
  filter(warehouse == "Budapest_1",
         date <= ymd("2024-03-15")) |> 
  mutate(month = ymd(paste0(year(date),"-",month(date),"-01")),
         day = wday(date))|> 
  select(date, user_activity_2, month, day) |> 
  summarize(.by = c(day, month),
            value = mean(user_activity_2)) #|> 
  plot_time_series(
    .date_var = month,
    .color_var = day,
    .value = value,
    .smooth = FALSE,
    .interactive = FALSE
  )
    

  rohlik_all |>
  filter(warehouse == "Brno_1",
         date <= ymd("2024-03-15")) |> 
  select(date, snow) |> 
  plot_seasonal_diagnostics(
    .date_var = date,
    .value = snow,
    .interactive = FALSE
  )

# anomaly detection

rohlik_all |> 
  filter(warehouse == "Frankfurt_1",
         date >= ymd("2022-02-01"),
         date <= ymd("2024-03-15")) |> 
  anomalize(.date_var = date,
                    .value = orders,
                    .iqr_alpha = 0.15,
                    .max_anomalies = 0.2) |>
  plot_anomalies(.date_var = date, .interactive = FALSE) +
  theme_minimal() +
  labs(title = NULL) +
  theme(legend.position = "none")

data_cleaned_munich <- rohlik_all |> 
  filter(warehouse == "Munich_1",
         date >= ymd("2021-07-21"),
         date <= ymd("2024-03-15")) |> 
  group_by(warehouse) |> 
  anomalize(
    .date_var      = date, 
    .value         = orders,
    .message       = FALSE
  ) |> 
  select(warehouse,
         date,
         orders = observed_clean) |> 
  mutate(orders = round(orders, 0))

data_cleaned_frankfurt <- rohlik_all |> 
  filter(warehouse == "Frankfurt_1",
         date >= ymd("2022-02-18"),
         date <= ymd("2024-03-15")) |> 
  group_by(warehouse) |> 
  anomalize(
    .date_var      = date, 
    .value         = orders,
    .iqr_alpha     = 0.15,
    .message       = FALSE
  ) |> 
  select(warehouse,
         date,
         orders = observed_clean) |> 
  mutate(orders = round(orders, 0))

data_cleaned_rest <- rohlik_all |> 
  filter(warehouse %in% c("Prague_1", "Prague_2", "Prague_3", "Budapest_1", "Brno_1"),
         date <= ymd("2024-03-15")) |> 
  group_by(warehouse) |> 
  anomalize(
    .date_var      = date, 
    .value         = orders,
    .iqr_alpha     = 0.15,
    .message       = FALSE
  ) |> 
  select(warehouse,
         date,
         orders = observed_clean) |> 
  mutate(orders = round(orders, 0))

data_cleaned <- bind_rows(
  data_cleaned_frankfurt,
  data_cleaned_munich,
  data_cleaned_rest
)

expanded_data <- full_time_series %>%
  left_join(rohlik_all, by = c("warehouse", "date"))

# Replace NA values with 0 for numeric/integer columns and NA for character columns
expanded_data <- expanded_data |> 
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

expanded_data <- expanded_data |> 
  left_join(rohlik_all |> 
              select(warehouse,
                     date,
                     holiday_name,
                     holiday,
                     shops_closed,
                     winter_school_holidays,
                     school_holidays),
            join_by(warehouse == warehouse, date == date),
            keep = FALSE)


expanded_data |>
  filter(warehouse == "Frankfurt_1",
         date <= ymd("2024-03-15")) |> 
  select(date, orders) |> 
  plot_time_series(
    .date_var = date,
    .value = orders,
    .smooth = FALSE,
    .interactive = FALSE
  )

write_rds(expanded_data, "data/expanded_data.RDS")
