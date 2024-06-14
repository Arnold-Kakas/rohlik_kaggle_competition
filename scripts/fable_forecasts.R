library(tsibble)
library(fable)
library(feasts)
library(fabletools)
library(lubridate)
library(tidyverse)

df <- read_rds("data/full_data_cleaned.RDS") |> 
  mutate(orders = ifelse(orders < 0, 0, orders)) |> 
  filter(date <= ymd("2024-03-15"))

df_tsibble <- df |> 
  as_tsibble(key = warehouse,
             index = date,
             regular = TRUE) |>
  fill_gaps() |> 
  fill(orders, .direction = "down")

df_agg <- df_tsibble |> 
  aggregate_key(warehouse, orders)


df_tsibble |> 
  filter(warehouse == "Frankfurt_1") |> 
  model(
    STL(orders ~ season(period = 52) +
          season(period = 364),
        robust = TRUE)
  ) |>
  components() |>
  autoplot() + labs(x = "Observation")


df_tsibble |>
  filter(warehouse == "Prague_1") |> 
  model(model = my_dcmp_spec) |>
  forecast(h = "60 days") |>
  autoplot(df_tsibble) +
  labs(y = "Orders")
