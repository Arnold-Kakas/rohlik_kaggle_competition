library(tsibble)
library(fable)
library(feasts)
library(fabletools)
library(lubridate)

df <- read_rds("data/full_data_cleaned.RDS") |> 
  mutate(orders = ifelse(orders < 0, 0, orders)) |> 
  filter(date <= ymd("2024-03-15"))

df_tsibble <- df |> 
  as_tsibble(key = warehouse,
             index = date,
             regular = TRUE)|>
  fill_gaps(orders = 0) 

df_agg <- df_tsibble |> 
  aggregate_key(warehouse, orders)


df_tsibble |> 
  filter(warehouse == "Munich_1") |> 
  model(
    STL(orders ~ season(period = 52) +
          season(period = 365),
        robust = TRUE)
  ) |>
  components() |>
  autoplot() + labs(x = "Observation")

