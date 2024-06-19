library(tidyverse)

xgb <- read_csv("data/xgb_pred.csv") |> rename(xgb = orders)
lgbm <- read_csv("data/lgbm_pred.csv") |> rename(lgbm = orders)

submission <- xgb |> 
  left_join(lgbm, join_by(id == id), keep = FALSE) |> 
  mutate(orders = round((xgb*0.4 + lgbm*0.6), 0)) |> 
  select(id, orders)

write_csv(submission, "data/ensemble.csv")
