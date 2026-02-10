source("calc_phi0.R")

# Generate  all combinations of some simple input values
aridity_index <- c(0.2, 0.5, 0.8, 1.0, 5.0, 10.0)
temp <- seq(0, 50, by = 1)
mean_gdd_temp <- seq(5, 30, by = 5)

data <- expand.grid(
  temp = temp,
  aridity_index = aridity_index,
  mean_gdd_temp = mean_gdd_temp
)

# Run the reference implementation, which is not parallelised so need mapply
data$phio <- round(
  mapply(
    calc_phi0, data$aridity_index, data$temp, data$mean_gdd_temp
  ),
  digits = 8
)

# Save reference dataset.
write.csv(data, "sandoval_kphio.csv", row.names = FALSE)
