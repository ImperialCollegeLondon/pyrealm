source("calc_phi0.R")

aridity_index <- c(0.2, 0.5, 0.8, 1.0, 5.0, 10.0)
temp <- seq(0, 50, by = 1)
mean_gdd_temp <- seq(5, 30, by = 5)

data <- expand.grid(
    temp = temp,
    aridity_index = aridity_index,
    mean_gdd_temp = mean_gdd_temp
)

# Function is not parallelised so loop over inputs
data$phio <- NA

for (row_idx in seq_along(data$aridity_index)) {
    data$phio[row_idx] <- with(
        data[row_idx, ],
        calc_phi0(AI = aridity_index, tc = temp, mGDD0 = mean_gdd_temp)
    )
}

data$phio <- round(data$phio, digits = 8)
write.csv(data, "sandoval_kphio.csv", row.names = FALSE)
