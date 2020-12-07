library(rpmodel)

tc <- 20
p <- 101325

tc_mat <- matrix(c(15,20,25,30), ncol=2, byrow=TRUE)
p_mat <- matrix(c(100325, 101325, 102325, 103325), ncol=2, byrow=TRUE)

# calc_density_h20

round(calc_density_h2o(tc, p), 4)
[1] 998.2056

round(calc_density_h2o(tc, p_mat), 4)
         [,1]     [,2]
[1,] 998.2052 998.2056
[2,] 998.2061 998.2066


round(calc_density_h2o(tc_mat, p_mat), 4)
         [,1]     [,2]
[1,] 999.1006 998.2056
[2,] 997.0475 995.6515
