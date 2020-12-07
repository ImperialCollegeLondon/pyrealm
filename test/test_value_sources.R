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


# calc_ftemp_arrh (using KattgeKnorr ha value)

calc_ftemp_arrh( tc + 273.15, 71513)
[1] 0.611382245
> calc_ftemp_arrh( tc_mat + 273.15, 71513)
             [,1]        [,2]
[1,] 0.3674597823 0.611382245
[2,] 1.0000000000 1.609304720
>


# calc_ftemp_inst_rd

round(calc_ftemp_inst_rd(tc), 4)
[1] 0.6747

val = calc_ftemp_inst_rd(tc_mat)
> round(val, 4)
      [,1]   [,2]
[1,] 0.444 0.6747
[2,] 1.000 1.4456


# calc_ftemp_inst_vcmax

> calc_ftemp_inst_vcmax(tc)
[1] 0.6370757237
> calc_ftemp_inst_vcmax(tc_mat)
            [,1]         [,2]
[1,] 0.404673462 0.6370757237
[2,] 1.000000000 1.5427221126
>