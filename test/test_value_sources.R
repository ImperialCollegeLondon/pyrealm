library(rpmodel)

# Inputs
tc <- 20
p <- 101325
soilm <- 0.2
meanalpha <- 1
elev <- 1000
co2 <- 413.03

tc_mat <- matrix(c(15,20,25,30), ncol=2, byrow=TRUE)
p_mat <- matrix(c(100325, 101325, 102325, 103325), ncol=2, byrow=TRUE)
soilm_mat <- matrix(c(0.1, 0.2, 0.5, 0.7), ncol=2, byrow=TRUE)
meanalpha_mat <- matrix(c(0.2, 1.0, 0.5, 0.7), ncol=2, byrow=TRUE)
elev_mat <- matrix(c(900, 1000, 1100, 1200), ncol=2, byrow=TRUE)
co2_mat <- matrix(c(373.03, 393.03, 413.03, 433.03), nrow=2, byrow=TRUE)


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

# calc_ftemp_kphio

> calc_ftemp_kphio(tc)
[1] 0.656
> calc_ftemp_kphio(tc_mat)
       [,1]  [,2]
[1,] 0.6055 0.656
[2,] 0.6895 0.706

> calc_ftemp_kphio(tc, TRUE)
[1] 0.0438
> calc_ftemp_kphio(tc_mat, TRUE)
       [,1]   [,2]
[1,] 0.0352 0.0438
[2,] 0.0495 0.0523
>

# calc_gammastar

> calc_gammastar(tc, p)
[1] 3.3392509
> calc_gammastar(tc_mat, p)
          [,1]      [,2]
[1,] 2.5508606 3.3392509
[2,] 4.3320000 5.5718448
> calc_gammastar(tc_mat, p_mat)
          [,1]      [,2]
[1,] 2.5256856 3.3392509
[2,] 4.3747535 5.6818245
>

# calc_kmm

> calc_kmm(tc, p)
[1] 46.099278
> calc_kmm(tc_mat, p)
          [,1]       [,2]
[1,] 30.044262  46.099278
[2,] 70.842252 108.914368
> calc_kmm(tc_mat, p_mat)
          [,1]       [,2]
[1,] 29.877494  46.099278
[2,] 71.146937 109.725844

# calc_soilmstress
# This is **not vectorised** in rpmodel 1.0.6

> calc_soilmstress(soilm, meanalpha)
[1] 0.86

> matrix(mapply(calc_soilmstress, soilm_mat, meanalpha), ncol=2)
        [,1]    [,2]
[1,] 0.78125  0.86000
[2,] 0.99125  1.00000
>
> matrix(mapply(calc_soilmstress, soilm_mat, meanalpha_mat), ncol=2)
           [,1]       [,2]
[1,] 0.40069444 0.86000000
[2,] 0.98173611 1.00000000


# calc_viscosity_h2o

> calc_viscosity_h2o(tc, p)
[1] 0.0010015972
> # NOTE using p_mat here -
> calc_viscosity_h2o(tc, p_mat)
             [,1]         [,2]
[1,] 0.0010015975 0.0010015972
[2,] 0.0010015968 0.0010015965
>
> calc_viscosity_h2o(tc_mat, p_mat)
              [,1]          [,2]
[1,] 0.00113756998 0.00100159716
[2,] 0.00089002254 0.00079722171

# calc_patm

> calc_patm(elev)
[1] 90241.542
> calc_patm(elev_mat)
          [,1]      [,2]
[1,] 91303.561 90241.542
[2,] 89189.548 88147.507


# co2_to_ca
# Not exported

> rpmodel:::co2_to_ca(co2, p)
[1] 41.850265
> rpmodel:::co2_to_ca(co2_mat, p)
          [,1]      [,2]
[1,] 37.797265 39.823765
[2,] 41.850265 43.876765
> rpmodel:::co2_to_ca(co2_mat, p_mat)
          [,1]      [,2]
[1,] 37.424235 39.823765
[2,] 42.263295 44.742825
>
