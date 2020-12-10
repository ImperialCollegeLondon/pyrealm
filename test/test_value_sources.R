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


# Values taken from internals of example(rpmodel)
kmm <- 46.09928
gammastar <- 3.33925
ns_star <- 1.12536
ca <- 40.53
vpd <- 1000

prop_mat <- matrix(c(0.95, 1.0, 1.05, 1.1), ncol=2)
kmm_mat <- kmm * prop_mat
gammastar_mat <- gammastar * prop_mat
ns_star_mat <- ns_star * prop_mat
ca_mat <- ca * prop_mat
vpd_mat <- vpd * prop_mat


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

# rpmodel:::calc_chi_c4()
# NOTE: this function doesn't do anything but return 1.0 scalars, but it
# doesn't need to do anything else. There is no need to capture the input shape.

> rpmodel:::calc_chi_c4()
$chi
[1] 1

$mc
[1] 1

$mj
[1] 1

$mjoc
[1] 1

> rpmodel:::calc_optimal_chi(kmm, gammastar, ns_star, ca, vpd, beta=146.0)
$chi
[1] 0.69435213

$mc
[1] 0.33408383

$mj
[1] 0.71230386

$mjoc
[1] 2.13211114

> rpmodel:::calc_optimal_chi(kmm_mat, gammastar, ns_star_mat, ca, vpd, beta=146.0)
$chi
           [,1]       [,2]
[1,] 0.69471370 0.69402371
[2,] 0.69435213 0.69372406

$mc
           [,1]       [,2]
[1,] 0.34492189 0.32390633
[2,] 0.33408383 0.31433074

$mj
           [,1]       [,2]
[1,] 0.71242488 0.71219384
[2,] 0.71230386 0.71209338

$mjoc
          [,1]      [,2]
[1,] 2.0654673 2.1987648
[2,] 2.1321111 2.2654271


> rpmodel:::calc_optimal_chi(kmm_mat, gammastar_mat, ns_star_mat, ca_mat, vpd_mat, beta=146.0)
$chi
           [,1]       [,2]
[1,] 0.69955736 0.68935938
[2,] 0.69435213 0.68456214

$mc
           [,1]       [,2]
[1,] 0.33597077 0.33226381
[2,] 0.33408383 0.33050567

$mj
           [,1]       [,2]
[1,] 0.71403643 0.71062217
[2,] 0.71230386 0.70898771

$mjoc
          [,1]      [,2]
[1,] 2.1252933 2.1387287
[2,] 2.1321111 2.1451605
