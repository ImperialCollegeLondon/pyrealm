library(bigleaf)

# calc_vp_stat

Esat.slope(21)
#   Esat     Delta
# 1 2.480904 0.152347

Esat.slope(21, 'Allen_1998')
#   Esat     Delta
# 1 2.487005 0.1527629

Esat.slope(21, 'Alduchov_1996')
#   Esat     Delta
# 1 2.481888 0.1524929

# vp_to_vpd

e.to.VPD(1.9, 21)
# [1] 0.5809042
e.to.VPD(1.9, 21, 'Allen_1998')
# [1] 0.5870054

# rh_to_vpd

rH.to.VPD(0.7, 21)
# [1] 0.7442712
rH.to.VPD(0.7, 21,  'Allen_1998')
# [1] 0.7461016

# sh_to_vp

q.to.e(0.006, 99.024)

# sh_to_vpd

q.to.VPD(0.006, 21, 99.024)
# [1] 1.529159
q.to.VPD(0.006, 21, 99.024,  'Allen_1998')
# [1] 1.535260


vp1 <- seq(0, 20, length=61)
ta1 <- seq(0, 60, length=61)

vp2 <- 
