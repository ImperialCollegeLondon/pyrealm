
# C3 and C4 Chi models
oc_c3 <-  rpmodel:::calc_optimal_chi(kmm = 46.09928, gammastar = 3.33925,
                                	 ns_star = 1.12536, ca = 40.53, vpd = 1000, beta=146.0)
oc_c4 <- rpmodel:::calc_chi_c4()
                                	 

# C3 Chis into Jmax
jmax_c3_c4 <- rpmodel:::calc_lue_vcmax_c4(kphio = 0.081785, ftemp_kphio = 0.656,
                            			  soilmstress = 1, c_molmass=12.0107)

jmax_c3_none <- rpmodel:::calc_lue_vcmax_none(oc_c3, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  soilmstress = 1, c_molmass=12.0107)

jmax_c3_wang17 <- rpmodel:::calc_lue_vcmax_wang17(oc_c3, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  	soilmstress = 1, c_molmass=12.0107)

jmax_c3_smith19 <- rpmodel:::calc_lue_vcmax_smith19(oc_c3, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  	soilmstress = 1, c_molmass=12.0107)


# C4 Chis into Jmax
jmax_c4_c4 <- rpmodel:::calc_lue_vcmax_c4(kphio = 0.081785, ftemp_kphio = 0.656,
                            			  soilmstress = 1, c_molmass=12.0107)

jmax_c4_none <- rpmodel:::calc_lue_vcmax_none(oc_c4, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  soilmstress = 1, c_molmass=12.0107)

jmax_c4_wang17 <- rpmodel:::calc_lue_vcmax_wang17(oc_c4, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  	soilmstress = 1, c_molmass=12.0107)

jmax_c4_smith19 <- rpmodel:::calc_lue_vcmax_smith19(oc_c4, kphio = 0.081785, ftemp_kphio = 0.656,
            	                			  	soilmstress = 1, c_molmass=12.0107)



rbind(jmax_c4_c4, jmax_c4_none, jmax_c4_wang17, jmax_c4_smith19,
	  jmax_c3_c4, jmax_c3_none, jmax_c3_wang17, jmax_c3_smith19)