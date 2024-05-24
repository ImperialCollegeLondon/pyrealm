#################################################################################
### kjb workings 2024, incorporating a SIMPLE canopy radiative transfer model ###
#################################################################################

install.packages('tidyverse')
install.packages('devtools')
install.packages('solrad')
install.packages('covr')
install.packages('markdown')

# This did not install cleanly and needed extensive fixes to rebuild
devtools::install_github("vflo/rpmodel_lt")

# After several rounds of discussion, we try to follow the workings of the BESS model
# (Ryu et al. 2011) which draws heavily on de Pury & Farquhar 1997): 
# TWO LEAF: sun and shade groups TWO STREAM: direct beam and diffuse light

# June 2023, Victor Flo Sierra has written an R function for irradiance partitioning
# based on the worked example provided in deP&F.

# To aid comparisons we draw on the same FLUXNET2015 subset as presented in Mengoli et
# al. (2021).  


# Our first step is to generate a sub-daily tibble for input to the P-model function.

library(tidyverse)
library(lubridate)
library(solrad)
library(rpmodellt)

### We have the FLUXNET file from Giulia in csv format

df_Vie_hh <- read_csv(file = "Mengoli et al 2018_input-data/2014/FLX_BE-Vie_2014.csv",
                      show_col_types = FALSE) %>% 
  
  # render the time-stamp field more tractable
  mutate(step = ymd_hm(TIMESTAMP_START),
         date = as_date(step),
         month = lubridate::month(date),
         sitename = "BE-Vie") %>% 
  
  # set our desired date-range, here August
  filter(month == 8) %>% 
  
  # apply quality control filter to exclude medium to poor records:
  filter(NEE_CUT_REF_QC <= 1, VPD_F_QC <= 1, TA_F_QC <= 1, CO2_F_MDS_QC <= 1, PA_F_QC <= 1) %>% 
  
  # sort out our units for PA (here kPa) and VPD (here hPa); I'm going to leave PPFD as µmol Photons:
  mutate(vpd = VPD_F * 100,
         patm = PA_F * 1000) %>% 
  
  # select the variables we need for the P-model function
  select(sitename, date, step, PPFD_IN, vpd, TA_F, CO2_F_MDS, patm, GPP_DT_CUT_REF) %>% 
  
  # adopt the input names recognised by rpmodel:
  rename(tc = TA_F, ppfd = PPFD_IN, co2 = CO2_F_MDS, gpp_obs = GPP_DT_CUT_REF)


## We have daily fAPAR (MODIS) from Beni's workings so I can combine them here (ignoring sub-daily variations)

df_Vie_beni <- read_csv(file = "Mengoli et al 2018_input-data/2014/data_Beni2_BE-Vie_2014.csv", 
                        show_col_types = FALSE) %>% 
  mutate(date = lubridate::ymd(TIME)) %>% 
  select(c(date, fapar_spl))

## We also have half-hourly LAI (MODIS) estimates downloaded from Giulia's github repository

df_Vie_lai <- read_csv(file = "Mengoli et al 2018_input-data/2014/data_LAI_BE-Vie_2014.csv", 
                        show_col_types = FALSE) %>% 
  mutate(step = lubridate::ymd_hm(TIMESTAMP_START)) %>% 
  select(c(step, LAI))


# and we use the join argument to merge these separate tibbles

df_Vie_hh <- df_Vie_hh %>% 
  left_join(df_Vie_beni, by = "date") %>% 
  left_join(df_Vie_lai, by = "step") %>% 
  rename(fapar = fapar_spl)



# let's have a look at that - are the values reasonable?

summary(df_Vie_hh)

# no obvious problems there. 

######
## We'll start with the existing Big-Leaf representation of the P-model, but at the half-hourly time-step:
######

library(rpmodellt) # from Victor, based on the P-model with acclimation presented in Mengoli et al. 2021


### We use this run to generate big-leaf Vcmax estimates relating to the top of the canopy.  
## from Julia's workings these estimates are acclimated - as based on a weighted mean for the antecedent period (here 15 days) and optimised for noon conditions:


df_output <- as_tibble(rpmodel_subdaily(TIMESTAMP = df_Vie_hh$step, tc = df_Vie_hh$tc, 
                                          vpd = df_Vie_hh$vpd, co2 = df_Vie_hh$co2, ppfd = df_Vie_hh$ppfd, 
                                          patm =df_Vie_hh$patm, fapar = df_Vie_hh$fapar, 
                        # optional extras
                        # in particular notice the acclimation options (time-frame, averaging etc.)
                        LAI = NA,  
                        elv = 493, 
                        u=NA, # windspeed
                        ustar = NA,
                        canopy_height = NA, 
                        z = NA, leafwidth = NA,
                        netrad = NA, 
                        beta = 146.0, c_cost = 0.41,
                        do_leaftemp = FALSE,  gb_method = "simple", #Choudhury_1988"
                        do_acclimation = TRUE,
                        upscaling_method = "noon", hour_reference_T = 12,
                        acclim_days = 15, weighted_accl = TRUE, 
                        epsleaf = 0.96, #thermal absorptivity of the leaf
                        energy_params = list(
                          ste_bolz = 5.67e-8, #W m^-2 K^-4
                          cpm = 75.38, #J mol^-1 ºC-1
                          J_to_mol = 4.6, #Conversion factor from J m-2 s-1 (= W m-2) to umol (quanta) m-2 s-1
                          frac_PAR = 0.5, #Fraction of incoming solar irradiance that is PAR
                          fanir = 0.35 #Fraction of NIR absorbed
                          )
                        ))


## So we have a tibble with 19 variables generated by the P-model run.  


# and I combine that (in a very generic way) with the starting dataframe; and drop those simulated variables that we won't use subsequently

df_input <- df_Vie_hh %>% 
  cbind(df_output) %>% 
  select(-c(gpp, ca, chi, xi, mj, mc, iwue, gs, e, vpd_leaf, ns_star)) %>% 
  # I'll attach a suffix to the rpmodel generated variables to avoid any confusion
  # here I opt for 'assim' rather than 'gpp' for consistency of units
  rename(gpp_pmod = assim, gammastar_pmod = gammastar, kmm_pmod = kmm, ci_pmod = ci,
         vcmax_pmod = vcmax, jmax_pmod = jmax, vcmax25_pmod = vcmax25, rd_pmod =rd) 
  

## we now generate extinction coefficients to express the vertical gradient in photosynthetic capacity after the equation provided in Figure 10 of Lloyd et al. (2010):

df_input <- df_input %>% 
  mutate(kv_Lloyd = exp(0.00963 * vcmax_pmod - 2.43)) # Here I opt for vcmax rather than Vcmax25 


######
### Next the radiance partitioning after de Pury & Farquhar; 
## for now we ignore the clumping index term introduced in BESS
######

## we use a function provided by Victor Flo Sierra:  
# A number of parameters are provided from the deP&F paper (e.g. Tables 2 and 3):
# fa: scattering coefficient of PAR in the atmosphere
# sigma: leaf scattering coefficient of PAR (reflection and transmissivity)
# rho_cd: canopy reflection coefficient for diffuse PAR
# kd_prime: diffuse and scattered diffuse PAR extinction coefficient


library(solrad) # a CRAN package used for calculating solar radiation and related variables


irradiance_partition <- function(sun_shade = c("sun","shade"), TIMESTAMP, PPFD, LAI, PA, 
                                 PA0 = 101325, fa = 0.426, sigma = 0.15, 
                                 rho_cd = 0.036, kd_prime = 0.719, lat, long){
  
  stopifnot("TIMESTAMP must be a POSIXct, POSIXtl or a Date object" = all(lubridate::is.instant(TIMESTAMP)))
  if(any(PA <10000)) warning("Check if PA is in Pascal")
  if(any(PPFD > 3000)) warning("Check if PPFD is in micromol m-2 s-1")
  
  #Solar elevation angle
  DOY <- lubridate::yday(TIMESTAMP)
  hour <- lubridate::hour(TIMESTAMP)
  DOY_dec <- solrad::DayOfYear(TIMESTAMP)
  beta_angle <- solrad::Altitude(DOY = DOY_dec, 
                                 Lat = lat,
                                 Lon = long,
                                 SLon = long,
                                 DS = 0) * pi/180 # converting degrees to radians
  # beam extinction coefficient
  kb = 0.5/sin(beta_angle)
  # I overwrite Victor's step here, if beta is small (say 5°), then kb should be large
  #kb = ifelse(beta_angle<=0, 1e-10, kb)
  kb <- if_else(beta_angle * (180/pi) <= 1, 30, kb) # a beta of 1° would deliver a kb of >28
  # beam and scattered beam extinction coefficient
  kb_prime = 0.46/sin(beta_angle)
  #kb_prime = ifelse(beta_angle<=0, 1e-10,kb_prime )
  kb_prime <- if_else(beta_angle * (180/pi) <= 1, 27, kb_prime)
  # fraction of diffuse radiation
  m = (PA/PA0)/sin(beta_angle)
  fd = (1-0.72^m)/(1+0.72^m*(1/fa - 1))
  # beam irradiance horizontal leaves
  rho_h = (1-(1-sigma)^0.5)/(1+(1-sigma)^0.5)
  # beam irradiance uniform leaf-angle distribution
  rho_cb = (1-exp(-2*rho_h*kb/(1+kb)))
  # diffuse irradiance
  I_d = PPFD*fd
  I_d = ifelse(I_d<0,0,I_d)
  # beam irradiance
  I_b = PPFD*(1-fd)
  # scattered beam irradiance
  I_bs = I_b*((1-rho_cb)*kb_prime*exp(-kb_prime*LAI)-(1-sigma)*kb*exp(-kb*LAI))
  # Irradiance sun exposed
  I_c = (1-rho_cb)*I_b*(1-exp(-kb_prime*LAI))+(1-rho_cd)*I_d*(1-exp(-kd_prime*LAI))
  
  a = I_b*(1-sigma)*(1-exp(-kb*LAI))
  b = I_d*(1-rho_cd)*(1-exp(-(kd_prime+kb)*LAI))*kd_prime/(kd_prime+kb)
  c = I_b*(((1-rho_cb)*(1-exp(-(kb_prime+kb)*LAI))*(kb_prime/(kb_prime+kb)))-
             (1-sigma)*((1-exp(-2*kb*LAI)))/2)
  
  Isun = a+b+c
  
  # I introduce a clause here to exclude hours of obscurity (here a beta_angle > 1°, expressed in Radians)
  Ishade = if_else(beta_angle > 0.02,
                   I_c - Isun,
                   0)
  
  # don't quite understand this bit about sun_shade, replace for now with a list of the two fractions
  #if(sun_shade == "sun"){return(Isun)}else
    #if(sun_shade == "shade"){return(Ishade)}else{"sun_shade variable must be either sun or shade"}
  
  output <- list(
    Icsun = Isun,
    Icshade = Ishade,
    sEa = beta_angle, # solar elevation angle (aka solar altitude angle)
    kb = kb # beam irradiance extinction coefficient
    )
  
  return(output)
  
  }


## and we can run that partitioning function supplying inputs from our mini BE-Vie dataset


df_dePF <- as_tibble(irradiance_partition(TIMESTAMP = df_input$step, PPFD = df_input$ppfd, 
                                          PA = df_input$patm, LAI = df_input$LAI, 
                                          lat = 50.30493, long = 5.99812))


# and again, I combine that with the starting dataframe

df_input <- df_input %>% 
  cbind(df_dePF) 


#######
### Canopy photosynthesis
### So the two-leaf, two-stream model has separate calculations for the two fractions: sun and shade
######

## we derive separate photosynthetic parameters for the two groups, following BESS (or Mengoli) formulations:

df_input <- df_input %>% 
  
  # start with carboxylation
  mutate(Vmax25_canopy = LAI * vcmax25_pmod * ((1 - exp(-kv_Lloyd)) / kv_Lloyd),
         Vmax25_sun = LAI * vcmax25_pmod * ((1 - exp(-kv_Lloyd - kb * LAI))/ (kv_Lloyd + kb * LAI)),
         Vmax25_shade = Vmax25_canopy - Vmax25_sun,
         
         # next we convert those sun and shade values for ambient temperature using an Arrhenius function
         Vmax_sun = Vmax25_sun*exp(64800*(tc-25)/(298*8.314*(tc+273))),
         Vmax_shade = Vmax25_shade*exp(64800*(tc-25)/(298*8.314*(tc+273))),
         
         # now the photosynthetic estimates
         Av_sun = Vmax_sun * (ci_pmod - gammastar_pmod)/(ci_pmod + kmm_pmod),
         Av_shade = Vmax_shade * (ci_pmod - gammastar_pmod)/(ci_pmod + kmm_pmod), 
         
         ## and now separate Jmax estimates for sun and shade;
         
         Jmax25_sun = 29.1 + 1.64 * Vmax25_sun, # Eqn 31, after Wullschleger
         Jmax25_shade = 29.1 + 1.64 * Vmax25_shade,
         
         #phi0 = 0.352/8 + 0.022*tc - 0.00034*tc^2,
         
         #Jmax25_sun = (4 * phi0 * Icsun) / 
           #sqrt((1 / (1 - (0.41*(ci_pmod + 2*gammastar_pmod)/ (ci_pmod - gammastar_pmod))^(2/3))) - 1),
         #Jmax25_shade = (4 * phi0 * Icshade) / 
           #sqrt((1 / (1 - (0.41*(ci_pmod + 2*gammastar_pmod)/ (ci_pmod - gammastar_pmod))^(2/3))) - 1),
         
         
         # temperature correction (Mengoli 2021 Eqn 3b); relevant temperatures given in Kelvin
         Jmax_sun = Jmax25_sun * exp((43990/8.314) * (1/298 - 1/(tc+273))),
         Jmax_shade = Jmax25_shade * exp((43990/8.314) * (1/298 - 1/(tc+273))),
         
         # and now to calculate J and Aj for each group
         
         #J_sun = (4 * phi0 * Icsun) / sqrt(1 + ((4 * phi0 * Icsun) / Jmax_sun)^2),
         #J_shade = (4 * phi0 * Icshade) / sqrt(1 + ((4 * phi0 * Icshade) / Jmax_shade)^2),
         
         J_sun = (Jmax_sun * Icsun * (1 - 0.15) / (Icsun + 2.2 * Jmax_sun)),
         J_shade = (Jmax_shade * Icshade * (1 - 0.15) / (Icshade + 2.2 * Jmax_shade)),
         
         
         Aj_sun = (J_sun / 4) * (ci_pmod - gammastar_pmod) / (ci_pmod + 2 * gammastar_pmod),
         Aj_shade = (J_shade / 4) * (ci_pmod - gammastar_pmod) / (ci_pmod + 2 * gammastar_pmod),

         
         # gross assimilation for each class of leaves is the minimum of the two rates
         # again restricted to daylight hours (solar Elevation angle > 1°, here in Radians)
         Acanopy_sun = if_else(sEa > 0.02, 
                               if_else(Av_sun > Aj_sun, Aj_sun, Av_sun),
                               0),
         Acanopy_shade = if_else(sEa > 0.02, 
                                 if_else(Av_shade > Aj_shade, Aj_shade, Av_shade),
                                 0),
         
         # and we account for canopy respiration
         gpp_canopy = if_else(sEa > 0.02,
                              Acanopy_sun + Acanopy_shade - (rd_pmod * LAI),
                              0))



######
## let's have a look at that; and we try to recreate the time-series plots per Mengoli et al. (e.g. their Figure 2)
######

# a standard plotting argument for formatting axis labels:
kj.lab_size <- theme(axis.title.x=element_text(size=15), 
                     axis.title.y=element_text(size=15), 
                     axis.text.x=element_text(size=10), 
                     axis.text.y=element_text(size=10))



df_input %>% 
  filter(date > "2014-08-19", date < "2014-08-23") %>% 
  ggplot(aes(x = step)) +
  geom_line(mapping = aes(y = gpp_obs, colour = "FLUXNET_ecv"), linewidth = 1) +
  geom_line(mapping = aes(y = gpp_pmod, colour = "Mengoli_2021"), linewidth = 1) +
  geom_line(mapping = aes(y = gpp_canopy, colour ="BESS_lite"), linewidth = 1) +
  labs(title = "Vielsalm, Belgium 2014",
       x = "Timestep",
       y = expression("GPP"~ (µmol~CO[2]~m^{-2}~s^{-1}))) +
  kj.lab_size +
  guides(colour = guide_legend(title = NULL)) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 15))




boxplot(df_input$Vmax_shade, main = "vmax_shade")

# so we are estimating higher Vmax for shade than sun???

df_input %>% 
  filter(date > "2014-08-19", date < "2014-08-23") %>% 
  ggplot(aes(x = step)) +
  geom_line(mapping = aes(y = Vmax_sun, colour = "sun_canopy"), linewidth = 1) +
  geom_line(mapping = aes(y = Vmax_shade, colour = "shade_canopy"), linewidth = 1) +
  geom_line(mapping = aes(y = vcmax_pmod, colour = "Julia_leaf"), linewidth = 1) +
  labs(title = "Vielsalm, Belgium 2014",
       x = "Timestep",
       y = expression("Vcmax"~ (µmol~CO[2]~m^{-2}~s^{-1}))) +
  kj.lab_size +
  guides(colour = guide_legend(title = NULL)) +
  theme(legend.position = "bottom",
        legend.text = element_text(size = 15))


# contrast this with Fig 10 in deP & F where the canopy level is constant (in BESS this could be coming from a look-up table by PFT).  Our big-leaf estimates (red-line) are dynamic showing acclimation to antecedent conditions.

# Here Vc_shade is NEVER lower than Vc_sun.  Why???

# for our period here (BE-Vie) in August the solar Elevation angle reaches about 50° in the middle of the day.  That translates to a kb of ca 0.65.  That combined with our low kv of 0.25 (that serves to bolster total canopy capacity) means that at best the Vcsun/Vcshade ratio approaches unity.  Hence the kissing curves around midday. 

df_input %>% 
  filter(date > "2014-08-19", date < "2014-08-23") %>% 
  ggplot(aes(x = step)) +
  geom_line(mapping = aes (y = sEa*180/pi)) 
  #geom_line(mapping = aes (y = Icsun, colour = "Irradiance absorbed by sun leaves")) +
  #geom_line(mapping = aes (y = Icshade, colour = "Irradiance absorbed by shade leaves")) +
  #geom_line(mapping = aes (y = Jmax_sun/Vmax_sun)) + 
  #geom_line(mapping = aes(y = kb, colour = "beam_extinction-coefficient"), linewidth = 1)


######
### Performance metrics; the time-series plot of canopy GPP looks encouraging for this BESS-implementation.  How about the usual Rsquared and RMSE indicators?
######

source("../../kjb workings/functions/analyse_modobs.R")

with(df_input, analyse_modobs(gpp_canopy, gpp_obs, heat = T, 
                            plot.title = "Model performance metrics",
                            xlab = "Simulated GPP (BESS_deP&F)",
                            ylab = expression("Inferred GPP"~ (µmol~CO[2]~m^{-2}~s^{-1}))
                            )
     )
                            

# So this reinforces that here I have a problem with early morning simulations where the respiration term is pulling us into negative territory





