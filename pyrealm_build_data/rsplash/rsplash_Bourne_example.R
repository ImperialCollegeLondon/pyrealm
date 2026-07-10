# install.packages(
#     "https://cran.r-project.org/src/contrib/Archive/topmodel/topmodel_0.7.5.tar.gz",
#     repos = NULL, type = "source"
# )

# if(!require(devtools)){install.packages(devtools)}
#     devtools::install_github( "dsval/rsplash")


library(rsplash)
library(xts)

data(Bourne)

# Export forcings
forcings <- Bourne$forcing
forcings <- data.frame(date = index(forcings), coredata(forcings))
forcings$elev <- Bourne$md$elev_m
forcings$lat <- Bourne$md$lat
write.csv(forcings, "rsplash_Bourne_inputs.csv")

# run splash
# - the data is modified to try and factor out the SPLASHv2 adaptation for
#   slope and aspect. Aspect should not matter if the slope is zero?
run1 <- splash.point(
    sw_in = Bourne$forcing$sw_in, # shortwave radiation W/m2
    tc = Bourne$forcing$Ta, # air temperature C
    pn = Bourne$forcing$P, # precipitation mm
    lat = Bourne$md$latitude, # latitude deg
    elev = Bourne$md$elev_m, # elevation masl
    slop = 0, # Bourne$md$slop_250m, # slope deg
    asp = Bourne$md$asp_250m, # aspect deg
    soil_data = Bourne$soil, # soil data
    Au = Bourne$md$Aups_250m, # upslope area m2
    resolution = 250.0 # resolution pixel dem used to get Au
)

# Export results
write.csv(run1, "rsplash_Bourne_outputs.csv")
