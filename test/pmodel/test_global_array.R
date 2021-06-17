
library(ncdf4)
library(rpmodel)

# Load the model inputs from netCDF file
ds <- nc_open('../../data/pmodel_inputs.nc')

temp <- ncvar_get(ds, 'temp')
co2 <- ncvar_get(ds, 'CO2')
elev <- ncvar_get(ds, 'elevation')
vpd <- ncvar_get(ds, 'VPD')
fapar <- ncvar_get(ds, 'fAPAR')
ppfd <- ncvar_get(ds, 'ppfd')


# Run the model with no temperature effect on lue
model <- rpmodel(tc=temp, vpd=vpd, co2=co2, fapar=fapar,
                 ppfd=ppfd, elv=elev, kphio=0.05, do_ftemp_kphio=FALSE)

# Save the GPP output to a new netCDF file
gpp_var <- ncvar_def('gpp','', ds$dim)
gpp_file <- nc_create('rpmodel_global_gpp_no_ftkphio.nc', gpp_var)
ncvar_put(gpp_file, gpp_var, model$gpp)

# Tidy up
nc_close(gpp_file)

# Run the model with temperature effects on lue
model <- rpmodel(tc=temp, vpd=vpd, co2=co2, fapar=fapar,
                 ppfd=ppfd, elv=elev, kphio=0.05, do_ftemp_kphio=TRUE)

# Save the GPP output to a new netCDF file
gpp_var <- ncvar_def('gpp','', ds$dim)
gpp_file <- nc_create('rpmodel_global_gpp_do_ftkphio.nc', gpp_var)
ncvar_put(gpp_file, gpp_var, model$gpp)

# Tidy up
nc_close(gpp_file)

nc_close(ds)

