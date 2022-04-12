from typing import Optional, Union, Tuple
import numpy as np

from pyrealm.utilities import check_input_shapes, summarize_attrs
from pyrealm.param_classes import C3C4Params

# Design Notes:
#
# see paper Lavergne, Harrison and Prentice (submitted)


class C3C4competition:
    """Implementation of the C3/C4 competition model

    This class provides an implementation of the calculations of C3/C4 competition,
    described by :cite:`Lavergne:submitted`. All of the properties of
    the C3/C4 model are derived from regression functions and estimates of gross primary productivity for C3 and C4 plants.

    """

    def __init__(self,
                 tc: Union[float, np.ndarray],
                 gppc3: Union[float, np.ndarray],
                 gppc4: Union[float, np.ndarray],
                 treecover: Union[float, np.ndarray],
                 cropland: Union[float, np.ndarray],
                 C3C4_params: C3C4Params = C3C4Params()
                 ):

        # Check inputs are broadcastable
        self.shape = check_input_shapes(tc, gppc3,gppc4)
        self.gppc3 = gppc3
        self.gppc4 = gppc4
        self.treecover = treecover
        self.cropland = cropland
        
        self.C3C4_params = C3C4_params

    def __repr__(self):

        return f"C3C4competition(adv={self.adv})"


def c4fraction(tc: Union[float, np.ndarray],
               gppc3: Union[float, np.ndarray],
               gppc4: Union[float, np.ndarray],
               treecover: Union[float, np.ndarray],
               cropland: Union[float, np.ndarray],
               C3C4_params: C3C4Params = C3C4Params()
               ) -> Union[float, np.ndarray]:
 
        """
        Calculate C4 fraction given estimated gross primary productivity for C3 and C4 plants.

        Args:
            gppc3: annual total gross primary productivity for C3 plants (gC m-2 yr-1)
            gppc4: annual total gross primary productivity for C4 plants (gC m-2 yr-1)
            treecover: tree cover (%) from MODIS
            cropland: cropland fraction (%) from MODIS

        Returns:
            adv: advantage of C4 plants
            C4fraction: fraction of C4 plants
        """

        
        
        # Step 1: calculate the advantage of C4 plants with annual sum of GPP for C3 and C4 plants
        #adv = (np.nansum(gppc4,axis=0) - np.nansum(gppc3,axis=0))/np.nansum(gppc3,axis=0)
        adv = (gppc4 - gppc3)/gppc3
        
        adv[adv == np.inf] = np.nan
        
        # Step 2:
        x = adv/np.exp(1/(1+treecover))
        F4 = 1.0 / (1.0 + np.exp(-C3C4_params.k * (x - C3C4_params.q)))
        
        # Step 3: remove areas with treecover > 54.4%
        
        h = (C3C4_params.a * np.power(gppc3/1000,C3C4_params.b) + C3C4_params.c)/(C3C4_params.a * np.power(2.8,C3C4_params.b) + C3C4_params.c)
        
        h[h>1] = 1
        h[h<0] = 0
            
        F4 = F4*(1-h)
        
        # Step 4: remove areas with Tc < -24 degC
        F4[tc < -24] = 0
        
        # Step 5: remove cropland areas
        F4[cropland >= 50] = np.nan
            
        return adv,F4
        
        
def gpp_tot(F4: Union[float, np.ndarray],
               gppc3: Union[float, np.ndarray],
               gppc4: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
 
        """
        Calculate total GPP given the C4 fraction

        Args:
            gppc3: annual total gross primary productivity for C3 plants (gC m-2 yr-1)
            gppc4: annual total gross primary productivity for C4 plants (gC m-2 yr-1)
            F4: fraction of C4 plants

        Returns:
            gpp_tot: annual total GPP considering fraction of C4 and C3 plants (gC m-2 yr-1)
            gppc3: annual total gross primary productivity for C3 plants (gC m-2 yr-1)
            gppc4: annual total gross primary productivity for C4 plants (gC m-2 yr-1)
        """

        gpp_tot = F4*gppc4 + (1-F4)*gppc3
        
        gpp_c3 = gppc3*(1-F4)
        gpp_c4 = gppc4*F4
        
        return gpp_tot,gpp_c3,gpp_c4

