from easydict import EasyDict as edict
import underworld as uw
from underworld import function as fn


class component_phases():
    """
    Class that allows you to create 'phase functions' for a mineral component

    """
    def __init__(self,name, depths,temps, widths,claps,densities):
        """
        Class initialiser.
        Parameter
        ---------
        name : str
            'Component', e.g olivine, pyroxene-garnet
        depths: list
            list of transition depths in kilometers
        widths: list 
            list of transition widths in kilometers
        claps: list 
            list of Clapeyron slopes in Pa/K
        densities: list
            list of density changes in kg/m3
        Returns
        -------
        mesh : dp
        Dictionary storing the phase-transition vales

        """
        if not isinstance(depths,list):
            raise TypeError("depths object passed in must be of type 'list'")
        if not isinstance(temps,list):
            raise TypeError("temps object passed in must be of type 'list'")
        if not isinstance(widths,list):
            raise TypeError("widths object passed in must be of type 'list'")
        if not isinstance(claps,list):
            raise TypeError("claps object passed in must be of type 'list'")
        if not isinstance(densities,list):
            raise TypeError("densities object passed in must be of type 'list'")
        if not len(depths) == len(widths) == len(claps) == len(densities):
            raise ValueError( "All lists of phase values should be the same length")
        self.dp = edict({})
        self.dp.name = name
        self.dp.depths = depths
        self.dp.temps = temps
        self.dp.widths = widths
        self.dp.claps = claps
        self.dp.densities = densities
        
    def build_nd_dict(self, lengthscale, densityscale, gravityscale, tempscale):
        self.ndp = edict({})
        self.ndp.name = self.dp.name
        self.ndp.depths = [i/lengthscale for i in self.dp.depths]
        self.ndp.temps = [i/tempscale for i in self.dp.temps]
        self.ndp.widths = [i/lengthscale for i in self.dp.widths]
        self.ndp.claps = [(i*(tempscale/(densityscale*gravityscale*lengthscale))) for i in self.dp.claps]
        
    def nd_reduced_pressure(self, depthFn, temperatureField, depthPh, clapPh, tempPh):
        """
        Creates an Underworld function, representing the 'reduced pressure'
        """
        return (depthFn - depthPh) - clapPh*(temperatureField - tempPh)

    def nd_phase(self, reduced_p, widthPh):
        
        """
        Creates an Underworld function, representing the phase function in the domain
        """
        return 0.5*(1. + fn.math.tanh(reduced_p/(widthPh)))
    
    def phase_function_sum(self, temperatureField, depthFn):
        """
        Creates an Underworld function, representing the Sum of the individual phase functions:
        -----------
        temperatureField : underworld.mesh._meshvariable.MeshVariable
        
        ...need to put warning in about running build_nd_dict first 
        """    
        
        pf_sum = uw.function.misc.constant(0.)
        
        for phaseId in range(len(test.dp['depths'])):
            #build reduced pressure
            rp = self.nd_reduced_pressure(depthFn, 
                                   temperatureField,
                                   self.ndp['depths'][phaseId ],
                                   self.ndp['claps'][phaseId ],
                                   self.ndp['temps'][phaseId ])
            #build phase function
            pf = self.nd_phase(rp, test.ndp['widths'][phaseId ])
            pf_sum += pf
        
        return pf_sum
    
    def buoyancy_sum(self, temperatureField, depthFn, gravityscale, lengthscale, diffusivityscale, viscosityscale):
        """
        Creates an Underworld function, representing the Sum of the individual phase functions...
        and the associated density changes:
        
        pf_sum = Sum_k{ (Ra*delRho_k*pf_k/rho_0*eta_0*delta_t)}
        -----------
        temperatureField : underworld.mesh._meshvariable.MeshVariable
        
        ...need to put warning in about running build_nd_dict first 
        """
        bouyancy_factor = (gravityscale*lengthscale**3)/(viscosityscale*diffusivityscale)
        
        pf_sum = uw.function.misc.constant(0.)
        
        for phaseId in range(len(test.dp['depths'])):
            #build reduced pressure
            rp = self.nd_reduced_pressure(depthFn, 
                                   temperatureField,
                                   self.ndp['depths'][phaseId ],
                                   self.ndp['claps'][phaseId ],
                                   self.ndp['temps'][phaseId ])
            #build phase function
            pf = self.nd_phase(rp, test.ndp['widths'][phaseId ])
            pf_sum += bouyancy_factor*pf*self.dp['densities'][phaseId ] #we want the dimensional densities here
        
        return pf_sum