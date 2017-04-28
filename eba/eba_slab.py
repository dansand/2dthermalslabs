
# coding: utf-8

# ## Composite, non-linear rheology, extended Boussinesq approximation:
# 
# 
# The viscous rheology in this model is similar to the Garel et al paper cited below. Other parts of the model setup are similar to Čížková and Bina (2015), Arredondo and Billen (2016).
# 
# Here we use a dimensionless system. For the psuedo-plastic effective rheology a Drucker-prager model is used.
# 
# 
# **Keywords:** subduction, composite rheology, dislocation creep
# 
# 
# **References:**
# 
# 
# Garel, Fanny, et al. "Interaction of subducted slabs with the mantle transition‐zone: A regime diagram from 2‐D thermo‐mechanical models with a mobile trench and an overriding plate." Geochemistry, Geophysics, Geosystems 15.5 (2014): 1739-1765.
# 
# Čížková, Hana, and Craig R. Bina. "Geodynamics of trench advance: Insights from a Philippine-Sea-style geometry." Earth and Planetary Science Letters 430 (2015): 408-415.
# 
# Arredondo, Katrina M., and Magali I. Billen. "The Effects of Phase Transitions and Compositional Layering in Two-dimensional Kinematic Models of Subduction." Journal of Geodynamics (2016).
# 

# In[1]:

import numpy as np
import underworld as uw
import math
from underworld import function as fn
import glucifer

import os
import sys
import natsort
import shutil
from easydict import EasyDict as edict
import operator
import pint
import time
import operator
from slippy2 import boundary_layer2d
from slippy2 import material_graph
from slippy2 import spmesh
from slippy2 import phase_function
from unsupported.interfaces import markerLine2D


from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()


# In[2]:

#####
#Stubborn version number conflicts - need to figure out my Docker container runs an old version. For now...
#####
try:
    natsort.natsort = natsort.natsorted
except:
    natsort.natsort = natsort.natsort


# In[ ]:




# In[3]:

#store = glucifer.Store('subduction')
#figParticle = glucifer.Figure( store, figsize=(960,300), name="Particles" )

#figParticle.save_database('test.gldb')


# Model name and directories
# -----

# In[4]:

############
#Model letter and number
############


#Model letter identifier default
Model = "T"

#Model number identifier default:
ModNum = 0

#Any isolated letter / integer command line args are interpreted as Model/ModelNum

if len(sys.argv) == 1:
    ModNum = ModNum 
elif sys.argv[1] == '-f': #
    ModNum = ModNum 
else:
    for farg in sys.argv[1:]:
        if not '=' in farg: #then Assume it's a not a paramter argument
            try:
                ModNum = int(farg) #try to convert everingthing to a float, else remains string
            except ValueError:
                Model  = farg


# In[5]:

###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
xdmfPath = outputPath + 'xdmf/'
outputFile = 'results_model' + Model + '_' + str(ModNum) + '.dat'

if uw.rank()==0:
    # make directories if they don't exist
    if not os.path.isdir(outputPath):
        os.makedirs(outputPath)
    if not os.path.isdir(checkpointPath):
        os.makedirs(checkpointPath)
    if not os.path.isdir(imagePath):
        os.makedirs(imagePath)
    if not os.path.isdir(dbPath):
        os.makedirs(dbPath)
    if not os.path.isdir(filePath):
        os.makedirs(filePath)
    if not os.path.isdir(xdmfPath):
        os.makedirs(xdmfPath)
        
comm.Barrier() #Barrier here so no procs run the check in the next cell too early


# In[6]:

###########
#Check if starting from checkpoint
###########

checkdirs = []
for dirpath, dirnames, files in os.walk(checkpointPath):
    if files:
        print dirpath, 'has files'
        checkpointLoad = True
        checkdirs.append(dirpath)
    if not files:
        print dirpath, 'is empty'
        checkpointLoad = False


# In[7]:

# setup summary output file (name above)
if checkpointLoad:
    checkpointLoadDir = natsort.natsort(checkdirs)[-1]
    if uw.rank() == 0:
        shutil.copyfile(os.path.join(checkpointLoadDir, outputFile), outputPath+outputFile)
    comm.Barrier()
    f_o = open(os.path.join(outputPath, outputFile), 'a')
    prevdata = np.genfromtxt(os.path.join(outputPath, outputFile), skip_header=0, skip_footer=0)
    if len(prevdata.shape) == 1: #this is in case there is only one line in previous file
        realtime = prevdata[-1]  #This corresponds to the column you write time data to
    else:
        realtime = prevdata[prevdata.shape[0]-1, -1]
    step = int(checkpointLoadDir.split('/')[-1])
    timevals = [0.]
else:
    f_o = open(outputPath+outputFile, 'w')
    realtime = 0.
    step = 0
    timevals = [0.]


# Setup parameters
# -----
# 
# Set simulation parameters for test.

# **Use pint to setup any unit conversions we'll need**

# In[8]:

u = pint.UnitRegistry()
cmpery = 1.*u.cm/u.year
mpermy = 1.*u.m/u.megayear
year = 1.*u.year
spery = year.to(u.sec)
cmpery.to(mpermy)


# In[9]:

box_half_width =4000e3
age_at_trench = 100e6
cmperyear = box_half_width / age_at_trench #m/y
mpersec = cmperyear*(cmpery.to(u.m/u.second)).magnitude #m/sec
print(cmperyear, mpersec )


# **Set parameter dictionaries**
# 
# * Parameters are stored in dictionaries. 
# * If starting from checkpoint, parameters are loaded using pickle
# * If params are passed in as flags to the script, they overwrite 

# In[10]:

###########
#Parameter / settings dictionaries get saved&loaded using pickle
###########
 
dp = edict({}) #dimensional parameters
sf = edict({}) #scaling factors
ndp = edict({}) #dimensionless paramters
md = edict({}) #model paramters, flags etc
#od = edict({}) #output frequencies


# In[11]:

dict_list = [dp, sf, ndp, md]
dict_names = ['dp.pkl', 'sf.pkl', 'ndp.pkl', 'md.pkl']

def save_pickles(dict_list, dict_names, dictPath):
    import pickle
    counter = 0
    for pdict in dict_list:
        myfile = os.path.join(dictPath, dict_names[counter])
        with open(myfile, 'wb') as f:
            pickle.dump(pdict, f)
        counter+=1


#ended up having to pretty much write a hard-coded function
#All dictionaries we want checkpointed will have to  be added here 
#and where the function is called
#Fortunately, this function is only called ONCE

def load_pickles():
    import pickle
    dirpath = os.path.join(checkpointPath, str(step))
    dpfile = open(os.path.join(dirpath, 'dp.pkl'), 'r')
    dp = pickle.load(dpfile)
#    #
    ndpfile = open(os.path.join(dirpath, 'ndp.pkl'), 'r')
    ndp = edict(pickle.load(ndpfile))
    #
    sffile = open(os.path.join(dirpath, 'sf.pkl'), 'r')
    sf = edict(pickle.load(sffile))
    #
    mdfile = open(os.path.join(dirpath, 'md.pkl'), 'r')
    md = edict(pickle.load(mdfile))
    return dp, ndp, sf, md


# In[12]:

###########
#Store the physical parameters, scale factors and dimensionless pramters in easyDicts
#Mainly helps with avoiding overwriting variables
###########


#Style => parameters_like_this

dp = edict({#Main physical paramters
           'depth':2000e3, #Depth
           'LS':2900.*1e3, #Length scale
           'rho':3300.,  #reference density
           'g':9.8, #surface gravity
           'eta0':1e20, #Dislocation creep at 250 km, 1573 K, 1e-15 s-1
           'k':1e-6, #thermal diffusivity
           'a':3e-5, #surface thermal expansivity
           'R':8.314, #gas constant
           'Cp':1250., #Specific heat (Jkg-1K-1)
           'TP':1573., #mantle potential temp (K)
           'TS':273., #surface temp (K)
            #Rheology - flow law paramters
           'cm':2e6, #mantle cohesion in Byerlee law
           'cc':2e6, #crust cohesion in Byerlee law
           'fcm':0.2,   #mantle friction coefficient in Byerlee law (tan(phi))
           'fcc':0.02,   #crust friction coefficient 
           'Adf':3e-11, #pre-exp factor for diffusion creep
           'Ads':5e-16, #pre-exp factor for dislocation creep
           'Apr':1e-150,#pre-exp factor for Peierls creep
           #'Apr':1e-145,#pre-exp factor for Peierls creep
           'Edf':3e5,
           'Eds':5.4e5,
           'Epr':5.4e5,
           'Vdf':4e-6,
           'Vds':12e-6,
           'Vpr':10e-6,
           'Alm':6e-17,
           'Elm':2.0e5,
           'Vlm':1.5e-6,
           'SR':1e-15, #reference strain rate
           'n':3.5, #Dislocation creep stress exponent
           'np':20., #Peierls creep stress exponent 
           #Rheology - cutoff values
           'eta_min':1e18, 
           'eta_max':1e25, #viscosity max in the mantle material
           'eta_min_crust':1e18, #viscosity min in the weak-crust material
           'eta_max_crust':1e20, #viscosity max in the weak-crust material
           'ysMax':10000*1e6, #10 GPa
           #Rheology - upper /lowr mantle transition
           'viscIncreaseDepths':[660e3],
           'viscIncreaseWeights':[1.],
           'viscIncreaseWidths':[10e3],
           #Length scales
           'MANTLETOCRUST':8.*1e3, #Crust depth
           'HARZBURGDEPTH':38e3,
           'CRUSTTOMANTLE':800.*1e3,
           'LITHTOMANTLE':(900.*1e3),
           'MANTLETOLITH':200.*1e3, 
           'TOPOHEIGHT':10.*1e3,  #rock-air topography limits
           'CRUSTTOECL':100.*1e3,
           'LOWMANTLEDEPTH':660.*1e3, 
           'CRUSTVISCUTOFF':250.*1e3, #Deeper than this, crust material rheology reverts to mantle rheology
           'AGETRACKDEPTH':100e3, #above this depth we track the age of the lithsphere (below age is assumed zero)
           #Slab and plate parameters
           'roc':250e3,     #radius of curvature of slab
           'subzone':0.0,   #X position of subduction zone...km
           'lRidge':-1.*(5000e3),  #For depth = 670 km, aspect ratio of 4, this puts the ridges at MINX, MAXX
           'rRidge':(5000e3),
           'maxDepth':450e3,
           'theta':70., #Angle to truncate the slab (can also control with a maxDepth param)
           'slabmaxAge':100e6, #age of subduction plate at trench
           'platemaxAge':100e6, #max age of slab (Plate model)
           'opmaxAge':100e6, #age of op
           'sense':'Right', #dip direction
           #'op_age_fac':0.2, #this controls the overidding plate age reduction
           #Misc
           'rDepth':250e3, #reference depth (used to scale / normalize the flow laws)
           'StALS':100e3, #depth of sticky air layer
           'Steta_n':1e19, #stick air viscosity, normal
           'Steta_s':1e18, #stick air viscosity, shear 
           'plate_vel':4,
           'low_mantle_visc_fac':1.
             })

#append any derived parameters to the dictionary
#Adiabatic heating stuff

#dp.dTa = (dp.a*dp.g*(dp.TP))/dp.Cp #adibatic gradient, at Tp
#dp.dTa = 0.0005 #adiabatic gradient, upper mantle value used in Garel et al. 
#dp.deltaTa = (dp.TP + dp.dTa*dp.LS) - dp.TS  #Adiabatic Temp at base of mantle, minus Ts
#dp.rTemp= dp.TP + dp.rDepth*dp.dTa #reference temp, (potential temp + adiabat)

#the base-of-mantle value comes from integrating the adiabatic condition across the mantle
dp.deltaTa = (3110.65) - dp.TS  #Adiabatic Temp at base of mantle, minus Ts, 
                                #this is tied to specific choices of expansivity, Specific heat. 


# In[ ]:




# In[13]:



#((dp.a*dp.g*(dp.TP))/dp.Cp)
#4.05e-5/3e-5
#dp.a/3e-5


# In[14]:

#Modelling and Physics switches

md = edict({'refineMesh':False,
            'stickyAir':False,
            'subductionFault':False,
            'symmetricIcs':False,
            'velBcs':False,
            'aspectRatio':5., #
            'compBuoyancy':False, #use compositional buoyancy, as well as thermal
            'phaseBuoyancy':False,
            'periodicBcs':False,
            'RES':64,
            'PIC_integration':True,
            'ppc':25,
            'elementType':"Q1/dQ0",
            #'elementType':"Q2/DPC1",
            'viscMechs':['diffusion', 'dislocation', 'peierls', 'yielding'],
            'viscCombine':'harmonic', #'harmonic', 'min'
            'secInvFac':math.sqrt(1.),
            'courantFac':0.5 #extra limitation on timestepping
            })


# In[15]:

t = 1./dp.Adf
print(t/3.333e10)


# In[16]:

###########
#If starting from a checkpoint load params from file
###########

if checkpointLoad:
    dp, ndp, sf, md = load_pickles()  #remember to add any extra dictionaries


# In[17]:

###########
#If command line args are given, overwrite
#Note that this assumes that params as commans line args/
#only append to the 'dimensional' and 'model' dictionary (not the non-dimensional)
###########    


###########
#If extra arguments are provided to the script" eg:
### >>> uw.py 2 dp.arg1=1 dp.arg2=foo dp.arg3=3.0
###
###This would assign ModNum = 2, all other values go into the dp dictionary, under key names provided
###
###Two operators are searched for, = & *=
###
###If =, parameter is re-assigned to givn value
###If *=, parameter is multipled by given value
###
### >>> uw.py 2 dp.arg1=1 dp.arg2=foo dp.arg3*=3.0
###########

for farg in sys.argv[1:]:
    try:
        (dicitem,val) = farg.split("=") #Split on equals operator
        (dic,arg) = dicitem.split(".") #colon notation
        if '*=' in farg:
            (dicitem,val) = farg.split("*=") #If in-place multiplication, split on '*='
            (dic,arg) = dicitem.split(".")
            
        if val == 'True': 
            val = True
        elif val == 'False':     #First check if args are boolean
            val = False
        else:
            try:
                val = float(val) #next try to convert  to a float,
            except ValueError:
                pass             #otherwise leave as string
        #Update the dictionary
        if farg.startswith('dp'):
            if '*=' in farg:
                dp[arg] = dp[arg]*val #multiply parameter by given factor
            else:
                dp[arg] = val    #or reassign parameter by given value
        if farg.startswith('md'):
            if '*=' in farg:
                md[arg] = md[arg]*val #multiply parameter by given factor
            else:
                md[arg] = val    #or reassign parameter by given value
                
    except:
        pass
            

comm.barrier()


# In[18]:

dp.deltaTa - 2750.0


# In[19]:

if not checkpointLoad:
    
    #Scaling factors, used to map the dimensional parameters to dimensionless
    
    sf = edict({'stress':dp.LS**2/(dp.k*dp.eta0),
            'lith_grad':dp.rho*dp.g*(dp.LS)**3/(dp.eta0*dp.k) ,
            'vel':dp.LS/dp.k,
            'SR':dp.LS**2/dp.k,
            'W':(dp.rho*dp.g*dp.LS)/(dp.R*dp.deltaTa), #Including adiabatic compression, and deltaTa
            'E': 1./(dp.R*dp.deltaTa), #using deltaTa, the guesstimated adiabatic temp differnnce to scale these paramters
            'Ads':1./((dp.eta0**(-1.*dp.n))*(dp.k**(1. - dp.n))*(dp.LS**(-2.+ (2.*dp.n)))),
            'Adf':dp.eta0,
            'Apr':2.6845783276046923e+40 #same form as Ads, but ndp.np =20. (hardcoded because numbers are too big)
                

           })
    
     #dimensionless parameters
    ndp = edict({
             #Rayliegh number includes full adiabtic temp change, as this is what we scale temp with. 
             'RA':(dp.g*dp.rho*dp.a*dp.deltaTa*(dp.LS)**3)/(dp.k*dp.eta0),       
             #'RA':(dp.g*dp.rho*dp.a*(dp.TP - dp.TS)*(dp.LS)**3)/(dp.k*dp.eta0),
             'depth':dp.depth/dp.LS,
             'Di': dp.a*dp.g*dp.LS/dp.Cp, #Dissipation number
             'H':0.,
             #Temperatures and reference depth
             'TSP':0., #dimensionless potential temp
             'TPP':(dp.TP - dp.TS)/dp.deltaTa, #dimensionless potential temp
             'TS':dp.TS/dp.deltaTa,
             'TP':dp.TP/dp.deltaTa,
             #'rTemp':(dp.rTemp- dp.TS)/dp.deltaTa,
             #'rDepth':dp.rDepth/dp.LS,
              #Rheology - flow law paramters  
             'Ads':dp.Ads*sf.Ads,
             'Adf':dp.Adf*sf.Adf,
             'Apr':dp.Apr*sf.Apr,
             'Alm':dp.Alm*sf.Adf,
             'Wdf':dp.Vdf*sf.W,
             'Edf':dp.Edf*sf.E,
             'Wps':dp.Vpr*sf.W,
             'Eps':dp.Epr*sf.E,
             'Wds':dp.Vds*sf.W,      #{df  => diffusion creep}
             'Eds':dp.Eds*sf.E,      #{ds  => dislocation creep}
             'Elm':dp.Elm*sf.E,      #{pr  => Peierls creep}
             'Elm':dp.Elm*sf.E,
             'Wlm':dp.Vlm*sf.W,
             'cm':dp.cm*sf.stress,
             'cc':dp.cc*sf.stress,    #{dimensionless cohesion in mantl, crust, interface}
             'fcmd':dp.fcm*sf.lith_grad, 
             'fccd':dp.fcc*sf.lith_grad, #{dimensionless friction coefficient in mantle, crust, interface}
             'n':dp.n, #Dislocation creep stress exponent
             'np':dp.np, #Peierls creep stress exponent 
             #Rheology - cutoff values
             'eta_min':dp.eta_min/dp.eta0, 
             'eta_max':dp.eta_max/dp.eta0, #viscosity max in the mantle material
             'eta_min_crust':dp.eta_min_crust/dp.eta0, #viscosity min in the weak-crust material
             'eta_max_crust':dp.eta_max_crust/dp.eta0, #viscosity max in the weak-crust material
             'ysMax':dp.ysMax*sf.stress,
             #Rheology - upper /lowr mantle transition
             'viscIncreaseDepths':[z/dp.LS for z in dp.viscIncreaseDepths],
             'viscIncreaseWidths':[z/dp.LS for z in dp.viscIncreaseWidths],
             'viscIncreaseWeights':dp.viscIncreaseWeights,
             #Lengths scales
             'MANTLETOCRUST':dp.MANTLETOCRUST/dp.LS, #Crust depth
             'HARZBURGDEPTH':dp.HARZBURGDEPTH/dp.LS,
             'CRUSTTOMANTLE':dp.CRUSTTOMANTLE/dp.LS,
             'LITHTOMANTLE':dp.LITHTOMANTLE/dp.LS,
             'MANTLETOLITH':dp.MANTLETOLITH/dp.LS,
             'TOPOHEIGHT':dp.TOPOHEIGHT/dp.LS,  #rock-air topography limits
             'CRUSTTOECL':dp.CRUSTTOECL/dp.LS,
             'LOWMANTLEDEPTH':dp.LOWMANTLEDEPTH/dp.LS, 
             'CRUSTVISCUTOFF':dp.CRUSTVISCUTOFF/dp.LS, #Deeper than this, crust material rheology reverts to mantle rheology
             'AGETRACKDEPTH':dp.AGETRACKDEPTH/dp.LS,
             #Slab and plate parameters
             'roc':dp.roc/dp.LS,         #radius of curvature of slab
             'subzone':dp.subzone/dp.LS, #X position of subduction zone...km
             'lRidge':dp.lRidge/dp.LS,   #For depth = 670 km, aspect ratio of 4, this puts the ridges at MINX, MAXX
             'rRidge':dp.rRidge/dp.LS,
             'maxDepth':dp.maxDepth/dp.LS,
             #Misc
             'Steta_n':dp.Steta_n/dp.eta0, #stick air viscosity, normal
             'Steta_s':dp.Steta_n/dp.eta0, #stick air viscosity, shear 
             'StALS':dp.StALS/dp.LS,
             'plate_vel':sf.vel*dp.plate_vel*(cmpery.to(u.m/u.second)).magnitude,
             'low_mantle_visc_fac':dp.low_mantle_visc_fac
            })

    #Append any more derived paramters
    ndp.SR = dp.SR*sf.SR #characteristic strain rate
    ndp.StRA = (3300.*dp.g*(dp.LS)**3)/(dp.eta0 *dp.k) #Composisitional Rayleigh number for rock-air buoyancy force
    ndp.TaP = 1. - ndp.TPP,  #Dimensionless adiabatic component of deltaT


# In[20]:

#dp.deltaTa/(dp.TP - dp.TS)
#650710714.5
#ndp.RA
#ndp.Di
#dp.deltaTa


# ### Output Frequency

# In[21]:

#Metric output stuff
figures =  'store' #glucifer Store won't work on all machines, if not, set to 'gldb' 

#The following are step-based actions
swarm_repop, swarm_update = 5, 10
checkpoint_every = 100
metric_output = 10
sticky_air_temp = 1e6


#The following are time-based actions
filesMy = 0.5e6 #dimensional time intervale to write files
files_freq  = filesMy*(3600.*365.*24.)/sf.SR  #applies to files and gldbs


# ### Model/ mesh  setup parameters

# In[22]:

###########
#Model setup parameters
###########

dim = 2          # number of spatial dimensions


#Domain and Mesh paramters
Xres = int(md.RES*8)   #more than twice the resolution in X-dim, which will be 'corrected' by the refinement we'll use in Y



MINY = 1. - (ndp.depth)
if md.stickyAir:
    Yres = int(md.RES)
    MAXY = 1. + dp.StALS/dp.LS #150km
    
else:
    Yres = int(md.RES)
    MAXY = 1.


periodic = [False, False]
if md.periodicBcs:
    periodic = [True, False]
    
    
half_width = np.round(0.5*(ndp.depth)*md.aspectRatio, 1)
MINX = -1.*half_width

MAXX = half_width
MAXY = 1.





# In[23]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (Xres, Yres), 
                                 minCoord    = (MINX, MINY), 
                                 maxCoord    = (MAXX, MAXY), periodic=periodic)





velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )


velocityField.data[:]       = [0.,0.]
pressureField.data[:]       = 0.
temperatureDotField.data[:] = 0.


# In[24]:

coordinate = fn.input()
depthFn = 1. - coordinate[1] #a function providing the depth
xFn = coordinate[0]  #a function providing the x-coordinate
yFn = coordinate[1]  #a function providing the y-coordinate


# In[25]:

#


# In[26]:

mesh.reset()

jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
yFn = coordinate[1]
yField = uw.mesh.MeshVariable( mesh=mesh, nodeDofCount=1 )
yField.data[:] = 0.
yBC = uw.conditions.DirichletCondition( variable=yField, indexSetsPerDof=(jWalls,) )

# set bottom wall temperature bc
for index in mesh.specialSets["MinJ_VertexSet"]:
    yField.data[index] = mesh.minCoord[1]
# set top wall temperature bc
for index in mesh.specialSets["MaxJ_VertexSet"]:
    yField.data[index] = mesh.maxCoord[1]
    
    
    
s = 3.5
intensityFac = 2.5
intensityFn = (((yFn - MINY)/(MAXY-MINY))**s)
intensityFn *= intensityFac
intensityFn += 1.


yLaplaceEquation = uw.systems.SteadyStateHeat(temperatureField=yField, fn_diffusivity=intensityFn, conditions=[yBC,])

# get the default heat equation solver
yLaplaceSolver = uw.systems.Solver(yLaplaceEquation)
# solve
yLaplaceSolver.solve()


#Get the array of Y positions - copy may be necessary, not sure. 
newYpos = yField.data.copy() 

uw.barrier()
with mesh.deform_mesh():
     mesh.data[:,1] = newYpos[:,0]


# In[27]:

#mesh.save('mesh.h5')


# In[28]:

#fig= glucifer.Figure(quality=3)

#fig.append( glucifer.objects.Surface(mesh,intensityFn, discrete=True))
#fig.append( glucifer.objects.Mesh(mesh))
#fig.show()
#fig.save_database('test.gldb')


# In[29]:

#THis is a hack for adding a sticky air domain, we redefine MAXY and things like the temperature stencil work from Y = 1. 

if md.stickyAir:
    MAXY = 1.


# Initial conditions
# -------
# 

# In[30]:

#Explicit Euler to get the adiabatic temp

nSamples=1000
depths = np.linspace(0.,1. - mesh.minCoord[1],nSamples)

dz = np.diff(depths)
#c0 = ((dp.a*dp.g*dp.LS)/dp.Cp)
c0 = ndp.Di 
_temps = np.zeros(nSamples)


_temps[0] = ndp.TP #the key thing is we need to work in dimensionless absolute temps, i.e. ndp.TP, not ndp.TPP
i = 1
for delta_z in dz:
    _temps[i] = (_temps[i-1] * (1. + c0*delta_z))
    i +=1

    
#now cenvert to the T' system.
_temps -=  ndp.TS


# In[31]:

#ndp.TP


# In[32]:

meshDepths = mesh.maxCoord[1] - mesh.data[:,1]

tempInterp = np.interp(meshDepths, depths, np.array(_temps) )

temperatureField.data[:,0] = tempInterp


# In[33]:


#potTempFn = ndp.TPP + (depthFn)*ndp.Di*(ndp.TP) #a function providing the (linearised) at potential temp

#the adiabatic heating relies on the dimensionless absolute temperature, not T'
abHeatFn = -1.*velocityField[1]*(temperatureField + ndp.TS)*ndp.Di #a function providing the adiabatic heating rate


# In[34]:

#Fix the ridge locations of greater than boundaries... 

if ndp.lRidge < mesh.minCoord[0]:
    ndp.lRidge = mesh.minCoord[0] + (10e3/dp.LS)
if ndp.rRidge > mesh.maxCoord[0]:
    ndp.rRidge = mesh.maxCoord[0] - (10e3/dp.LS)


# In[35]:

def age_fn(xFn, sz = 0.0, lMOR=MINX, rMOR=MAXX, opFac=1., conjugate_plate = False):
    """
    Simple function to generate a discrete 1-d (i.e x-coordinate) function for the age of the thermal BC. 
    All paramters are dimensionless
    sz: location of subduction zone
    lMOR: location of left-hand MOR
    rMOR: location of right-hand MOR
    opFac: uniform reduce the age of the right hand plate by this factor
    conjugate_plate: if True, build plates on the outer sides of the MORs, if False, age = 0. 
    """
    
    if lMOR < MINX:
        lMOR = MINX
    if rMOR > MAXX:
        rMOR = MAXX

    if conjugate_plate:
        ageFn = fn.branching.conditional([(operator.and_(xFn > lMOR, xFn < sz) , (xFn + abs(lMOR))/(abs(sz-lMOR))), 
                                      (operator.and_(xFn < rMOR, xFn >= sz), (1.-(xFn + abs(sz))/abs(rMOR-sz))*opFac),
                                      (xFn > rMOR, opFac*(xFn -abs(rMOR)) / abs(rMOR-sz) ),
                                      (True, fn.math.abs((((xFn + abs(lMOR)) / (abs(sz-lMOR))))))
                                         ])
    else:    
        
        ageFn = fn.branching.conditional([(operator.and_(xFn > lMOR, xFn < sz) , (xFn + abs(lMOR))/(abs(sz-lMOR))), 
                                      (operator.and_(xFn < rMOR, xFn >= sz), (1.-(xFn + abs(sz))/abs(rMOR-sz))*opFac),

                                      (True, 0.0)])
    return ageFn


# In[36]:

#dp.opmaxAge = 100e6
#dp.slabmaxAge = 32e6

#%pylab inline
#ageFn = age_fn(xFn, sz =ndp.subzone, lMOR=ndp.lRidge,rMOR=ndp.rRidge, conjugate_plate=True, opFac = dp.opmaxAge/dp.slabmaxAge)
#plt.plot(ageFn.evaluate(tWalls))


# In[37]:

###########
#Thermal initial condition 2: 
#if symmetricIC == False, we build an asymmetric subduction-zone
###########


#  a few conversions
ageAtTrenchSeconds = min(dp.platemaxAge*(3600*24*365), dp.slabmaxAge*(3600*24*365))
Org = (ndp.subzone, MAXY-ndp.roc)


#This builds the age function
ageFn = age_fn(xFn, sz =ndp.subzone, lMOR=ndp.lRidge,rMOR=ndp.rRidge, conjugate_plate=True, opFac = dp.opmaxAge/dp.slabmaxAge)


#dimensionlize the age function
ageFn *= ageAtTrenchSeconds

w0 = (2.*math.sqrt(dp.k*ageAtTrenchSeconds))/dp.LS #diffusion depth of plate at the trench

tempBL = (temperatureField) *fn.math.erf((depthFn*dp.LS)/(2.*fn.math.sqrt(dp.k*ageFn))) + ndp.TSP #boundary layer function

#if depth is less than 2.*w0, use the thermal stencil, otherwise the adiabatic temp field
tempTBL =  fn.branching.conditional([(depthFn < 2.*w0, tempBL),
                          (True, temperatureField)])

if not md.symmetricIcs:
    if not checkpointLoad:
        out = uw.utils.MeshVariable_Projection( temperatureField, tempTBL) #apply function with projection
        out.solve()




# In[38]:

#ageAtTrenchSeconds, 
#dp.k, 
#w0*dp.LS
#ndp.roc, ndp.MANTLETOCRUST


# In[39]:

#Now build the perturbation part
def inCircleFnGenerator(centre, radius):
    coord = fn.input()
    offsetFn = coord - centre
    return fn.math.dot( offsetFn, offsetFn ) < radius**2





#We use three circles to define our slab and crust perturbation,  
Oc = inCircleFnGenerator(Org , ndp.roc)
Oc2 = inCircleFnGenerator(Org , ndp.roc + (0.5*ndp.MANTLETOCRUST)) #increases the amount of crust in the interface
Ic = inCircleFnGenerator(Org , ndp.roc - 2.*w0)
Cc = inCircleFnGenerator(Org , ndp.roc - (1.*ndp.MANTLETOCRUST)) #... weak zone on 'inside' of slab
Hc = inCircleFnGenerator(Org , ndp.roc - ndp.HARZBURGDEPTH) #... Harzburgite layer 
dx = (ndp.roc)*(np.math.tan((np.math.pi/180.)*dp.theta))


#We'll also create a triangle which will truncate the circles defining the slab...
if dp.sense == 'Left':
    ptx = ndp.subzone - dx
else:
    ptx = ndp.subzone + dx
coords = ((0.+ ndp.subzone, MAXY), (0.+ ndp.subzone, MAXY-ndp.roc), (ptx, MAXY))
Tri = fn.shape.Polygon(np.array(coords))

#tempBL = (temperatureField) *fn.math.erf((depthFn*dp.LS)/(2.*fn.math.sqrt(dp.k*ageFn))) + ndp.TSP #boundary layer function

#Actually apply the perturbation
if not md.symmetricIcs:
    if not checkpointLoad:
        sdFn = ((ndp.roc - fn.math.sqrt((coordinate[0] - Org[0])**2. + (coordinate[1] - Org[1])**2.)))
        slabFn = ndp.TPP*fn.math.erf((sdFn*dp.LS)/(2.*math.sqrt(dp.k*ageAtTrenchSeconds))) + ndp.TSP
        for index, coord in enumerate(mesh.data):
            if (
                Oc.evaluate(tuple(coord)) and
                Tri.evaluate(tuple(coord)) and not
                Ic.evaluate(tuple(coord)) and
                coord[1] > (1. - ndp.maxDepth)
                ): #In the quarter-circle defining the lithosphere
                temperatureField.data[index] = slabFn.evaluate(mesh)[index]


# In[40]:

#tempBL.evaluate((-1.7 + 1e-6, 0.028315867670597919))
#temperatureField.evaluate((-1.7 , 1. - 0.028315867670597919))[0][0]
#tempBL.evaluate((mesh.minCoord[0], mesh.maxCoord[1] -distFromSlabTop[index][0]))
#mesh.minCoord[0], mesh.maxCoord[1] -distFromSlabTop[index][0]


# In[41]:

#ndp.TPP, ndp.TP, 
#tempBL = (temperatureField) *fn.math.erf((depthFn*dp.LS)/(2.*fn.math.sqrt(dp.k*ageFn))) + ndp.TSP #boundary layer function
#fn.input?


# In[42]:

## Make sure material in sticky air region is at the surface temperature.
for index, coord in enumerate(mesh.data):
            if coord[1] >= MAXY:
                temperatureField.data[index] = ndp.TSP


# In[43]:

#temperatureField.data.max(), temperatureField.data.mean(), temperatureField.data.std(), temperatureField.data.min()


# In[44]:

#fig= glucifer.Figure(quality=3)

#fig.append( glucifer.objects.Surface(mesh,temperatureField, discrete=True))
#fig.append( glucifer.objects.Mesh(mesh))


#fig.show()
#fig.save_database('test.gldb')


# In[45]:

#ageFn = age_fn(xFn, sz =ndp.subzone, lMOR=ndp.lRidge,rMOR=ndp.rRidge, conjugate_plate=True, opFac = dp.opmaxAge/dp.slabmaxAge)
#ageFn.evaluate(mesh).min()


# Boundary conditions
# -------

# In[46]:

iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls =mesh.specialSets["MinJ_VertexSet"]


#this should already be set
#for index in mesh.specialSets["MinJ_VertexSet"]:
#    temperatureField.data[index] = 1.0 

for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = ndp.TSP

VelBCs = mesh.specialSets["Empty"]



if md.velBcs:
    for index in list(tWalls.data):

        if (mesh.data[int(index)][0] < (ndp.subzone - 0.05*md.aspectRatio) and 
            mesh.data[int(index)][0] > (mesh.minCoord[0] + 0.05*md.aspectRatio)): #Only push with a portion of teh overiding plate
            #print "first"
            VelBCs.add(int(index))
            #Set the plate velocities for the kinematic phase
            velocityField.data[index] = [ndp.plate_vel, 0.]
        
        elif (mesh.data[int(index)][0] > (ndp.subzone + 0.05*md.aspectRatio) and 
            mesh.data[int(index)][0] < (mesh.maxCoord[0] - 0.05*md.aspectRatio)):
            #print "second"
            VelBCs.add(int(index))
            #Set the plate velocities for the kinematic phase
            velocityField.data[index] = [0., 0.]
        

#If periodic, we'll fix a the x-vel at a single node - at the bottom left (index 0)
Fixed = mesh.specialSets["Empty"]
Fixed.add(int(0))        
        

if periodic[0] == False:
    if md.velBcs:
        print(1)
        freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( iWalls + VelBCs, jWalls) )
    else:
        print(2)
        freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( iWalls, jWalls) )



if periodic[0] == True:
    if md.velBcs:
        print(3)
        freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( Fixed + VelBCs , jWalls) )
    else:
        print(4)
        freeslipBC = uw.conditions.DirichletCondition( variable      = velocityField, 
                                               indexSetsPerDof = ( Fixed, jWalls) )
        
    
    
    
# also set dirichlet for temp field
dirichTempBC = uw.conditions.DirichletCondition(     variable=temperatureField, 
                                              indexSetsPerDof=(tWalls,) )
dT_dy = [0.,0.]

# also set dirichlet for temp field
neumannTempBC = uw.conditions.NeumannCondition( dT_dy, variable=temperatureField, 
                                         nodeIndexSet=bWalls)



# In[47]:

#check VelBCs are where we want them
#test = np.zeros(len(tWalls.data))
#VelBCs
#tWalls.data
#tWalls.data[VelBCs.data]
#test[np.in1d(tWalls.data, VelBCs.data)] = 1.
#test



# In[ ]:




# Swarm setup
# -----
# 

# In[48]:

###########
#Material Swarm and variables
###########

#create material swarm
gSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)
#vSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)


#create swarm variables
yieldingCheck = gSwarm.add_variable( dataType="int", count=1 )
#tracerVariable = gSwarm.add_variable( dataType="int", count=1)
materialVariable = gSwarm.add_variable( dataType="int", count=1 )
ageVariable = gSwarm.add_variable( dataType="double", count=1 )
#testVariable = gSwarm.add_variable( dataType="float", count=1 )


#these lists  are part of the checkpointing implementation
varlist = [materialVariable, yieldingCheck, ageVariable]
varnames = ['materialVariable', 'yieldingCheck', 'ageVariable']


# In[49]:

mantleIndex = 0
crustIndex = 1
harzIndex = 2
airIndex = 3



if checkpointLoad:
    checkpointLoadDir = natsort.natsort(checkdirs)[-1]
    temperatureField.load(os.path.join(checkpointLoadDir, "temperatureField" + ".hdf5"))
    pressureField.load(os.path.join(checkpointLoadDir, "pressureField" + ".hdf5"))
    velocityField.load(os.path.join(checkpointLoadDir, "velocityField" + ".hdf5"))
    gSwarm.load(os.path.join(checkpointLoadDir, "swarm" + ".h5"))
    for ix in range(len(varlist)):
        varb = varlist[ix]
        varb.load(os.path.join(checkpointLoadDir,varnames[ix] + ".h5"))

else:

    # Layouts are used to populate the swarm across the whole domain
    layout = uw.swarm.layouts.PerCellRandomLayout(swarm=gSwarm, particlesPerCell=int(md.ppc))
    gSwarm.populate_using_layout( layout=layout ) # Now use it to populate.


    # Swarm variables
    materialVariable.data[:] = mantleIndex
    #tracerVariable.data[:] = 1
    yieldingCheck.data[:] = 0
    ageVariable.data[:] = -1

    #Set initial air and crust materials (allow the graph to take care of lithsophere)
    #########
    #This initial material setup will be model dependent
    #########
    for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
        if (1. - gSwarm.particleCoordinates.data[particleID][1]) < ndp.MANTLETOCRUST:
                 materialVariable.data[particleID] = crustIndex
                
                
                
#also create a swarm for viewing. This one doesn't need to be checkpointed               
#layout1 = uw.swarm.layouts.PerCellRandomLayout(swarm=vSwarm, particlesPerCell=int(md.ppc/8))
#vSwarm.populate_using_layout( layout=layout1 ) # Now use it to populate.


# In[ ]:




# In[50]:

##############
#Set the initial particle age for particles above the critical depth; 
#only material older than crustageCond will be transformed to crust / harzburgite
##############

ageVariable.data[:] = 0. #start with all zero
ageVariable.data[:] = ageFn.evaluate(gSwarm)/sf.SR
crustageCond = 2e6*(3600.*365.*24.)/sf.SR #set inital age above critical depth. (x...Ma)



ageConditions = [ (depthFn < ndp.AGETRACKDEPTH, ageVariable),  #In the main loop we add ageVariable + dt here
                  (True, 0.) ]
                 
#apply conditional 
ageVariable.data[:] = fn.branching.conditional( ageConditions ).evaluate(gSwarm)

ageDT = 0.#this is used in the main loop for short term time increments


# In[51]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,ageVariable))
#fig.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True, valueRange =[1e-3,1e5]))

#fig.append( glucifer.objects.Surface(mesh, strainRate_2ndInvariant, logScale=True, valueRange =[1e-3,1e5] ))
#fig.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.002))
#fig.append( glucifer.objects.Surface(mesh,densityMapFn))
#fig.append( glucifer.objects.Surface(mesh,raylieghFn))

#fig.show()


# In[52]:

##############
#Here we set up a directed graph object that we we use to control the transformation from one material type to another
##############

#All depth conditions are given as (km/D) where D is the length scale,
#note that 'model depths' are used, e.g. 1-z, where z is the vertical Underworld coordinate
#All temp conditions are in dimensionless temp. [0. - 1.]

#Need a list of all material indexes (safer in parallel)
material_list = [0,1,2,3]

if not checkpointLoad:
    materialVariable.data[:] = 0 #Initialize to zero 

#Setup the graph object
DG = material_graph.MatGraph()

#Important: First thing to do is to add all the material types to the graph (i.e add nodes)
DG.add_nodes_from(material_list)

#Now set the conditions for transformations

hs = 2e3/dp.LS  #add some hysteresis to the depths of transition

#... to mantle
DG.add_transition((crustIndex,mantleIndex), depthFn, operator.gt, ndp.CRUSTTOMANTLE + hs)
DG.add_transition((harzIndex,mantleIndex), depthFn, operator.gt, ndp.CRUSTTOMANTLE + hs)
DG.add_transition((airIndex,mantleIndex), depthFn, operator.gt, ndp.TOPOHEIGHT + hs)

#... to crust
DG.add_transition((mantleIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
DG.add_transition((mantleIndex,crustIndex), xFn, operator.lt, ndp.subzone) #No crust on the upper plate
DG.add_transition((mantleIndex,crustIndex), ageVariable, operator.gt, crustageCond)


DG.add_transition((harzIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
DG.add_transition((harzIndex,crustIndex), xFn, operator.lt, ndp.subzone) #This one sets no crust on the upper plate
DG.add_transition((harzIndex,crustIndex), ageVariable, operator.gt, crustageCond)

#... to Harzbugite
DG.add_transition((mantleIndex,harzIndex), depthFn, operator.lt, ndp.HARZBURGDEPTH)
DG.add_transition((mantleIndex,harzIndex), depthFn, operator.gt, ndp.MANTLETOCRUST)
DG.add_transition((mantleIndex,harzIndex), ageVariable, operator.gt, crustageCond) #Note we can mix functions and swarm variabls

#... to air
DG.add_transition((mantleIndex,airIndex), depthFn, operator.lt,0. - ndp.TOPOHEIGHT)
DG.add_transition((crustIndex,airIndex), depthFn, operator.lt, 0. - ndp.TOPOHEIGHT)


# In[53]:

##############
#For the slab_IC, we'll also add a crustal weak zone following the dipping perturbation
##############

if checkpointLoad != True:
    if not md.symmetricIcs:
        for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
            if (
                Oc2.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                Tri.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                gSwarm.particleCoordinates.data[particleID][1] > (1. - ndp.maxDepth) and
                Cc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) == False
                
                ):
                materialVariable.data[particleID] = crustIndex
                
            elif (
                Oc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                Tri.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                gSwarm.particleCoordinates.data[particleID][1] > (1. - ndp.maxDepth) and
                Hc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) == False
                
                ):
                materialVariable.data[particleID] = harzIndex


# In[54]:

##############
#This is how we use the material graph object to test / apply material transformations
##############

if not checkpointLoad:
    DG.build_condition_list(materialVariable)

    for i in range(1): 
        materialVariable.data[:] = fn.branching.conditional(DG.condition_list).evaluate(gSwarm)


# In[55]:

fig= glucifer.Figure()
fig.append( glucifer.objects.Points(gSwarm,temperatureField))
#fig.append( glucifer.objects.Surface(mesh, corrTempFn))
#fig.show()
#fig.save_database('test.gldb')


# ## phase and compositional buoyancy

# In[56]:

from unsupported_dan.phase import phase_utils


# In[ ]:




# In[57]:

##############
#Set up phase buoyancy contributions
#the phase function approach of Yuen and Christenson is implemented in the Slippy2 phase_function class 
##############

buoyancyFactor = (dp.g*dp.LS**3)/(dp.eta0*dp.k)
slopeFactor = (dp.deltaTa/(dp.rho*dp.g*dp.LS))


olDepths= np.array([410e3,660e3, 2740e3])/dp.LS                   #depths of phase transitions along adiabat
olTemps = (np.array([1600., 1900., 2300]) - dp.TS)/dp.deltaTa   #temperatures of phase transitions along adiabat
olWidths = np.array([10e3, 20e3, 20e3])/dp.LS                   #width if transition
olClaps= np.array([2.e6, -2.5e6, 10e6])*slopeFactor             #Clapeyron slope of transition
olBuoyancies= np.array([180., 400., 61.6])*buoyancyFactor       #Scaled buoyancy contributions



#olivine
olivinePhase = phase_utils.Phases(name = 'ol', 
                        depths=olDepths,                  #depths of phase transitions along adiabat
                        temps = olTemps,                  #temperatures of phase transitions along adiabat
                        widths = olWidths,                #width if transition
                        claps=olClaps,                    #Clapeyron slope of trnasition
                        buoyancies = olBuoyancies)        #density change of phase transition


olivine_phase_buoyancy = olivinePhase.buoyancy_sum(temperatureField, depthFn)


#Pyroxene/garnet

grtDepths = np.array([400e3, 720e3, 2740e3])/dp.LS            #depths of phase transitions along adiabat
grtTemps = (np.array([ 1600., 1900., 2300]) - dp.TS)/dp.deltaTa   #temperatures of phase transitions along adiabat
grtWidths = np.array([ 20e3, 20e3, 20e3])/dp.LS                   #width of transition
grtClaps= np.array([ 1.e6, 1.e6,  10e6])*slopeFactor             #Clapeyron slope of transition
grtBuoyancies= np.array([ 150., 400., 61.6])*buoyancyFactor       #Scaled buoyancy contributions



garnetPhase = phase_utils.Phases(name = 'ol', 
                        depths=grtDepths,                  #depths of phase transitions along adiabat
                        temps = grtTemps,                  #temperatures of phase transitions along adiabat
                        widths = grtWidths,                #width if transition
                        claps=grtClaps,                    #Clapeyron slope of trnasition
                        buoyancies = grtBuoyancies)        #density change of phase transition


garnet_phase_buoyancy = garnetPhase.buoyancy_sum(temperatureField, depthFn)

#Eclogite

eclDepths = np.array([60e3])/dp.LS            #depths of phase transitions along adiabat
eclTemps = (np.array([1000.]) - dp.TS)/dp.deltaTa   #temperatures of phase transitions along adiabat
eclWidths = np.array([2e3])/dp.LS                   #width of transition
eclClaps= np.array([0.e6])*slopeFactor             #Clapeyron slope of transition
eclBuoyancies= np.array([540.])*buoyancyFactor       #Scaled buoyancy contributions


#We make the eclogite a separate phase transition as we only want to apply it to the basalt crust
eclogitePhase = phase_utils.Phases(name = 'ecl', 
                        depths=eclDepths,                  #depths of phase transitions along adiabat
                        temps = eclTemps,                  #temperatures of phase transitions along adiabat
                        widths = eclWidths,                #width if transition
                        claps=eclClaps,                    #Clapeyron slope of trnasition
                        buoyancies = eclBuoyancies)        #density change of phase transition



eclogite_phase_buoyancy = eclogitePhase.buoyancy_sum(temperatureField, depthFn)


# In[ ]:




# In[58]:

fig= glucifer.Figure(quality=3)
#fig.append( glucifer.objects.Points(gSwarm,olivine_phase_buoyancy/ndp.RA , discrete=True, fn_mask=lithRestFn))
fig.append( glucifer.objects.Surface(mesh,olivine_phase_buoyancy/ndp.RA , discrete=True))
#fig.append( glucifer.objects.Points(gSwarm,temperatureField , discrete=True, fn_mask=lithRestFn))
#fig.append( glucifer.objects.Mesh(mesh))
#fig.show()
#fig.save_database('test.gldb')
#fig.save_image('active.png')
#fig.save_image('active_phase.png')


# In[59]:

#md.compBuoyancy = True
#md.phaseBuoyancy = True


# In[60]:

##############
#Set up compositional buoyancy contributions
##############
#if we are using crust greater then 6 km, we correct the density contribution, 
#by using a weighted contribution of pure pyroxene / garnet (i.e basalt) and Harzburgite
crustCorrectionFac = 8e3/dp.MANTLETOCRUST

harzDensity = 3240. 
basaltDensity = 3000.

basalt_comp_buoyancy  = (dp.rho - (crustCorrectionFac*basaltDensity + (1. - crustCorrectionFac)*(harzDensity)))*buoyancyFactor
#basalt_comp_buoyancy  = (dp.rho - basaltDensity)*buoyancyFactor
harz_comp_buoyancy = (dp.rho - harzDensity)*buoyancyFactor
pyrolite_comp_buoyancy = 0.

#print(basalt_comp_buoyancy, harz_comp_buoyancy, pyrolite_comp_buoyancy)


pyrolitebuoyancyFn =  (ndp.RA*temperatureField)
harzbuoyancyFn =      (ndp.RA*temperatureField) 
basaltbuoyancyFn =    (ndp.RA*temperatureField)
    
    
if md.compBuoyancy: 
    pyrolitebuoyancyFn +=  pyrolite_comp_buoyancy                           
    harzbuoyancyFn    +=   harz_comp_buoyancy                           
    basaltbuoyancyFn   += basalt_comp_buoyancy 
                           

if md.phaseBuoyancy:
    pyrolitebuoyancyFn -= (0.6*olivine_phase_buoyancy + 0.4*garnet_phase_buoyancy) 
    harzbuoyancyFn -= (0.75*olivine_phase_buoyancy + 0.25*garnet_phase_buoyancy)  
    basaltbuoyancyFn -=  crustCorrectionFac*eclogite_phase_buoyancy + ((1. - crustCorrectionFac)*olivine_phase_buoyancy + crustCorrectionFac*garnet_phase_buoyancy)


                          


# In[61]:

#fig= glucifer.Figure(quality=3)

#fig.append( glucifer.objects.Points(gSwarm,densityMapFn , discrete=True, fn_mask=vizVariable))
#fig.append( glucifer.objects.Surface(mesh, pyrolitebuoyancyFn/ndp.RA , discrete=True))
#fig.append( glucifer.objects.Points(gSwarm,densityMapFn))

#fig.append( glucifer.objects.Surface(mesh,basaltbuoyancyFn/ndp.RA , discrete=True))

#fig.append( glucifer.objects.Mesh(mesh))
#fig.show()
#fig.save_database('test.gldb')


# In[62]:

#%pylab inline
#rWalls = mesh.specialSets["MaxI_VertexSet"]

#plt.plot( mesh.data[rWalls.data][:,1], pyrolitebuoyancyFn.evaluate(rWalls), c='r')
#plt.plot(mesh.data[rWalls.data][:,1], harzbuoyancyFn.evaluate(rWalls), c='g')
#plt.plot(mesh.data[rWalls.data][:,1], basaltbuoyancyFn.evaluate(rWalls), c='b')


# ## Faults / interfaces

# In[63]:

from unsupported_dan.interfaces.marker2D import markerLine2D
from unsupported_dan.faults.faults2D import fault2D, fault_collection


# In[64]:

def copy_markerLine2D(ml, thickness=False, ID=False):
    
    """
    
    """
    if not thickness:
        thickness = ml.thickness
    if not ID:
        ID = -1*ml.ID
    new_line = markerLine2D(mesh, velocityField, [], [], thickness,  ID)
    if ml.swarm.particleCoordinates.data.shape[0] > 0:
        new_line.swarm.add_particles_with_coordinates(ml.swarm.particleCoordinates.data.copy())
        
    new_line.rebuild()
    return new_line


# In[65]:

###########
#Initial Coordinates for interfaces and faults
###########

#subduction fault
introPoint = ndp.subzone - abs(ndp.subzone - ndp.lRidge)/2. #half way between ridge and Sz
#faultthickness = ndp.MANTLETOCRUST/2. #initiale fault at half-depth of crust
faultthickness = 3.*ndp.MANTLETOCRUST #initialize swarm deeper than crust

nfault = 200
faultCoords =np.zeros((nfault, 2))

reducedRocM = ndp.roc  - faultthickness
xlimslab = reducedRocM*math.cos(math.pi*(90. - dp.theta)/180)
faultCoords[:, 0] = np.linspace(introPoint, ndp.subzone + xlimslab, nfault) #note SZ location is hardcoded here 
for index, xval in np.ndenumerate(faultCoords[:,0]):
    #print index, xval
    #swarmCoords[index[0], 1] = 1. - isodepthFn.evaluate((xval, 0.)) #This bit for the plate 
    if  xval < ndp.subzone:
        faultCoords[index[0], 1] = MAXY - faultthickness #This bit for the plate 
        
    else:
        faultCoords[index[0], 1] = (MAXY - (faultthickness) - (reducedRocM - ( math.sqrt((reducedRocM**2 - (xval-ndp.subzone)**2)))))
        
    
faultCoords = faultCoords[faultCoords[:,1] > (MAXY - ndp.maxDepth)] #kill any deeper than cutoff


#surface tracer interface:
surfaceCoords =np.ones((nfault, 2))*1.
surfaceCoords[:,0] = np.linspace(MINX, MAXX, nfault)






# In[66]:

#Initiaze the swarms in a 
fault  = fault2D(mesh, velocityField, [], [], faultthickness, 0., 0., crustIndex)
surface  = fault2D(mesh, velocityField, [], [], ndp.StALS, 0., 0., airIndex)
#slab_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], 1e9/dp.LS, crustIndex)   #Note very large fault thickness 

fault_coll = fault_collection([fault])

#Initiaze the swarms in a 
#fault_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], faultthickness, 0.0, 0.0, crustIndex)
#surface_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], ndp.StALS, 0.0, 0.0, airIndex)
#slab_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], 1e9/dp.LS, 0.0, 0.0, crustIndex)   #Note very large fault thickness

#These lists are used to checkpoint the marker lines, similar to the swarm variables.        
interfaces = []
interfaces.append(fault)
interfaces.append(surface)
interfacenames = ['fault_seg', 'surface_seg']


#If restarting, load the swarms from file Interfaces are just swarms, so should be fine to rely on parallel h5 machinery here
if checkpointLoad:
    for ix in range(len(interfaces)):
        tempname = interfacenames[ix]
        interfaces[ix].swarm.load(os.path.join(checkpointLoadDir,  tempname + ".h5"))

#otherwise add the point from at the initial locations
else:
    fault.add_points(faultCoords[:, 0], faultCoords[:, 1])
    surface.add_points(surfaceCoords[:, 0], surfaceCoords[:, 1])
    #slab_seg.add_points(slabCoords[:, 0], slabCoords[:, 1]) 
    

#rebuild the fault, necessary to swicth off empty flag., if starting fom checkpoint
    
fault.rebuild()
surface.rebuild()


# In[67]:

#Add variables to the surface swarm
surfaceVelx = surface.swarm.add_variable( dataType="float", count=1 )
surfaceVelx.data[...] = velocityField[0].evaluate(surface.swarm)


surfaceStrainY = surface.swarm.add_variable( dataType="float", count=1 )
surfaceVisc = surface.swarm.add_variable( dataType="float", count=1 )
surfacePressure = surface.swarm.add_variable( dataType="float", count=1 )

#this one needs interpolation ... do in main loop


# In[68]:

#Add the necessary swarm variables
#note that these swarm vars. don't get checkpointed, they should rebuild from model state

proximityVariable      = gSwarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = gSwarm.add_variable( dataType="float", count=1 )
directorVector   = gSwarm.add_variable( dataType="double", count=2)

directorVector.data[:,:] = 0.0
proximityVariable.data[:] = 0
signedDistanceVariable.data[:] = 0.0


# In[69]:

#inform the mesh of the fault

sd, pts0 = fault.compute_signed_distance(gSwarm.particleCoordinates.data, distance=w0)
sp, pts0 = fault.compute_marker_proximity(gSwarm.particleCoordinates.data)

proximityVariable.data[np.logical_and(sd<0,sp == fault.ID)] = sp[np.logical_and(sd<0,sp == fault.ID)]

dv, nzv = fault.compute_normals(gSwarm.particleCoordinates.data)
directorVector.data[nzv] = dv[nzv]




edotn_SFn, edots_SFn = fault_coll.global_fault_strainrate_fns(velocityField, directorVector, proximityVariable)


# Rheology
# -----
# 
# 

# In[70]:

##############
#Set up viscosity increase functions (i.e upper - lower mantle)
##############

#dp.viscIncreaseDepths = [660e3, 1000e3]
#dp.viscIncreaseWeights = [0.5,0.5]
#dp.viscIncreaseWidths = [30e3,30e3]



viscDepthWeightFn = fn.misc.constant(0.)

for i in range(len(ndp.viscIncreaseDepths)):
    viscDepthWeightFn += (ndp.viscIncreaseWeights[i])*0.5*(fn.math.tanh((depthFn - ndp.viscIncreaseDepths[i])/(ndp.viscIncreaseWidths[i])) + 1.)
    
viscDepthWeightFn = 1. - viscDepthWeightFn


# In[71]:

##############
#Set up any functions required by the rheology
##############

sym_strainRate = fn.tensor.symmetric( 
                            velocityField.fn_gradient )

strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            sym_strainRate)/md.secInvFac # secInvFac sometimes required if differnet definition of eii is used.

def safe_visc(func, viscmin=ndp.eta_min, viscmax=ndp.eta_max):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


# In[72]:

############
#Rheology: create UW2 functions for all viscous mechanisms
#############

omega = fn.misc.constant(1.) #this function can hold any arbitary viscosity modifications 


##Diffusion Creep
diffusion = (1./ndp.Adf ) *fn.math.exp( ((ndp.Edf + (depthFn*ndp.Wdf))/((temperatureField+ ndp.TS)))) 


##Diffusion Creep
lmdiffusion = (1./ndp.Alm ) *fn.math.exp( ((ndp.Elm + (depthFn*ndp.Wlm))/((temperatureField+ ndp.TS)))) 



linearVisc = safe_visc(diffusion)

##Dislocation Creep
pefac = (ndp.Ads**((-1./ndp.n)))
srfac = strainRate_2ndInvariant**((1.-ndp.n)/ndp.n)
dislocation = pefac*srfac*fn.math.exp( ((ndp.Eds + (depthFn*ndp.Wds))/(ndp.n*(temperatureField + ndp.TS))))


##Peirls Creep
pefac = (ndp.Apr**((-1./ndp.np)))
srfac = strainRate_2ndInvariant**((1.-ndp.np)/ndp.np)
peierls = pefac*srfac*fn.math.exp( ((ndp.Eps + (depthFn*ndp.Wps))/(ndp.np*(temperatureField+ ndp.TS))))


##Define the mantle Plasticity
ys =  ndp.cm + (depthFn*ndp.fcmd)
ysf = fn.misc.min(ys, ndp.ysMax)
yielding = ysf/(2.*(strainRate_2ndInvariant)) 

##Crust rheology
crustys =  ndp.cc + (depthFn*ndp.fccd)
crustysf = fn.misc.min(crustys, ndp.ysMax)
crustyielding = crustysf/(2.*(strainRate_2ndInvariant)) 


# In[73]:

#crustyielding.evaluate(mesh).min(), crustyielding.evaluate(mesh).mean(),crustyielding.evaluate(mesh).max()
#finalcrustviscosityFn.evaluate(mesh).min(),finalcrustviscosityFn.evaluate(mesh).max(), finalcrustviscosityFn.evaluate(mesh).mean()
#strainRate_2ndInvariant.evaluate(mesh).min(), strainRate_2ndInvariant.evaluate(mesh).mean(), strainRate_2ndInvariant.evaluate(mesh).max()
#safe_visc(diffusion).evaluate(mesh).mean()
#interfaceviscosityFn.evaluate(mesh).min(), interfaceviscosityFn.evaluate(mesh).max(), interfaceviscosityFn.evaluate(mesh).mean()
#interfaceyielding.evaluate(mesh).min(), interfaceyielding.evaluate(mesh).mean(), interfaceyielding.evaluate(mesh).max()
#ndp.eta_min, ndp.eta_max_interface
#finalviscosityFn.evaluate(mesh).min(),finalviscosityFn.evaluate(mesh).max(), finalviscosityFn.evaluate(mesh).mean()


# In[74]:

############
#Rheology: combine viscous mechanisms in various ways 
#harmonic: harmonic average of all mechanims
#min: minimum effective viscosity of the mechanims
#mixed: takes the minimum of the harmonic and the plastic effective viscosity
#############

#Map viscMechs list (defined in setup), to the actual functions, requires that same names are used.
viscdict = {}
for i in md.viscMechs:
    viscdict[i] = locals()[i]


#Harmonic average of all mechanisms    
if md.viscCombine == 'harmonic':
    denom = fn.misc.constant(0.)
    for mech in viscdict.values():
        denom += 1./mech
    uppermantleviscosityFn = safe_visc(1./denom)
    #harmonic_test = uppermantleviscosityFn
    #Only diffusuion creep for lower mantle
    finalviscosityFn  = (viscDepthWeightFn*uppermantleviscosityFn) + (1. - viscDepthWeightFn)*(lmdiffusion*ndp.low_mantle_visc_fac)

    #Add the weaker crust mechanism, plus any cutoffs
    crust_denom = denom + (1./crustyielding)
    crustviscosityFn = safe_visc(1./crust_denom, viscmin=ndp.eta_min_crust, viscmax=ndp.eta_max_crust)
    #Crust viscosity only active above between CRUSTVISCUTOFF and MANTLETOCRUST
    finalcrustviscosityFn  = fn.branching.conditional([(depthFn < ndp.CRUSTVISCUTOFF, crustviscosityFn),
                                                     (True, finalviscosityFn)])
    
    
    
if md.viscCombine == 'min':
    mantleviscosityFn = fn.misc.constant(ndp.eta_max)
    for mech in viscdict.values():
        mantleviscosityFn = fn.misc.min(mech, mantleviscosityFn )
    uppermantleviscosityFn  = safe_visc(mantleviscosityFn)
    #min_test = mantleviscosityFn
    #Only diffusion creep for lower mantle
    finalviscosityFn  = (viscDepthWeightFn*uppermantleviscosityFn) + (1. - viscDepthWeightFn)*(lmdiffusion*ndp.low_mantle_visc_fac)
    #Add the weaker crust and interface mechanisms, plus any cutoffs
    crustviscosityFn = safe_visc(fn.misc.min(finalviscosityFn, crustyielding), viscmin=ndp.eta_min, viscmax=ndp.eta_max_crust)
    #Crust viscosity only active above between CRUSTVISCUTOFF and MANTLETOCRUST
    finalcrustviscosityFn  = fn.branching.conditional([(depthFn < ndp.CRUSTVISCUTOFF, crustviscosityFn),
                                                     (True, finalviscosityFn)])


# In[75]:

#viscMinConditions = fn.misc.min(diffusion, dislocation, peierls, yielding)
#for mech in [safe_visc(diffusion), safe_visc(dislocation), safe_visc(peierls), safe_visc(yielding)]:
#    viscMin = fn.misc.min(mech, viscMin) 

viscMin = fn.misc.constant(ndp.eta_max)
#Generate the minimum viscosity function 
if 'diffusion' in md.viscMechs:
    viscMin = fn.misc.min(safe_visc(diffusion), viscMin)
if 'dislocation' in md.viscMechs:
    viscMin = fn.misc.min(safe_visc(dislocation), viscMin)
if 'peierls' in md.viscMechs:
    viscMin = fn.misc.min(safe_visc(peierls), viscMin)
if 'yielding' in md.viscMechs:
    viscMin = fn.misc.min(safe_visc(yielding), viscMin)


    
dm = 1e-5
   

#Working but static
#viscMinConditions = [ ( operator.and_((viscMin > (ndp.eta_max - dm) ),
#                          (viscMin < (ndp.eta_max + dm) ))  , 0),  #visc = ndp.eta_max
#                      ( operator.and_((viscMin > (diffusion - dm) ),
#                           (viscMin < (diffusion + dm) ))  , 1),  #visc = diffusion
#                      ( operator.and_((viscMin > (dislocation - dm) ),
#                           (viscMin < (dislocation + dm) ))  , 2),#visc = dislocation
#                      ( operator.and_((viscMin > (peierls - dm) ),
#                           (viscMin < (peierls + dm) ))  , 3),    #visc = peierls
#                      ( operator.and_((viscMin > (yielding - dm) ),
#                           (viscMin < (yielding + dm) ))  , 4),   #visc = yielding
#                      ( True                                           , 5) ] #visc = eta_min (should be)





mechCount = 0
viscMinConditions = []
viscMinConditions.append(( operator.and_((viscMin > (ndp.eta_max - dm) ),
                           (viscMin < (ndp.eta_max + dm) ))  , mechCount))

mechCount +=1

if 'diffusion' in md.viscMechs:
    viscMinConditions.append(( operator.and_((viscMin > (diffusion - dm) ),
                           (viscMin < (diffusion + dm) ))  , mechCount))
    mechCount +=1
    

if 'dislocation' in md.viscMechs:
    viscMinConditions.append(( operator.and_((viscMin > (dislocation - dm) ),
                           (viscMin < (dislocation + dm) ))  , mechCount))
    mechCount +=1
    
if 'peierls' in md.viscMechs:
    viscMinConditions.append(( operator.and_((viscMin > (peierls - dm) ),
                           (viscMin < (peierls + dm) ))  , mechCount))
    mechCount +=1
   

if 'yielding' in md.viscMechs:
    viscMinConditions.append(( operator.and_((viscMin > (yielding - dm) ),
                           (viscMin < (yielding + dm) ))  , mechCount))
    mechCount +=1
 

viscMinConditions.append( (True                                           , mechCount))



# use the branching conditional function to set each particle's index
fnViscMin = fn.branching.conditional( viscMinConditions )



# Stokes system setup
# -----
# 

# In[76]:

densityMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {airIndex:ndp.StRA,
                                    crustIndex:basaltbuoyancyFn, 
                                    mantleIndex:pyrolitebuoyancyFn,
                                    harzIndex:harzbuoyancyFn} )


# In[77]:


# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
gravity = ( 0.0, 1.0 )

# Now create a buoyancy force vector using the density and the vertical unit vector. 
buoyancyFn = densityMapFn * gravity


# In[79]:


if md.PIC_integration:
    stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              fn_viscosity=linearVisc, 
                              fn_bodyforce=buoyancyFn,
                             swarm=gSwarm)
    

else:
    stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              fn_viscosity=linearVisc, 
                              fn_bodyforce=buoyancyFn )


# In[80]:

solver = uw.systems.Solver(stokesPIC)
if not checkpointLoad:
    solver.solve() #A solve on the linear visocisty is unhelpful unless we're starting from scratch


# In[81]:

viscosityMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {crustIndex:finalcrustviscosityFn,
                                    mantleIndex:finalviscosityFn,
                                    harzIndex:finalviscosityFn} )



# In[82]:

#Add the non-linear viscosity to the Stokes system
stokesPIC.fn_viscosity = viscosityMapFn


# In[83]:

solver.set_inner_method("mumps")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e7)
solver.options.scr.ksp_rtol = 1.0e-4
solver.solve(nonLinearIterate=True)
solver.print_stats()


# In[ ]:




# In[84]:

#Check which particles are yielding
#yieldingCheck.data[:] = 0

#yieldconditions = [ ( mantleviscosityFn < Visc , 1), 
#               ( True                                           , 0) ]

# use the branching conditional function to set each particle's index
#yieldingCheck.data[:] = fn.branching.conditional( yieldconditions ).evaluate(gSwarm)


# In[85]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
#fig.append( glucifer.objects.Points(gSwarm, diffusion, logScale=True, valueRange =[1e-1,1e5]))
#fig.append( glucifer.objects.Surface(mesh, temperatureField ))#
#fig.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.002))
#fig.append( glucifer.objects.Surface(mesh,densityMapFn))
#fig.append( glucifer.objects.Surface(mesh,raylieghFn))
#fig.append( glucifer.objects.Points(swarmPlateBoundary, pointSize=4))
#fig.show()
#fig.save_database('test.gldb')


# ## Set up principal stress vectors
# 

# In[86]:

def eig2d(sigma):

    """
    Input: sigma, symmetric tensor, numpy array of length 3, with xx yy xy compenents
    
    Output:
    
    s1: first major stress fms will be the most extensive
    s2: second major stress the least extensive, most compressive
    deg: angle to the first major stress axis (most extensive in degrees anticlockwise from horizontal axis - x)
    """
 

    s11=sigma[0]
    #s12=sigma[2]/2.  #(engineering strain/stress)
    s12=sigma[2]
    s22=sigma[1]

    fac = 28.64788975654116; #90/pi - 2theta conversion

    x1 = (s11 + s22)/2.0;
    x2 = (s11 - s22)/2.0;
    R = x2 * x2 + s12 * s12;
    
    #Get the stresses 
    if(R > 0.0):         #if shear stress is not zero
        R = np.sqrt(R);
        s1 = x1 + R;
        s2 = x1 - R;
    else:
        s1 = x1;        #if shear stress is zero
        s2 = x1;

    if(x2 != 0.0):   
        deg = fac * math.atan2(s12,x2); #Return the arc tangent (measured in radians) of y/x. The fac contains the 2-theta    
    #else, the normal componets are equa - we are at the angle of maximum shear stress
    elif s12 <= 0.0: 
        deg= -45.0;
    else:
        deg=  45.0;
    return s1, s2, deg


# In[87]:

eig1       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
eig2       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )

ssr = sym_strainRate.evaluate(mesh)


principalAngles  = np.apply_along_axis(eig2d, 1, ssr[:, :])[:,2]


eig1.data[:,0] = np.cos(np.radians(principalAngles - 90.)) #most compressive 
eig1.data[:,1] = np.sin(np.radians(principalAngles - 90.))

eig2.data[:,0] = np.cos(np.radians(principalAngles ))      #most extensive
eig2.data[:,1] = np.sin(np.radians(principalAngles ))


# ## Stress variable

# In[88]:

from scipy.spatial import cKDTree as kdTree


def nn_evaluation(fromSwarm, _data, n=1, weighted=False):

    """
    This function provides nearest neighbour information for uw swarms,
    given the "_data", which could be the .data handle (coordinates) of a mesh or a differnet swarm, this function returns the indices of the n nearest neighbours in "fromSwarm"
    it also returns the inverse-distance if weighted=True.

    The function works in parallel.   

    The arrays come out a bit differently when used in nearest neighbour form
    (n = 1), or IDW: (n > 1). The examples belowe show how to fill out a swarm variable in each case.


    Usage n = 1:
    ------------
    ix, weights = nn_evaluation(swarm, mesh.data, n=n, weighted=False)
    toSwarmVar.data[:][:,0] =  fromSwarmVar.evaluate(fromSwarm)[_ix][:,0]

    Usage n > 1:
    ------------
    ix, weights = nn_evaluation(swarm, toSwarm.particleCoordinates.data, n=n, weighted=False)
    toSwarmVar.data[:][:,0] =  np.average(fromSwarmVar.evaluate(fromSwarm)[ix][:,:,0], weights=weights, axis=1)

    """ 


    #print("fromSwarm data shape", fromSwarm.particleCoordinates.data.shape)

    if len(_data) > 0: #this is required for safety in parallel

        #we rebuild the tree as we assume the fromSwarm is being advected
        fromSwarm.tree = kdTree(fromSwarm.particleCoordinates.data)
        tree = fromSwarm.tree
        d, ix = tree.query(_data, n)
        if n == 1:
            weights = np.ones(_data.shape[0])
        elif not weighted:
            weights = np.ones((_data.shape[0], n))*(1./n)
        else:
            weights = (1./d[:])/(1./d[:]).sum(axis=1)[:,None]
        return ix,  weights
    else:
        return  np.empty(0., dtype="int"),  np.empty(0., dtype="int")


# In[ ]:




# In[89]:

_ix, _weights = nn_evaluation(gSwarm, mesh.data, n=5, weighted=True)

# create a variable to hold sigma_xx
stressField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
stressVariableFn =  2.*sym_strainRate[0]*viscosityMapFn
#stressField.data[:,0] = np.average(stressVariableFn.evaluate(gSwarm)[_ix][:,:,0],weights=_weights, axis=1)


# In[ ]:




# In[90]:

#_ix1, _weights1 = nn_evaluation(gSwarm, surface.swarm.particleCoordinates.data, n=5, weighted=True)


#stressYFn =  2.*sym_strainRate[1]*viscosityMapFn
#surfaceStressY.data[:,0] = np.average(stressYFn.evaluate(gSwarm)[_ix1][:,:,0],weights=_weights1, axis=1)


# Advection-diffusion System setup
# -----

# In[91]:

advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField,
                                         fn_sourceTerm    = 0.0,
                                         fn_diffusivity = 1.0, 
                                         #conditions     = [neumannTempBC, dirichTempBC] )
                                         conditions     = [ dirichTempBC] )
order = 2
if md.periodicBcs:
    order = 1

materialadvector = uw.systems.SwarmAdvector( swarm         = gSwarm, 
                                     velocityField = velocityField, 
                                     order         = order)


# In[92]:

#population_control = uw.swarm.PopulationControl(gSwarm,deleteThreshold=0.2,splitThreshold=1.,maxDeletions=3,maxSplits=0, aggressive=True, particlesPerCell=int(md.ppc))

population_control = uw.swarm.PopulationControl(gSwarm,deleteThreshold=0.006,splitThreshold=0.1,maxDeletions=int(md.ppc/5),maxSplits=int(md.ppc/5), aggressive=True,aggressiveThreshold=0.9, particlesPerCell=int(md.ppc))


# ############
# #Slightly Diffuse the initial perturbation
# #############
# 
# #I was playing around with a tailored diffusivity to target the slab
# 
# inCircleFnGenerator#Now build the perturbation part
# def htan(centre, radius, widthPh, farVal = 0.01, fac = 10.):
#     coord = fn.input()
#     offsetFn = coord - centre
#     dist = fn.math.sqrt(fn.math.dot( offsetFn, offsetFn ))
#     
#     
#     return (((fn.math.tanh(((radius - dist))/widthPh) + 1.) /2.))*fac + farVal
# 
# tfun = htan((0.1, 0.9), 0.1, 0.1, 0.1)
# 
# for index in mesh.specialSets["MinJ_VertexSet"]:
#     temperatureField.data[index] = 
# for index in mesh.specialSets["MaxJ_VertexSet"]:
#     temperatureField.data[index] = 
# 
# timetoDifffuse = 0.#Million years
# incrementtoDiffuse = 0.2 #Million years
# 
# timetoDifffuse = (timetoDifffuse*1e6*(spery)/sf.SR).magnitude
# incrementtoDiffuse = (incrementtoDiffuse*1e6*(spery)/sf.SR).magnitude
# 
# totAdt = 0.
# it = 0
# while totAdt < timetoDifffuse:
#     dtad = advDiff.get_max_dt()
#     print("step") + str(it) 
#     advDiff.integrate(incrementtoDiffuse)
#     totAdt += incrementtoDiffuse
#     it += 1
#     
# #Reset Boundary conds.   
# for index in mesh.specialSets["MinJ_VertexSet"]:
#     temperatureField.data[index] = ndp.TBP
# for index in mesh.specialSets["MaxJ_VertexSet"]:
#     temperatureField.data[index] = ndp.TSP
#     
# comm.Barrier()

# Analysis functions / routines
# -----
# 
# Most of the metrics we want to calculate are either:
# 
# * extrema of some field / function
# * integral of some field / function
# * average value of some function (integral divide by area)
# 
# In addition, we also want to be able to determine these metrics over some restricted part of the domain, where the restriction may either be due some value of a field, a material type, or something more arbitrary.
# 
# Much of he challenge lies in defining these restriction functions in an efficient and robust way (i.e they don't break down as the model evolves)
# 
# For volume integrals, and extrema, we build a hierarchy of restriction functions, each borrowing from the previous, until we have divided the domain into a number of sub regions of interest. 
# 
# In general, averages are found afterwards by combining the integral and the area of the relavent subregion

# In[93]:

###################
#Volume Restriction functions
###################

#Level 1. Global
globRestFn = fn.misc.constant(1.)

#Level 2. Rock - air:
rockRestFn = uw.swarm.SwarmVariable(gSwarm, dataType='double', count=1)
rockRestFn.data[:] = 0.
rockRestFn.data[np.where(materialVariable.data[:] != airIndex)] = 1.
rockRestFn *= globRestFn #Add next level up in heirarchy


#Level 3. lithosphere - mantle:
mantleconditions = [ (                                  operator.and_(temperatureField < 0.9*ndp.TPP, operator.and_(xFn> ndp.lRidge,xFn< ndp.rRidge )), 1.),
                   (                                                   True , 0.) ]
lithRestFn = fn.branching.conditional(mantleconditions)
lithRestFn*=rockRestFn #Add next level up in heirarchy


#Level 4. lower plate - upper plate:


#This whole section simply builds a restriction Fn that separates the upper and lower plate 
#It's pretty cumbersome, and will need to advected, rebuilt
#can YOU think of a better way?
   
lowerPlateRestFn = gSwarm.add_variable( dataType="double", count=1 )
lowerPlateRestFn.data[:] = 0.0
sd, pts0 = fault.compute_signed_distance(gSwarm.particleCoordinates.data, distance=w0)
sp, pts0 = fault.compute_marker_proximity(gSwarm.particleCoordinates.data, distance=w0)

#lowerPlateRestFn.data[np.logical_and(sd>0,sp == fault.ID)] = sp[np.logical_and(sd>0,sp == fault.ID)]
lowerPlateRestFn.data[np.logical_and(sd>(-1.*faultthickness ),sp == fault.ID)] = sp[np.logical_and(sd>(-1.*faultthickness ),sp == fault.ID)]

lowerPlateRestFn *= lithRestFn #Add next level up in heirarchy
#The lowerPlateRestFn isn't working on restart - no idea why


#Level 5. hinge of lower plate:

hinge60Spatialconditions = [ (           operator.and_( (depthFn < (60e3/dp.LS)),  (xFn > ndp.subzone - 60e3/dp.LS)), 1.),
                   (                                                   True , 0.) ]

hinge60RestFn = fn.branching.conditional(hinge60Spatialconditions)
hinge60RestFn*=lowerPlateRestFn #Add next level up in heirarchy



hinge180Spatialconditions = [ (           operator.and_( (depthFn < (180e3/dp.LS)),  (xFn > ndp.subzone - 180e3/dp.LS)), 1.),
                   (                                                   True , 0.) ]

hinge180RestFn = fn.branching.conditional(hinge180Spatialconditions)
hinge180RestFn*=lowerPlateRestFn #Add next level up in heirarchy


#Level 6. crust/interface in hinge of lower plate:

interfaceRestFn = uw.swarm.SwarmVariable(gSwarm, dataType='double', count=1)
interfaceRestFn.data[:] = 0.
interfaceRestFn.data[np.where(materialVariable.data[:] == crustIndex)] = 1.
interfaceRestFn *= hinge60RestFn #Add next level up in heirarchy


# In[94]:


#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,lithRestFn))
#fig.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True, valueRange =[1e-3,1e5]))
#fig.append( glucifer.objects.Points(gSwarm, lowerPlateRestFn ))
#fig.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.002))
#fig.append( glucifer.objects.Surface(mesh,densityMapFn))
#fig.append( glucifer.objects.Surface(mesh,raylieghFn))
#fig.append( glucifer.objects.Points(slab_seg.swarm, pointSize=10))
#fig.save_database('test.gldb')


# In[95]:

respltconditions = [ 
                    (                                  hinge60RestFn*2. > rockRestFn*1., 1.),
                    (                                  lowerPlateRestFn*3. > hinge60RestFn*2. , 3.),
                    (                                  lithRestFn*5. > lowerPlateRestFn*3. , 4.),
                   (                                                   True , 0.) ]

respltFn = fn.branching.conditional(respltconditions )


# In[96]:

###################
#Surface Restriction functions
###################

def platenessFn(val = 0.1):
    normgradV = fn.math.abs(velocityField.fn_gradient[0]/fn.math.sqrt(velocityField[0]*velocityField[0])) #[du*/dx]/sqrt(u*u)



    srconditions = [ (                                  normgradV < val, 1.),
                   (                                                   True , 0.) ]


    return fn.branching.conditional(srconditions)

srRestFn = platenessFn(val = 0.1)


# In[97]:

###################
#Setup any Functions to be integrated
###################

sqrtv2 = fn.math.sqrt(fn.math.dot(velocityField,velocityField))
vx = velocityField[0]
v2x = fn.math.dot(velocityField[0],velocityField[0])
sqrtv2x = fn.math.sqrt(fn.math.dot(velocityField[0],velocityField[0]))
dw = temperatureField*velocityField[1]
sinner = fn.math.dot( strainRate_2ndInvariant, strainRate_2ndInvariant )
vd = 4.*viscosityMapFn*sinner #there's an extra factor of 2, which is necessary because the of factor of 0.5 in the UW second invariant 
dTdZ = temperatureField.fn_gradient[1]


# In[98]:

###################
#Create integral, max/min templates 
###################

def volumeint(Fn = 1., rFn=globRestFn):
    return uw.utils.Integral( Fn*rFn,  mesh )

def surfint(Fn = 1., rFn=globRestFn, surfaceIndexSet=mesh.specialSets["MaxJ_VertexSet"]):
    return uw.utils.Integral( Fn*rFn, mesh=mesh, integrationType='Surface', surfaceIndexSet=surfaceIndexSet)

def maxMin(Fn = 1.):
    #maxMin(Fn = 1., rFn=globRestFn
    #vuFn = fn.view.min_max(Fn*rFn) #the restriction functions don't work with the view.min_max fn yet
    vuFn = fn.view.min_max(Fn)
    return vuFn
    


# In[99]:

#Setup volume integrals on different sub regions

##Whole rock domain

_areaintRock = volumeint(rockRestFn)
_tempintRock = volumeint(temperatureField, rockRestFn)
_rmsintRock = volumeint(sqrtv2,rockRestFn)
_dwintRock = volumeint(dw,rockRestFn)
_vdintRock = volumeint(vd,rockRestFn)

##Lith 

_areaintLith  = volumeint(lithRestFn)
_tempintLith  = volumeint(temperatureField, lithRestFn)
_rmsintLith  = volumeint(sqrtv2,lithRestFn)
_dwintLith  = volumeint(dw,lithRestFn)
_vdintLith  = volumeint(vd,lithRestFn)

##Lower plate

_areaintLower  = volumeint(lowerPlateRestFn)
_tempintLower  = volumeint(temperatureField, lowerPlateRestFn)
_rmsintLower  = volumeint(sqrtv2,lowerPlateRestFn)
_dwintLower = volumeint(dw,lowerPlateRestFn)
_vdintLower  = volumeint(vd,lowerPlateRestFn)

##Hinge lower plate

_areaintHinge180  = volumeint(hinge180RestFn)
_vdintHinge180  = volumeint(vd,hinge180RestFn)

_areaintHinge60  = volumeint(hinge60RestFn)
_vdintHinge60  = volumeint(vd,hinge60RestFn)


##Interface

_areaintInterface  = volumeint(interfaceRestFn)
_vdintInterface = volumeint(vd,interfaceRestFn)


# In[100]:

#Setup surface integrals

_surfLength = surfint()
_rmsSurf = surfint(sqrtv2x)
_nuTop = surfint(dTdZ)
_nuBottom = surfint(dTdZ, surfaceIndexSet=mesh.specialSets["MinJ_VertexSet"])
_plateness = surfint(srRestFn)
_pressure = surfint(pressureField)


# In[101]:

#Setup max min fns (at the moment, we can't pass restriction function to view.min_max, so we're limited to whole volume or surface extrema)

##Whole rock domain

#_maxMinVisc = maxMin(viscosityMapFn1)  #These don't work on swarm variables or mapping functions, yet
#dummyFn = _maxMinVisc.evaluate(mesh)
#_maxMinStressInv = maxMin(2*viscosityMapFn1*strainRate_2ndInvariant) #These don't work on swarm variables or mapping functions, yet
#dummyFn = _maxMinStress.evaluate(mesh)
#_maxMinVd = maxMin(vd) #These don't work on swarm variables or mapping functions, yet
#dummyFn = _maxMinVd.evaluate(mesh)


_maxMinVel = maxMin(velocityField) 
dummyFn = _maxMinVel.evaluate(mesh)

_maxMinSr = maxMin(strainRate_2ndInvariant) 
dummyFn = _maxMinSr.evaluate(mesh)


#Surface extrema
_maxMinVxSurf = maxMin(vx)
dummyFn = _maxMinVxSurf.evaluate(tWalls)


# In[ ]:




# In[102]:

#Volume Ints
areaintRock = _areaintRock.evaluate()[0]
tempintRock = _tempintRock.evaluate()[0]
rmsintRock = _rmsintRock.evaluate()[0]
dwintRock = _dwintRock.evaluate()[0]
vdintRock = _vdintRock.evaluate()[0]
areaintLith = _areaintLith.evaluate()[0]
tempintLith = _tempintLith.evaluate()[0]
rmsintLith = _rmsintLith.evaluate()[0]
dwintLith = _dwintLith.evaluate()[0]
vdintLith = _vdintLith.evaluate()[0]
areaintLower = _areaintLower.evaluate()[0]
tempintLower = _tempintLower.evaluate()[0]
rmsintLower = _rmsintLower.evaluate()[0]
dwintLower = _dwintLower.evaluate()[0]
vdintLower = _vdintLower.evaluate()[0]
vdintHinge180 = _vdintHinge180.evaluate()[0]
areaintHinge180 = _areaintHinge180.evaluate()[0]
vdintHinge60 = _vdintHinge60.evaluate()[0]
areaintHinge60= _areaintHinge60.evaluate()[0]
vdintInterface = _vdintInterface.evaluate()[0]
areaintInterface= _areaintInterface.evaluate()[0]

#Surface Ints
surfLength = _surfLength.evaluate()[0]
rmsSurf = _rmsSurf.evaluate()[0]
nuTop = _nuTop.evaluate()[0]
nuBottom = _nuBottom.evaluate()[0]
plateness = _plateness.evaluate()[0]
pressureSurf = _pressure.evaluate()[0]


#Max mins
maxVel = _maxMinVel.max_global()
minVel = _maxMinVel.min_global() 
maxSr = _maxMinSr.max_global()
minSr = _maxMinSr.min_global()
maxVxsurf = _maxMinVxSurf.max_global()
minVxsurf = _maxMinVxSurf.min_global()


# print(areaintRock)
# print(tempintRock)
# print(rmsintRock)
# print(dwintRock)
# print(vdintRock)
# print(areaintLith)
# print(tempintLith )
# print(rmsintLith)
# print(dwintLith)
# print(vdintLith)
# print(areaintLower)
# print(tempintLower)
# print(rmsintLower) 
# print(dwintLower)
# print(vdintLower)
# print(vdintHinge60)
# print(vdintHinge180)
# print(vdintInterface)
# 
# print(surfLength)
# print(rmsSurf)
# print(nuTop)
# print(nuBottom)
# print(plateness)
# 
# 
# print(maxVel)
# print(minVel)
# print(maxSr)
# print(minSr)
# print(maxVxsurf)
# print(minVxsurf)

# Viz.
# -----

# In[ ]:




# In[103]:

#Build a depth dependent mask for the vizualisation

depthVariable      = gSwarm.add_variable( dataType="float", count=1 )
depthVariable.data[:] = depthFn.evaluate(gSwarm)

vizVariable      = gSwarm.add_variable( dataType="int", count=1 )
vizVariable.data[:] = 0

for index, value in enumerate(depthVariable.data[:]):
    #print index, value
    if np.random.rand(1)**5 > value/(MAXY - MINY):
        vizVariable.data[index] = 1
        
del index, value    #get rid of any variables that might be pointing at the .data handles (these are!)


# In[ ]:




# In[104]:

fig= glucifer.Figure()
fig.append( glucifer.objects.Points(gSwarm,temperatureField, fn_mask=vizVariable))
fig.append( glucifer.objects.Mesh(mesh))
#fig.show()
fig.save_database('test.gldb')


# In[105]:

if figures == 'gldb':
    #Pack some stuff into a database as well
    figDb = glucifer.Figure()
    #figDb.append( glucifer.objects.Mesh(mesh))
    figDb.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.0005))
    #figDb.append( glucifer.objects.Points(gSwarm,tracerVariable, colours= 'white black'))
    figDb.append( glucifer.objects.Points(gSwarm,materialVariable, fn_mask=vizVariable))

    #figDb.append( glucifer.objects.Points(gSwarm,viscMinVariable))
    figDb.append( glucifer.objects.Points(gSwarm,fnViscMin, fn_mask=vizVariable))
    #figDb.append( glucifer.objects.Points(gSwarm,fnViscMin))
    #figDb.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True))
    #figDb.append( glucifer.objects.Points(gSwarm, strainRate_2ndInvariant, logScale=True))
    figDb.append( glucifer.objects.Points(gSwarm,temperatureField, fn_mask=vizVariable))
    
    
    #figRestrict= glucifer.Figure()
    #figRestrict.append( glucifer.objects.Points(gSwarm,respltFn))
    #figRestrict.append( glucifer.objects.Points(gSwarm,lithRestFn))
    #figRestrict.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
    #figRestrict.append( glucifer.objects.Points(gSwarm,hinge180RestFn))
    #figRestrict.append( glucifer.objects.Points(gSwarm,interfaceRestFn))
    #figRestrict.append( glucifer.objects.Points(interfaces[0].swarm, colours="Blue Blue", pointSize=2.0, colourBar=False) )
    #figRestrict.append( glucifer.objects.Points(interfaces[1].swarm, colours="Red Red", pointSize=2.0, colourBar=False) )
    #figRestrict.append( glucifer.objects.Points(slab_seg.swarm, colours="Black Black", pointSize=2.0, colourBar=False) )

elif figures == 'store':
    fullpath = os.path.join(outputPath + "gldbs/")
    store1 = glucifer.Store(fullpath + 'subduction1.gldb')
    store2 = glucifer.Store(fullpath + 'subduction2.gldb')
    store3 = glucifer.Store(fullpath + 'subduction3.gldb')
    store4 = glucifer.Store(fullpath + 'subduction4.gldb')
    store5 = glucifer.Store(fullpath + 'subduction5.gldb')



    figTemp = glucifer.Figure(store1,figsize=(300*np.round(md.aspectRatio,2),300))
    figTemp.append( glucifer.objects.Points(gSwarm,temperatureField, fn_mask=vizVariable))

    figVisc= glucifer.Figure(store2, figsize=(300*np.round(md.aspectRatio,2),300))
    figVisc.append( glucifer.objects.Points(gSwarm,viscosityMapFn, logScale=True, valueRange =[1.,1e5], fn_mask=vizVariable))


    figMech= glucifer.Figure(store3, figsize=(300*np.round(md.aspectRatio,2),300))
    figMech.append( glucifer.objects.Points(gSwarm,fnViscMin, valueRange =[0.,mechCount], fn_mask=vizVariable))
    
    figMat= glucifer.Figure(store4, figsize=(300*np.round(md.aspectRatio,2),300))
    figMat.append( glucifer.objects.Points(gSwarm,materialVariable, fn_mask=vizVariable))
    figMat.append( glucifer.objects.VectorArrows(mesh,velocityField,resolutionI=int(16*md.aspectRatio), resolutionJ=16*2,  scaling=0.0001))
    
    
    #figStress= glucifer.Figure(store5, figsize=(300*np.round(md.aspectRatio,2),300))
    #figStress.append( glucifer.objects.Points(gSwarm,stressVariableFn, fn_mask=vizVariable))
    #figStress.append( glucifer.objects.VectorArrows(mesh, eig1, arrowHead=0.0, scaling=0.05, glyphs=3, resolutionI=int(16*md.aspectRatio), resolutionJ=16*2))
    #figStress.append( glucifer.objects.VectorArrows(mesh, eig2, arrowHead=0.0, scaling=0.05, glyphs=3, resolutionI=int(16*md.aspectRatio), resolutionJ=16*2))

    
    
    


# In[ ]:

ndp


# In[ ]:




# if figures == 'gldb':
#     #Pack some stuff into a database as well
#     figDb = glucifer.Figure()
#     #figDb.append( glucifer.objects.Mesh(mesh))
#     figDb.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.0005))
#     #figDb.append( glucifer.objects.Points(gSwarm,tracerVariable, colours= 'white black'))
#     figDb.append( glucifer.objects.Points(gSwarm,materialVariable))
#     figDb.append( glucifer.objects.Points(swarmPlateBoundary, pointSize=4))
# 
#     #figDb.append( glucifer.objects.Points(gSwarm,viscMinVariable))
#     figDb.append( glucifer.objects.Points(vSwarm,fnViscMin))
#     #figDb.append( glucifer.objects.Points(gSwarm,fnViscMin))
#     #figDb.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True))
#     #figDb.append( glucifer.objects.Points(gSwarm, strainRate_2ndInvariant, logScale=True))
#     figDb.append( glucifer.objects.Points(vSwarm,temperatureField))
#     
#     
#     #figRestrict= glucifer.Figure()
#     #figRestrict.append( glucifer.objects.Points(gSwarm,respltFn))
#     #figRestrict.append( glucifer.objects.Points(gSwarm,lithRestFn))
#     #figRestrict.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
#     #figRestrict.append( glucifer.objects.Points(gSwarm,hinge180RestFn))
#     #figRestrict.append( glucifer.objects.Points(gSwarm,interfaceRestFn))
#     #figRestrict.append( glucifer.objects.Points(interfaces[0].swarm, colours="Blue Blue", pointSize=2.0, colourBar=False) )
#     #figRestrict.append( glucifer.objects.Points(interfaces[1].swarm, colours="Red Red", pointSize=2.0, colourBar=False) )
#     #figRestrict.append( glucifer.objects.Points(slab_seg.swarm, colours="Black Black", pointSize=2.0, colourBar=False) )
# 
# elif figures == 'store':
#     fullpath = os.path.join(outputPath + "gldbs/")
#     store = glucifer.Store(fullpath + 'subduction.gldb')
# 
#     figTemp = glucifer.Figure(store,figsize=(300*np.round(md.aspectRatio,2),300))
#     figTemp.append( glucifer.objects.Points(vSwarm,temperatureField))
# 
#     figVisc= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
#     figVisc.append( glucifer.objects.Points(gSwarm,viscosityMapFn, logScale=True, valueRange =[1.,1e5]))
# 
#     figMech= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
#     figMech.append( glucifer.objects.Points(vSwarm,fnViscMin))
#     
#     figMat= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
#     figMat.append( glucifer.objects.Points(gSwarm,materialVariable))

# **Miscellania**

# In[106]:

##############
#Create a numpy array at the surface to get surface information on (using parallel-friendly evaluate_global)
##############

surface_xs = np.linspace(mesh.minCoord[0], mesh.maxCoord[0], mesh.elementRes[0] + 1)
surface_nodes = np.array(zip(surface_xs, np.ones(int(len(surface_xs)*mesh.maxCoord[1])))) #For evaluation surface velocity
normgradV = velocityField.fn_gradient[0]/fn.math.sqrt(velocityField[0]*velocityField[0])

tempMM = fn.view.min_max(temperatureField)
dummy = tempMM.evaluate(mesh)



# In[107]:

#Not parallel friendly yet
#tipXvels = np.sum(velocityField[0].evaluate(tipSwarm)) /\
#    tipSwarm.particleGlobalCount
#tipXvels = np.sum(velocityField[1].evaluate(tipSwarm)) /\
#    tipSwarm.particleGlobalCount
    
#tipXpos = np.sum(xFn.evaluate(tipSwarm)) /\
#    tipSwarm.particleGlobalCount
#tipYpos = np.sum(yFn.evaluate(tipSwarm)) /\
#    tipSwarm.particleGlobalCount


# In[108]:

##############
#These functions handle checkpointing
##############

def checkpoint1(step, checkpointPath,filename, filewrites):
    path = checkpointPath + str(step) 
    os.mkdir(path)
    ##Write and save the file, if not already a writing step
    if not step % filewrites == 0:
        f_o.write((35*'%-15s ' + '\n') % (areaintRock, tempintRock, rmsintRock, dwintRock, vdintRock,
                                  areaintLith, tempintLith,rmsintLith, dwintLith, vdintLith,
                                  areaintLower, tempintLower, rmsintLower, dwintLower, vdintLower, 
                                  areaintHinge180,vdintHinge180, areaintHinge60, vdintHinge60, 
                                  areaintInterface, vdintInterface, vdintInterface,
                                  minVel,maxVel, minVxsurf, maxVxsurf, surfLength,            
                                  rmsSurf, nuTop, nuBottom, plateness, ndp.subzone,ndp.lRidge, ndp.rRidge, realtime))
    filename.close()
    shutil.copyfile(os.path.join(outputPath, outputFile), os.path.join(path, outputFile))


def checkpoint2(step, checkpointPath, swarm, filename, varlist = [materialVariable], varnames = ['materialVariable']):
    path = checkpointPath + str(step) 
    velfile = "velocityField" + ".hdf5"
    tempfile = "temperatureField" + ".hdf5"
    pressfile = "pressureField" + ".hdf5"
    velocityField.save(os.path.join(path, velfile))
    temperatureField.save(os.path.join(path, tempfile))
    pressureField.save(os.path.join(path, pressfile))
    swarm.save(os.path.join(path, "swarm.h5") ) 
    for ix in range(len(varlist)):
        varb = varlist[ix]
        varb.save(os.path.join(path,varnames[ix] + ".h5"))
    
    #Save the parameters
    dict_list = [dp, sf, ndp, md] #if any of the dictionaries have changed, this list needs to be rebuilt
    save_pickles(dict_list, dict_names, path)
    
#Simple Checkpoint function for the faults / interfaces (markerLine2D)
def checkpoint3(step,  checkpointPath, interfaces,interfacenames ):
    path = checkpointPath + str(step)
    for ix in range(len(interfaces)):
        intf = interfaces[ix]
        intf.swarm.save(os.path.join(path,interfacenames[ix] + ".h5"))
    
    


# In[109]:

def plate_infoFn(velocityField,xFn,  depthLimit, xsearchlim = 1.0, currentloc = 0.0, plateType='convergent'):
    """
    This functions is used to track the plate boundaries in 2D.
    It returns the x value of the max / min of the x velocity gradient.
    
    """
    
    #Restriction fucntion 
    restictionFn = fn.branching.conditional([(operator.and_(fn.math.abs(xFn - currentloc) < xsearchlim, depthFn < depthLimit) , 1.),                                       
                                             (True, 0.0)])
    
    #Normalised surface vel gradient
    normgradV = velocityField.fn_gradient[0]/fn.math.sqrt(velocityField[0]*velocityField[0])

    
    plateFn0 = normgradV*restictionFn
    
    plateFn = fn.branching.conditional([(plateFn0 > 1e20, 0.),             #This attempts to fix inf values that appear at sidewalls                          
                                             (True, plateFn0)])
    
    extremFn = fn.view.min_max(plateFn)
    dummy = extremFn.evaluate(mesh)
    
    
    print(extremFn.min_global(), extremFn.max_global())

    #get global max
    if plateType == 'convergent':   
        testFn = fn.branching.conditional([(extremFn < extremFn.min_global() + 1e-10, coordinate[0]),                                       
                                             (True, (-9999999.))])
        extremFn2 = fn.view.min_max(testFn)
        dummy2 = extremFn2.evaluate(mesh)
        if extremFn2.max_global() == -9999999.:
            return currentloc
        else:
            return extremFn2.max_global()
        
    elif plateType == 'divergent':
        testFn = fn.branching.conditional([(extremFn > extremFn.max_global() - 1e-3, coordinate[0]),                                       
                                             (True, (-9999999.))])
        extremFn2 = fn.view.min_max(testFn)
        dummy2 = extremFn2.evaluate(mesh)
        if extremFn2.max_global() == -9999999.:
            return currentloc
        else:
            return extremFn2.max_global()
    else:
        raise ValueError('plateType should be one of convergent/divergent')


# In[110]:

##############
#This will allow us to evaluate viscous shear heating, and add the result directly to the temperature field
##############

viscDisMapFn = 4.*viscosityMapFn*sinner
viscDisFnmesh = uw.mesh.MeshVariable(mesh,nodeDofCount=1)
viscDisProj = uw.utils.MeshVariable_Projection( viscDisFnmesh, viscDisMapFn)
viscDisProj.solve()


# In[111]:

# initialise timer for computation
start = time.clock()


#increment time for our next image / file dump, should work for checkpoint restarts
next_image_step = (np.floor(realtime/files_freq)+ 1.) *files_freq 


# Main simulation loop
# -----
# 

# In[ ]:

#while step < 21:
while realtime < 1.:

    # solve Stokes and advection systems
    solver.solve(nonLinearIterate=True)
    
    
    #remove drift /null space in pressure
    pressureField.data[:] -= pressureSurf/surfLength
    
    files_this_step = False #don't write files / images unless conditions are triggered
    dt = advDiff.get_max_dt()*md.courantFac
    if step == 0:
        dt = 0.
        files_this_step = True #write files on the zeroth step
    
    #This relates to file writing at set period - override make sure we hit certain time values
    if realtime + dt >= next_image_step:
        dt = next_image_step - realtime
        files_this_step = True
        next_image_step += files_freq #increment time for our next image / file dump
        
    advDiff.integrate(dt)
    materialadvector.integrate(dt)
    #advect any interfaces
    fault.advection(dt)
    
    #Add the adiabatic adjustment 
    temperatureField.data[:] += dt*abHeatFn.evaluate(mesh)
    
    #Add the viscous heating term
    #Need to be skeptical of Projection for viscosity. Also try direct interpolation (IDW)
    #viscDisProj = uw.utils.MeshVariable_Projection( viscDisFnmesh, viscDisMapFn)
    #viscDisProj.solve()
    #temperatureField.data[:] += (ndp.Di/ndp.RA)*dt*viscDisFnmesh.evaluate(mesh)
    

    # Increment
    realtime += dt
    step += 1
    timevals.append(realtime)
    
    ################
    #Update temperature field in the air region
    #Do this better...
    ################
    if (step % sticky_air_temp == 0):
        for index, coord in enumerate(mesh.data):
            if coord[1] >= 1.:
                temperatureField.data[index] = ndp.TSP
                
                
    ################
    #Files output
    ################ 
    if files_this_step:
        
        gSwarm.update_particle_owners()
        
        #any fields / swarms to be saved go here
        
        #Save the fault swarm
        fnametemp1 = "faultSwarm" + "_" + str(step)
        fullpath1 = os.path.join(outputPath + "files/" + fnametemp1)
        fault.swarm.save(fullpath1)
             
        
        #Now the surface x vels.
        try:
            surfaceVelx.data[...] = velocityField[0].evaluate(surface.swarm)
        except:
            pass
        fnametemp = "velxSurface" + "_" + str(step) + '.h5'
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        surfaceVelx.save(fullpath)
        
        #Now the vertical stress at the surface, 
        ix1, weights1 = nn_evaluation(gSwarm, surface.swarm.particleCoordinates.data, n=10, weighted=True)
        
        try:
            surfaceStrainY.data[:] = sym_strainRate[1].evaluate(surface.swarm)#* \
        except:
            pass
        fnametemp = "surfaceStrainY" + "_" + str(step) + '.h5'
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        surfaceStrainY.save(fullpath)
        
        fnametemp = "surfaceVisc" + "_" + str(step) + '.h5'
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        try:
            surfaceVisc.data[:,0] =    np.average(viscosityMapFn.evaluate(gSwarm)[ix1][:,:,0], weights=weights1, axis=1)
        except:
            pass
        surfaceVisc.save(fullpath)

        
        fnametemp = "temperatureField" + "_" + str(step) + ".h5"
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        temperatureField.save(fullpath)
        
        try:
            surfacePressure.data[:] = pressureField.evaluate(surface.swarm)
        except:
            pass
        fnametemp = "surfacePressure" + "_" + str(step) + ".h5"
        fullpath = os.path.join(outputPath + "files/" + fnametemp)
        surfacePressure.save(fullpath)
        
        

                          

    ################
    #Update the subduction zone / plate information
    ################ 
    
    comm.barrier()
    if (step % metric_output == 0):
        

        ndp.subzone = plate_infoFn(velocityField,xFn,  30e3/dp.LS, xsearchlim = 300e3/dp.LS, currentloc = ndp.subzone, plateType='convergent')
        ndp.lRidge = plate_infoFn(velocityField,xFn,  30e3/dp.LS, xsearchlim = 300e3/dp.LS, currentloc = ndp.lRidge, plateType='divergent')
        ndp.rRidge = plate_infoFn(velocityField,xFn,  30e3/dp.LS, xsearchlim = 300e3/dp.LS, currentloc = ndp.rRidge, plateType='divergent')

    
         
            
    ################
    # Calculate the Metrics
    ################
    
    if (step % metric_output == 0):
        ###############
        #Rebuild the restriction functions where necessary
        ###############
        
        #Lithosphere - mantle:
        mantleconditions = [ (                                  operator.and_(temperatureField < 0.9*ndp.TPP, operator.and_(xFn> ndp.lRidge,xFn< ndp.rRidge )), 1.),
                   (                                                   True , 0.) ]
        lithRestFn = fn.branching.conditional(mantleconditions)
        lithRestFn*=rockRestFn #Add next level up in heirarchy
        
        #Lower plate
        
        lowerPlateRestFn = gSwarm.add_variable( dataType="double", count=1 ) #weirdly we have to rebuild this one...
        lowerPlateRestFn.data[:] = 0.0
        sd, pts0 = fault.compute_signed_distance(gSwarm.particleCoordinates.data, distance=w0)
        sp, pts0 = fault.compute_marker_proximity(gSwarm.particleCoordinates.data, distance=w0)
        lowerPlateRestFn.data[np.logical_and(sd>0,sp == fault.ID)] = sp[np.logical_and(sd>0,sp == fault.ID)]
        lowerPlateRestFn *= lithRestFn #Add next level up in heirarchy
        
        #Hinge
        
        hinge60Spatialconditions = [ (           operator.and_( (depthFn < MAXY - (60e3/dp.LS)),  (xFn > ndp.subzone - 60e3/dp.LS)), 1.),
                   (                                                   True , 0.) ]
        hinge60RestFn = fn.branching.conditional(hinge60Spatialconditions)
        hinge60RestFn*=lowerPlateRestFn #Add next level up in hierarchy



        hinge180Spatialconditions = [ (           operator.and_( (depthFn < MAXY - (180e3/dp.LS)),  (xFn > ndp.subzone - 180e3/dp.LS)), 1.),
                           (                                                   True , 0.) ]
        hinge180RestFn = fn.branching.conditional(hinge180Spatialconditions)
        hinge180RestFn*=lowerPlateRestFn #Add next level up in hierarchy
        
        
        #Reevaluate max min Functions
        dummyFn = _maxMinVel.evaluate(mesh)
        dummyFn = _maxMinSr.evaluate(mesh)
        dummyFn = _maxMinVxSurf.evaluate(tWalls)
        
        ###############
        #Metrics
        ###############
        areaintRock = _areaintRock.evaluate()[0] #trivial except when using sticky air
        tempintRock = _tempintRock.evaluate()[0]
        rmsintRock = _rmsintRock.evaluate()[0]
        dwintRock = _dwintRock.evaluate()[0]
        vdintRock = _vdintRock.evaluate()[0]
        areaintLith = _areaintLith.evaluate()[0]
        tempintLith = _tempintLith.evaluate()[0]
        rmsintLith = _rmsintLith.evaluate()[0]
        dwintLith = _dwintLith.evaluate()[0]
        vdintLith = _vdintLith.evaluate()[0]
        areaintLower = _areaintLower.evaluate()[0]
        tempintLower = _tempintLower.evaluate()[0]
        rmsintLower = _rmsintLower.evaluate()[0]
        dwintLower = _dwintLower.evaluate()[0]
        vdintLower = _vdintLower.evaluate()[0]
        vdintHinge180 = _vdintHinge180.evaluate()[0]
        areaintHinge180 = _areaintHinge180.evaluate()[0]
        vdintHinge60 = _vdintHinge60.evaluate()[0]
        areaintHinge60= _areaintHinge60.evaluate()[0]
        vdintInterface = _vdintInterface.evaluate()[0]
        areaintInterface= _areaintInterface.evaluate()[0]
        #Surface integrals
        rmsSurf = _rmsSurf.evaluate()[0]
        nuTop = _nuTop.evaluate()[0]
        nuBottom = _nuBottom.evaluate()[0]
        plateness = _plateness.evaluate()[0]
        #extrema
        maxVel = _maxMinVel.max_global()
        minVel = _maxMinVel.min_global() 
        maxSr = _maxMinSr.max_global()
        minSr = _maxMinSr.min_global()
        maxVxsurf = _maxMinVxSurf.max_global()
        minVxsurf = _maxMinVxSurf.min_global()
        # output to text file...root proc. handles this one
        if uw.rank()==0:
            f_o.write((35*'%-15s ' + '\n') % (areaintRock, tempintRock, rmsintRock, dwintRock, vdintRock,
                                  areaintLith, tempintLith,rmsintLith, dwintLith, vdintLith,
                                  areaintLower, tempintLower, rmsintLower, dwintLower, vdintLower, 
                                  areaintHinge180,vdintHinge180, areaintHinge60, vdintHinge60, 
                                  areaintInterface, vdintInterface, vdintInterface,
                                  minVel,maxVel, minVxsurf, maxVxsurf, surfLength,            
                                  rmsSurf, nuTop, nuBottom, plateness, ndp.subzone,ndp.lRidge, ndp.rRidge, realtime))
    
    
    ################
    #Also repopulate entire swarm periodically
    ################
    if step % swarm_repop == 0:
        population_control.repopulate()
        
        
                       
    ################
    #Gldb output
    ################ 
    if files_this_step: 
        
        #ReBuild the principal stress vector
        ssr = sym_strainRate.evaluate(mesh) #this does need to be here, 
        
        principalAngles  = np.apply_along_axis(eig2d, 1, ssr[:, :])[:,2]
        eig1.data[:,0] = np.cos(np.radians(principalAngles - 90.)) #most compressive 
        eig1.data[:,1] = np.sin(np.radians(principalAngles - 90.))
        eig2.data[:,0] = np.cos(np.radians(principalAngles ))      #most extensive
        eig2.data[:,1] = np.sin(np.radians(principalAngles ))
        
        
        
        #Rebuild the viz. mask
        
        vizVariable.data[:] = 0
        depthVariable.data[:] = depthFn.evaluate(gSwarm)
        for index, value in enumerate(depthVariable.data[:]):
            #print index, value
            if np.random.rand(1)**5 > value/(MAXY - MINY):
                vizVariable.data[index] = 1

        del index, value    #get rid of any variables that might be pointing at the .data handles (these are!)
        
        
        
        if figures == 'gldb':
            #Remember to rebuild any necessary swarm variables
            fnamedb = "dbFig" + "_" + str(step) + ".gldb"
            fullpath = os.path.join(outputPath + "gldbs/" + fnamedb)
            figDb.save_database(fullpath)
            
            #Temp figure
            #fnamedb = "restrictFig" + "_" + str(step) + ".gldb"
            #fullpath = os.path.join(outputPath + "gldbs/" + fnamedb)
            #figRestrict.save_database(fullpath)
        elif figures == 'store':      
            fullpath = os.path.join(outputPath + "gldbs/")
            store1.step = step
            store2.step = step
            store3.step = step
            store4.step = step
            store5.step = step
            #Save figures to store
            figVisc.save( fullpath + "Visc" + str(step).zfill(4))
            figMech.save( fullpath + "Mech" + str(step).zfill(4))
            figTemp.save( fullpath + "Temp"    + str(step).zfill(4))
            figMat.save( fullpath + "Mat"    + str(step).zfill(4))
            #figStress.save( fullpath + "Stress"    + str(step).zfill(4))
            #figSr.save( fullpath + "Str_rte"    + str(step).zfill(4))
            
            
            #Also checkpoint so gLuc Stores stay in sync with checkpoint
            if step % checkpoint_every != 0:
                if uw.rank() == 0:
                    checkpoint1(step, checkpointPath,f_o, metric_output)           
                checkpoint2(step, checkpointPath, gSwarm, f_o, varlist = varlist, varnames = varnames)
                checkpoint3(step,  checkpointPath, interfaces,interfacenames )
                f_o = open(os.path.join(outputPath, outputFile), 'a') #is this line supposed to be here?
            
            
    ################
    #xdmf output
    ################ 
    
    fullpath = os.path.join(outputPath + "xdmf/")
    if files_this_step:
        
        # rebuild the sigma_xx variable
        _ix, _weights = nn_evaluation(gSwarm, mesh.data, n=5, weighted=True)
        stressField.data[:,0] = np.average(stressVariableFn.evaluate(gSwarm)[_ix][:,:,0],weights=_weights, axis=1)

        
        
        fullpath = os.path.join(outputPath + "xdmf/")
        #if not os.path.exists(fullpath+"mesh.h5"):
        #    _mH = mesh.save(fullpath+"mesh.h5")
        
        try:
            _mH
        except:
            _mH = mesh.save(fullpath+"mesh.h5")
        
        mh = _mH
        vH = velocityField.save(fullpath + "velocity_" + str(step) +".h5")
        tH = temperatureField.save(fullpath + "temp_" + str(step) + ".h5")
        eH = eig1.save(fullpath + "eig_" + str(step) + ".h5")
        eH2 = eig2.save(fullpath + "eig2_" + str(step) + ".h5")
        sigXX = stressField.save(fullpath + "sigXX_" + str(step) + ".h5")
        velocityField.xdmf(fullpath + "velocity_" + str(step), vH, 'velocity', mh, 'mesh', modeltime=realtime)
        temperatureField.xdmf(fullpath + "temp_" + str(step), tH, 'temperature', mh, 'mesh', modeltime=realtime)
        eig1.xdmf(fullpath + "eig_" + str(step), eH, 'eig', mh, 'mesh', modeltime=realtime)
        eig2.xdmf(fullpath + "eig2_" + str(step), eH2, 'eig2', mh, 'mesh', modeltime=realtime)
        stressField.xdmf(fullpath + "sigXX_" + str(step), sigXX, 'sigXX', mh, 'mesh', modeltime=realtime)
        
    ################
    #Particle update
    ###############    
    #ageVariable.data[:] += dt #increment the ages (is this efficient?)
    ageDT += dt
    
    if step % swarm_update == 0:
        
        #Increment age stuff. 
        ageConditions = [ (depthFn < ndp.AGETRACKDEPTH, ageVariable + ageDT ),  #add ageDThere
                  (True, 0.) ]
        ageVariable.data[:] = fn.branching.conditional( ageConditions ).evaluate(gSwarm)        
        ageDT = 0. #reset the age incrementer
        
        
        #This is hardcoded to assume subduction is towards the right
        tempop = operator.lt

        #Update the relevant parts of the material graph
        #Remove and rebuild edges related to crust
        DG.remove_edges_from([(mantleIndex,crustIndex)])
        DG.add_edges_from([(mantleIndex,crustIndex)])
        DG.remove_edges_from([(harzIndex,crustIndex)])
        DG.add_edges_from([(harzIndex,crustIndex)])

        #... to crust
        DG.add_transition((mantleIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
        DG.add_transition((mantleIndex,crustIndex), xFn, operator.lt, 0.5*ndp.lRidge) #No crust on the upper plate
        DG.add_transition((mantleIndex,crustIndex), ageVariable, operator.gt, 0.2)

        DG.add_transition((harzIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
        DG.add_transition((harzIndex,crustIndex), xFn, tempop, 0.5*ndp.lRidge) #This one sets no crust on the upper plate
        DG.add_transition((harzIndex,crustIndex), ageVariable, operator.gt, crustageCond)
        
        
        #Make sure to rebuild the condition list if you want changes to be applied
        DG.build_condition_list(materialVariable)
        
        comm.barrier()
        
    
        
        
        #Apply any materialVariable changes
        for i in range(2): #go through twice
            materialVariable.data[:] = fn.branching.conditional(DG.condition_list).evaluate(gSwarm)
        
        #Also update any information related to faults / interfaces:

        sd, pts0 = fault.compute_signed_distance(gSwarm.particleCoordinates.data, distance=w0)
        sp, pts0 = fault.compute_marker_proximity(gSwarm.particleCoordinates.data)

        proximityVariable.data[np.logical_and(sd<0,sp == fault.ID)] = sp[np.logical_and(sd<0,sp == fault.ID)]

        dv, nzv = fault.compute_normals(gSwarm.particleCoordinates.data)
        directorVector.data[nzv] = dv[nzv]

        proximityVariable.data[gSwarm.particleCoordinates.data[:,1]  < (1. - ndp.CRUSTVISCUTOFF)] = 0.
        
        #And add extra particles to interfaces as necessary
        #subduction fault
        #introPoint = ndp.subzone - abs(ndp.subzone - ndp.lRidge)/2. #half way between ridge and Sz
        introPoint = ndp.subzone - abs(ndp.subzone - mesh.minCoord[0])/2. #half way between ridge and domain edge
        fault.add_points([introPoint],[MAXY - faultthickness])


        
    ################
    #Checkpoint
    ################
    if step % checkpoint_every == 0:
        if uw.rank() == 0:
            checkpoint1(step, checkpointPath,f_o, metric_output)           
        checkpoint2(step, checkpointPath, gSwarm, f_o, varlist = varlist, varnames = varnames)
        checkpoint3(step,  checkpointPath, interfaces,interfacenames )
        f_o = open(os.path.join(outputPath, outputFile), 'a') #is this line supposed to be here?
        
    
    
f_o.close()
print 'step =',step


# In[ ]:




# In[ ]:

#viscVariable = gSwarm.add_variable( dataType="float", count=1 )
#viscVariable.data[:] = viscosityMapFn.evaluate(gSwarm)

#buoyVariable = gSwarm.add_variable( dataType="float", count=1 )
#buoyVariable.data[:] = densityMapFn.evaluate(gSwarm)


# In[70]:

#Level 3. lithosphere - mantle:
tempMM = fn.view.min_max(temperatureField)
tempMM.evaluate(mesh)
TMAX = tempMM.max_global()
mantleconditions = [ (                                  temperatureField < 0.9*ndp.TPP, 1.),
                   (                                                   True , 0.) ]
lithRestFn = fn.branching.conditional(mantleconditions)
#lithRestFn*=rockRestFn #Add next level up in heirarchy


# In[207]:

1.1*ndp.TPP, 0.6*TMAX


# In[71]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
#fig.append( glucifer.objects.Points(gSwarm, materialVariable))
#fig.append( glucifer.objects.Surface(mesh, temperatureField ))#
#fig.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.002))
#fig.append( glucifer.objects.Surface(mesh,densityMapFn))
#fig.append( glucifer.objects.Surface(mesh,raylieghFn))
#fig.append( glucifer.objects.Points(fault.swarm, pointSize=4))
#fig.show()
#fig.save_database('test.gldb')


# In[78]:

#print('proc #' + str(comm.rank) + " " + str(Max_vx_surf ) )
#if uw.rank()==0:
#    print('Proc 0' +  " " + str(Max_vx_surf ) )


# In[ ]:

#figDb.save_database('test.gldb')


# In[134]:

#Add the viscous heating term
#Need to be skeptical of Projection for viscosity. Also try direct interpolation (IDW)
viscDisProj = uw.utils.MeshVariable_Projection( viscDisFnmesh, viscDisMapFn)
viscDisProj.solve()
#temperatureField.data[:] += ndp.Di*dt*viscDisFnmesh.evaluate(mesh)


# In[141]:

dt = advDiff.get_max_dt()*md.courantFac
test = (ndp.Di/ndp.RA)*dt*viscDisFnmesh


# In[69]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
#fig.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True, valueRange =[1e-3,1e5]))
#fig.append( glucifer.objects.Surface(mesh, stressField))#

#fig.show()


# In[143]:

#test.evaluate(mesh).max()


# In[90]:

#figMech= glucifer.Figure(figsize=(300*np.round(md.aspectRatio,2),300))
#figMech.append( glucifer.objects.Points(gSwarm,fnViscMin, valueRange =[0.,mechCount]))


# In[91]:

#figMech.show()


# In[121]:

#viscMinConditions


# In[92]:

#np.unique(fnViscMin.evaluate(gSwarm))


# In[ ]:



