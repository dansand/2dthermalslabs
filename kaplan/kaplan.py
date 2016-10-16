
# coding: utf-8

# ## Thermal subduction, linear rheology:
# 
# 
# The viscous rheology in this model is similar to the models described in the PhD thesis of Micheal Kaplan
# 
# 
# 
# **Keywords:** subduction, thermally-activated creep, 
# 
# 
# **References:**
# 
# 
# Kaplan, Michael. Numerical Geodynamics of Solid Planetary Deformation. Diss. University of Southern California, 2015.

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


# Model name and directories
# -----

# In[2]:

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


# In[3]:

###########
#Standard output directory setup
###########

outputPath = "results" + "/" +  str(Model) + "/" + str(ModNum) + "/" 
imagePath = outputPath + 'images/'
filePath = outputPath + 'files/'
checkpointPath = outputPath + 'checkpoint/'
dbPath = outputPath + 'gldbs/'
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

        
comm.Barrier() #Barrier here so no procs run the check in the next cell too early


# In[4]:

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


# In[5]:

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

# In[6]:

u = pint.UnitRegistry()
cmpery = 1.*u.cm/u.year
mpermy = 1.*u.m/u.megayear
year = 1.*u.year
spery = year.to(u.sec)
cmpery.to(mpermy)


# In[7]:

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
# 

# In[8]:

###########
#Parameter / settings dictionaries get saved&loaded using pickle
###########
 
dp = edict({}) #dimensional parameters
sf = edict({}) #scaling factors
ndp = edict({}) #dimensionless paramters
md = edict({}) #model paramters, flags etc
#od = edict({}) #output frequencies



# In[9]:

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


# In[10]:

###########
#Store the physical parameters, scale factors and dimensionless pramters in easyDicts
#Mainly helps with avoiding overwriting variables
###########


dp = edict({'LS':2900*1e3, #Scaling Length scale
            'depth':670*1e3, #Depth of domain
            'rho':3300.,  #reference density
            'g':9.8, #surface gravity
            'eta0':5e20, #reference viscosity
            'k':1e-6, #thermal diffusivity
            'a':3e-5, #surface thermal expansivity
            'R':8.314, #gas constant
            'TP':1673., #mantle potential temp (K)
            'TS':273., #surface temp (K)
            #Rheology - flow law paramters
            'Adf':3e-11, #pre-exp factor for diffusion creep
            'Edf':1e5, #Total viscosity variation in the Kaplan model
            'cm':40e6, #mantle cohesion in Byerlee law
            'cc':40e6, #mantle cohesion in Byerlee law
            'ci':40e6, #mantle cohesion in Byerlee law
            'cf':40e6, #mantle cohesion in Byerlee law
            'fcm':0.03,   #mantle friction coefficient in Byerlee law (tan(phi))
            'fcc':0.03,   #crust friction coefficient 
            'fci':0.03,   #subduction interface friction coefficient
            'fcf':0.03,   #subduction interface friction coefficient
            #Rheology - cutoff values
            'eta_min':1e17, 
            'eta_max':1e25, #viscosity max in the mantle material
            'eta_min_crust':2e19, #viscosity min in the weak-crust material
            'eta_max_crust':2e19, #viscosity max in the weak-crust material
            'eta_min_interface':5e19, #viscosity min in the subduction interface material
            'eta_max_interface':5e19, #viscosity max in the subduction interface material
            'eta_min_fault':5e19, #viscosity min in the subduction interface material
            'eta_max_fault':5e19, #viscosity max in the subduction interface material
            #Length scales
            'MANTLETOCRUST':8.*1e3, #Crust depth
            'HARZBURGDEPTH':40e3,
            'CRUSTTOMANTLE':800.*1e3,
            'LITHTOMANTLE':(900.*1e3),
            'MANTLETOLITH':200.*1e3, 
            'TOPOHEIGHT':10.*1e3,  #rock-air topography limits
            'CRUSTTOECL':100.*1e3,
            'LOWMANTLEDEPTH':660.*1e3, 
            'CRUSTVISCUTOFF':150.*1e3, #Deeper than this, crust material rheology reverts to mantle rheology
            'AGETRACKDEPTH':100e3, #above this depth we track the age of the lithsphere (below age is assumed zero)
            #Slab and plate parameters
            'roc':250e3,     #radius of curvature of slab
            'subzone':0.0,   #X position of subduction zone..
            'lRidge':-0.5*(670e3*6),  #For depth = 670 km, aspect ratio of 6, this puts the ridges at MINX, MAXX
            'rRidge':0.5*(670e3*6),
            'maxDepth':250e3,
            'theta':70., #Angle to truncate the slab (can also control with a maxDepth param)
            'slabmaxAge':60e6, #age of subduction plate at trench
            'platemaxAge':60e6, #max age of slab (Plate model)
            'sense':'Right', #dip direction
            'op_age_fac':0.5, #this controls the overidding plate age reduction
            #Misc
            'StALS':100e3, #depth of sticky air layer
            'Steta_n':1e19, #stick air viscosity, normal
            'Steta_s':1e18, #stick air viscosity, shear 
            'plate_vel':4,
            'low_mantle_visc_fac':30.
           })

#append any derived parameters to the dictionary
dp.deltaT = dp.TP - dp.TS




# In[15]:

#Modelling and Physics switches

md = edict({'refineMesh':True,
            'stickyAir':False,
            'subductionFault':False,
            'symmetricIcs':False,
            'velBcs':False,
            'aspectRatio':6,
            'compBuoyancy':False, #use compositional & phase buoyancy, or simply thermal
            'periodicBcs':False,
            'RES':64,
            'elementType':"Q1/dQ0"
            })


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


# In[19]:

#print('refine Mesh is: ', md.refineMesh)


# In[20]:

#Only build these guys first time around, otherwise the read from checkpoints
#Important because some of these params (like SZ location) may change during model evolution


if not checkpointLoad:

    
    
    sf = edict({'stress':dp.LS**2/(dp.k*dp.eta0),
                'lith_grad':dp.rho*dp.g*(dp.LS)**3/(dp.eta0*dp.k) , 
                'vel':dp.LS/dp.k,
                'SR':dp.LS**2/dp.k    
               })

    #dimensionless parameters

    ndp = edict({'RA':(dp.g*dp.rho*dp.a*(dp.TP - dp.TS)*(dp.LS)**3)/(dp.k*dp.eta0),
                 'Edf':math.log(dp.Edf),
                 'TSP':0., 
                 'TBP':1.,
                 'TPP':(dp.TP - dp.TS)/dp.deltaT, #dimensionless potential temp
                 'TS':dp.TS/dp.deltaT,
                 'TP':dp.TP/dp.deltaT,
                 'cm':dp.cm*sf.stress,
                 'cc':dp.cc*sf.stress,    #{dimensionless cohesion in mantle, crust, interface}
                 'ci':dp.ci*sf.stress,
                 'cf':dp.cf*sf.stress,
                 'fcmd':dp.fcm*sf.lith_grad, 
                 'fccd':dp.fcc*sf.lith_grad, #{dimensionless friction coefficient in mantle, crust, interface}
                 'fcid':dp.fci*sf.lith_grad, 
                 'fcfd':dp.fci*sf.lith_grad, 
                 #Rheology - cutoff values
                 'eta_min':dp.eta_min/dp.eta0, 
                 'eta_max':dp.eta_max/dp.eta0, #viscosity max in the mantle material
                 'eta_min_crust':dp.eta_min_crust/dp.eta0, #viscosity min in the weak-crust material
                 'eta_max_crust':dp.eta_max_crust/dp.eta0, #viscosity max in the weak-crust material
                 'eta_min_interface':dp.eta_min_interface/dp.eta0, #viscosity min in the subduction interface material
                 'eta_max_interface':dp.eta_max_interface/dp.eta0, #viscosity max in the subduction interface material
                 'eta_min_fault':dp.eta_min_fault/dp.eta0, #viscosity min in the subduction interface material
                 'eta_max_fault':dp.eta_max_fault/dp.eta0, #viscosity max in the subduction interface material   
                 #Length scales
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
                 'roc':dp.roc/dp.LS,     #radius of curvature of slab
                 'subzone':dp.subzone/dp.LS,   #X position of subduction zone..
                 'lRidge':np.round(dp.lRidge/dp.LS, 1),  #For depth = 670 km, aspect ratio of 4, this puts the ridges at MINX, MAXX
                 'rRidge':np.round(dp.rRidge/dp.LS),
                 'maxDepth':dp.maxDepth/dp.LS,    
                 #misc
                 'Steta_n':dp.Steta_n/dp.eta0, #stick air viscosity, normal
                 'Steta_s':dp.Steta_n/dp.eta0, #stick air viscosity, shear 
                 'StALS':dp.StALS/dp.LS,
                 'plate_vel':sf.vel*dp.plate_vel*(cmpery.to(u.m/u.second)).magnitude,
                 'low_mantle_visc_fac':dp.low_mantle_visc_fac
                })



    #Append any more derived paramters
    ndp.StRA = (3300.*dp.g*(dp.LS)**3)/(dp.eta0 *dp.k) #Composisitional Rayleigh number for rock-air buoyancy force
    dp.CVR = (0.1*(dp.k/dp.LS)*ndp.RA**(2/3.))
    ndp.CVR = dp.CVR*sf.vel #characteristic velocity


# In[23]:

ndp.eta_min_crust


# In[24]:

ndp.plate_vel, sf.vel, (cmpery.to(u.m/u.second)).magnitude


# **Model setup parameters**

# In[51]:

###########
#Model setup parameters
###########

dim = 2          # number of spatial dimensions


#Domain and Mesh paramters
Xres = int(md.RES*8)   #more than twice the resolution in X-dim, which will be 'corrected' by the refinement we'll use in Y


MINY = 1. - (dp.depth/dp.LS)
if md.stickyAir:
    Yres = int(md.RES)
    MAXY = 1. + dp.StALS/dp.LS #150km
    
else:
    Yres = int(md.RES)
    MAXY = 1.


periodic = [False, False]
if md.periodicBcs:
    periodic = [True, False]
    
    
hw = np.round(0.5*(dp.depth/dp.LS)*md.aspectRatio, 1)
MINX = -1.*hw

MAXX = hw
MAXY = 1.
    
    

Xres = int(md.RES*md.aspectRatio) #careful
#if MINY == 0.5:
#    Xres = int(2.*RES*md.aspectRatio)
    

if md.stickyAir:
    Yres = int(md.RES)
    MAXY = np.round(MAXY + dp.StALS/dp.LS, 2)
    
else:
    Yres = int(md.RES)
    MAXY = np.round(MAXY, 2)

periodic = [False, False]
if md.periodicBcs:
    periodic = [True, False]
    
#elementType = "Q1/dQ0"


#System/Solver stuff
PIC_integration=True
ppc = 25

#Metric output stuff
figures =  'store' #glucifer Store won't work on all machines, if not, set to 'gldb' 
swarm_repop, swarm_update = 10, 10
gldbs_output = 100
checkpoint_every, files_output = 200, 200
metric_output = 10
sticky_air_temp = 1e6


# In[ ]:




# Create mesh and finite element variables
# ------

# In[52]:

mesh = uw.mesh.FeMesh_Cartesian( elementType = (md.elementType),
                                 elementRes  = (Xres, Yres), 
                                 minCoord    = (MINX, MINY), 
                                 maxCoord    = (MAXX, MAXY), periodic=periodic)

velocityField       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
pressureField       = uw.mesh.MeshVariable( mesh=mesh.subMesh, nodeDofCount=1 )
temperatureField    = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )
temperatureDotField = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=1 )


# In[53]:

coordinate = fn.input()
depthFn = MAXY - coordinate[1] #a function providing the depth


xFn = coordinate[0]  #a function providing the x-coordinate
yFn = coordinate[1]


# In[54]:

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
    
    
    
s = 2.5
intensityFac = 1.5
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


# In[ ]:




# In[55]:

#fig= glucifer.Figure()
#fig.append(glucifer.objects.Mesh(mesh))
#fig.append( glucifer.objects.Surface(mesh,intensityFn, discrete=True))
#fig.show()
#fig.save_database('test.gldb')


# In[56]:

#THis is a hack for adding a sticky air domain, we refine MAXY and things like the temperature stencil work from Y = 1. 

if md.stickyAir:
    MAXY = 1.


# Initial conditions
# -------
# 

# In[ ]:




# In[57]:


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
    r_grad =  1./(abs(rMOR-sz))
    l_grad =  1./(abs(sz-lMOR))
    if conjugate_plate:
        ageFn = fn.branching.conditional([(operator.and_(xFn > lMOR, xFn < sz) , (xFn + abs(lMOR))/(abs(sz-lMOR))), 
                                      (operator.and_(xFn < rMOR, xFn >= sz), (1.-(xFn + abs(sz))/abs(rMOR-sz))*opFac),
                                      (xFn > rMOR, r_grad*opFac*(xFn -abs(rMOR)) / (abs(MAXX-rMOR))),
                                      (True, l_grad*fn.math.abs((((xFn + abs(lMOR)) / (abs(lMOR - MINX))))))
                                         ])
    else:    
        
        ageFn = fn.branching.conditional([(operator.and_(xFn > lMOR, xFn < sz) , (xFn + abs(lMOR))/(abs(sz-lMOR))), 
                                      (operator.and_(xFn < rMOR, xFn >= sz), (1.-(xFn + abs(sz))/abs(rMOR-sz))*opFac),

                                      (True, 0.0)])
    return ageFn


# In[58]:

ndp.TPP


# In[59]:

###########
#Thermal initial condition - half-space cooling
###########

#  a few conversions
ageAtTrenchSeconds = min(dp.platemaxAge*(3600*24*365), dp.slabmaxAge*(3600*24*365))
#slab perturbation params (mostly dimensionless / model params here)
phi = 90. - dp.theta
Org = (ndp.subzone, MAXY-ndp.roc)

#First build the top TBL
ageFn = age_fn(xFn, sz =ndp.subzone, lMOR=ndp.lRidge,rMOR=ndp.rRidge, conjugate_plate=True, opFac = dp.op_age_fac)
#dimensionlize the age function
ageFn *= ageAtTrenchSeconds #seconds to year
w0 = (2.3*math.sqrt(dp.k*ageAtTrenchSeconds))/dp.LS #diffusion depth of plate at the trench

tempBL = (ndp.TPP - ndp.TSP)*fn.math.erf((depthFn*dp.LS)/(2.*fn.math.sqrt(dp.k*ageFn))) + ndp.TSP #boundary layer function
tempTBL =  fn.branching.conditional([(depthFn < w0, tempBL),
                          (True, ndp.TPP)])

if not md.symmetricIcs:
    if not checkpointLoad:
        out = uw.utils.MeshVariable_Projection( temperatureField, tempTBL) #apply function with projection
        out.solve()



# In[60]:

#w0*dp.LS


# In[61]:

#Now build the perturbation part
def inCircleFnGenerator(centre, radius):
    coord = fn.input()
    offsetFn = coord - centre
    return fn.math.dot( offsetFn, offsetFn ) < radius**2



#We use three circles to define our slab and crust perturbation,  
Oc = inCircleFnGenerator(Org , ndp.roc)
Ic = inCircleFnGenerator(Org , ndp.roc - w0)
Cc = inCircleFnGenerator(Org , ndp.roc - (1.2*ndp.MANTLETOCRUST)) #... weak zone on 'outside' of slab
Hc = inCircleFnGenerator(Org , ndp.roc - ndp.HARZBURGDEPTH) #... Harzburgite layer 
dx = (ndp.roc)/(np.math.tan((np.math.pi/180.)*phi))

#We'll also create a triangle which will truncate the circles defining the slab...
if dp.sense == 'Left':
    ptx = ndp.subzone - dx
else:
    ptx = ndp.subzone + dx
coords = ((0.+ ndp.subzone, MAXY), (0.+ ndp.subzone, MAXY-ndp.roc), (ptx, MAXY))
Tri = fn.shape.Polygon(np.array(coords))

#Actually apply the perturbation - could probably avoid particle walk here
if not md.symmetricIcs:
    if not checkpointLoad:
        sdFn = ((ndp.roc - fn.math.sqrt((coordinate[0] - Org[0])**2. + (coordinate[1] - Org[1])**2.)))
        slabFn = ndp.TPP*fn.math.erf((sdFn*dp.LS)/(2.*math.sqrt(dp.k*ageAtTrenchSeconds))) + ndp.TSP
        for index, coord in enumerate(mesh.data):
            if (
                Oc.evaluate(tuple(coord)) and
                Tri.evaluate(tuple(coord)) and not
                Ic.evaluate(tuple(coord)) and
                coord[1] > (MAXY - ndp.maxDepth)
                ): #In the quarter-circle defining the lithosphere
                temperatureField.data[index] = slabFn.evaluate(mesh)[index]


# In[62]:

#sdFn = ((RocM - fn.math.sqrt((coordinate[0] - Org[0])**2. + (coordinate[1] - Org[1])**2.)))
#slabFn = ndp.TPP*fn.math.erf((sdFn*dp.LS)/(2.*math.sqrt(dp.k*ageAtTrenchSeconds))) + ndp.TSP
#sdFn, slabFn


# In[63]:

#Make sure material in sticky air region is at the surface temperature.
for index, coord in enumerate(mesh.data):
            if coord[1] >= MAXY:
                temperatureField.data[index] = ndp.TSP


# In[64]:

#fn.math.erf((sdFn*dp.LS)/(2.*fn.math.sqrt(dp.k*(slabmaxAge*(3600*24*365))))) 
#CRUSTVISCUTOFF, MANTLETOCRUST*3


# def matplot_field(temperatureField, dp):
#     if uw.nProcs() != 1:
#         print("only in Serial folks")
#     else:
#         import matplotlib.pyplot as pyplt
#         try :
#             if(__IPYTHON__) :
#                 get_ipython().magic(u'matplotlib inline')
#         except NameError :
#             pass
#         field_data = temperatureField.data.reshape(mesh.elementRes[1] + 1, mesh.elementRes[0] + 1)
#         fig, ax = pyplt.subplots(figsize=(32,2))
#         ql = dp.LS/1e3
#         pyplt.ioff()
#         cax =ax.imshow(np.flipud(field_data), cmap='coolwarm', aspect = 0.5, extent=[0,ql*aspectRatio,ql, 0])
#         fig.colorbar(cax, orientation='horizontal' )
#         #ax.set_x([0,dp.LS*aspectRatio])
#         pyplt.tight_layout()
#         
#         return fig, ax
#         
# fig, ax = matplot_field(temperatureField, dp)
# fig.savefig('test.png')       

# In[65]:

temperatureField.data.min(), temperatureField.data.max()


# In[66]:

#fig= glucifer.Figure(quality=3)
#fig.append( glucifer.objects.Surface(mesh,temperatureField ))
#fig.append( glucifer.objects.Points(gSwarm,temperatureField ))
#fig.append( glucifer.objects.Mesh(mesh))
#fig.show()

##fig.save_database('test.gldb')
#fig.save_image('test.png')


# Boundary conditions
# -------

# In[37]:

for index in mesh.specialSets["MinJ_VertexSet"]:
    temperatureField.data[index] = ndp.TBP
for index in mesh.specialSets["MaxJ_VertexSet"]:
    temperatureField.data[index] = ndp.TSP
    
iWalls = mesh.specialSets["MinI_VertexSet"] + mesh.specialSets["MaxI_VertexSet"]
jWalls = mesh.specialSets["MinJ_VertexSet"] + mesh.specialSets["MaxJ_VertexSet"]
tWalls = mesh.specialSets["MaxJ_VertexSet"]
bWalls =mesh.specialSets["MinJ_VertexSet"]

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



# In[ ]:




# In[38]:

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

# In[39]:

###########
#Material Swarm and variables
###########

#create material swarm
gSwarm = uw.swarm.Swarm(mesh=mesh, particleEscape=True)

#create swarm variables
yieldingCheck = gSwarm.add_variable( dataType="int", count=1 )
materialVariable = gSwarm.add_variable( dataType="int", count=1 )
ageVariable = gSwarm.add_variable( dataType="double", count=1 )


#these lists  are part of the checkpointing implementation
varlist = [materialVariable, yieldingCheck, ageVariable]
varnames = ['materialVariable', 'yieldingCheck', 'ageVariable']


# In[40]:

mantleIndex = 0
crustIndex = 1
harzIndex = 2
airIndex = 3



if checkpointLoad:
    temperatureField.load(os.path.join(checkpointLoadDir, "temperatureField" + ".hdf5"))
    pressureField.load(os.path.join(checkpointLoadDir, "pressureField" + ".hdf5"))
    velocityField.load(os.path.join(checkpointLoadDir, "velocityField" + ".hdf5"))
    gSwarm.load(os.path.join(checkpointLoadDir, "swarm" + ".h5"))
    for ix in range(len(varlist)):
        varb = varlist[ix]
        varb.load(os.path.join(checkpointLoadDir,varnames[ix] + ".h5"))

else:

    # Layouts are used to populate the swarm across the whole domain
    layout = uw.swarm.layouts.PerCellRandomLayout(swarm=gSwarm, particlesPerCell=ppc)
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


# ###########
# #This block sets up a checkboard layout of passive tracers
# ###########
# 
# square_size = 0.1
# xlist = np.arange(mesh.minCoord[0] + square_size/2., mesh.maxCoord[0] + square_size/2., square_size)
# xlist = zip(xlist[:], xlist[1:])[::2]
# ylist = np.arange(mesh.minCoord[1] + square_size/2., mesh.maxCoord[1] + square_size/2., square_size)
# ylist = zip(ylist[:], ylist[1:])[::2]
# xops = []
# for vals in xlist:
#     xops.append( (operator.and_(   operator.gt(coordinate[0],vals[0]),   operator.lt(coordinate[0],vals[1])  ),0.) )
# xops.append((True,1.))
# 
# testfunc = fn.branching.conditional(xops) 
# 
# yops = []
# for vals in ylist:
#     yops.append( (operator.and_(   operator.gt(coordinate[1],vals[0]),   operator.lt(coordinate[1],vals[1])  ),0.) )
# yops.append((True,testfunc))
# 
# testfunc2 = fn.branching.conditional(yops) 
# tracerVariable.data[:] = testfunc.evaluate(gSwarm)
# tracerVariable.data[:] = testfunc2.evaluate(gSwarm)

# In[ ]:




# In[41]:

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


# In[42]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,ageVariable))
#fig.append( glucifer.objects.Points(gSwarm, viscosityMapFn, logScale=True, valueRange =[1e-3,1e5]))


#fig.show()


# In[43]:

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
DG.add_transition((mantleIndex,crustIndex), xFn, operator.lt, ndp.subzone + 4.*ndp.MANTLETOCRUST) #No crust on the upper plate
DG.add_transition((mantleIndex,crustIndex), ageVariable, operator.gt, crustageCond)


DG.add_transition((harzIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
DG.add_transition((harzIndex,crustIndex), xFn, operator.lt, ndp.subzone + 4.*ndp.MANTLETOCRUST) #This one sets no crust on the upper plate
DG.add_transition((harzIndex,crustIndex), ageVariable, operator.gt, crustageCond)

#... to Harzbugite
DG.add_transition((mantleIndex,harzIndex), depthFn, operator.lt, ndp.HARZBURGDEPTH)
DG.add_transition((mantleIndex,harzIndex), depthFn, operator.gt, ndp.MANTLETOCRUST)
DG.add_transition((mantleIndex,harzIndex), ageVariable, operator.gt, crustageCond) #Note we can mix functions and swarm variabls


#... to air
DG.add_transition((mantleIndex,airIndex), depthFn, operator.lt,0.)
DG.add_transition((crustIndex,airIndex), depthFn, operator.lt, 0. )
DG.add_transition((harzIndex,airIndex), depthFn, operator.lt, 0. )


# In[44]:

#CRUSTTOMANTLE, HARZBURGDEPTH, 0. + 7.*MANTLETOCRUST


# In[45]:

#gSwarm.particleCoordinates.data[particleID][1]


# In[46]:

##############
#For the slab_IC, we'll also add a crustal weak zone following the dipping perturbation
##############

if checkpointLoad != True:
    if not md.symmetricIcs:
        for particleID in range(gSwarm.particleCoordinates.data.shape[0]):
            if gSwarm.particleCoordinates.data[particleID][1] < 0.:
                materialVariable.data[particleID] = airIndex
                
            elif (
                Oc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                Tri.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                gSwarm.particleCoordinates.data[particleID][1] > (MAXY - ndp.maxDepth) and
                Cc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) == False
                
                ):
                materialVariable.data[particleID] = crustIndex
                
            elif (
                Oc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                Tri.evaluate(list(gSwarm.particleCoordinates.data[particleID])) and
                gSwarm.particleCoordinates.data[particleID][1] > (MAXY - ndp.maxDepth) and
                Hc.evaluate(list(gSwarm.particleCoordinates.data[particleID])) == False
                
                ):
                materialVariable.data[particleID] = harzIndex


# In[47]:

##############
#This is how we use the material graph object to test / apply material transformations
##############
DG.build_condition_list(materialVariable)

for i in range(2): #Need to go through a number of times
    materialVariable.data[:] = fn.branching.conditional(DG.condition_list).evaluate(gSwarm)


# In[48]:

#maxDepth


# In[49]:

#fig2= glucifer.Figure()
#fig2.append( glucifer.objects.Points(gSwarm,materialVariable))
#fig2.append( glucifer.objects.Surface(mesh, depthFn))

#fig2.show()
#fig2.save_database('test.gldb')


# ## Fault stuff
# 
# 

# In[50]:

import marker2D


# In[71]:

def update_swarm_from_faults(faults, proximityVariable, normalVectorVariable, signedDistanceVariable):
    """
    Compute fault attributes from the marker-line objects in the 'faults' list.
    Specifically:
    
      - proximityVariable carries information about which fault each swarm particle is close to (0 means none)
      - normalVectorVariable maps the orientation of the fault to nearby swarm particles
      - signedDistanceVariable carries the distance (positive means 'inside')  
      
      Unchecked error: if these variables are from different swarms 
      
    """
    
    if type(faults) == list:
    
        for fault_seg in faults:
            swarm = proximityVariable.swarm

            f, nz = fault_seg.compute_marker_proximity(swarm.particleCoordinates.data)    
            proximityVariable.data[nz] = f[nz]

            dv, nzv = fault_seg.compute_normals(swarm.particleCoordinates.data)
            normalVectorVariable.data[nzv] = dv[nzv]

            sd, dnz = fault_seg.compute_signed_distance(swarm.particleCoordinates.data)
            signedDistanceVariable.data[dnz] = sd[dnz]
    else:
        fault_seg = faults
        swarm = proximityVariable.swarm

        f, nz = fault_seg.compute_marker_proximity(swarm.particleCoordinates.data)    
        proximityVariable.data[nz] = f[nz]

        dv, nzv = fault_seg.compute_normals(swarm.particleCoordinates.data)
        normalVectorVariable.data[nzv] = dv[nzv]

        sd, dnz = fault_seg.compute_signed_distance(swarm.particleCoordinates.data)
        signedDistanceVariable.data[dnz] = sd[dnz]
    
    return


def update_swarm_from_line(fault, signedDistanceVariable):
    """
    Compute fault attributes from the marker-line objects in the 'faults' list.
    Specifically:
    
      - proximityVariable carries information about which fault each swarm particle is close to (0 means none)
      - normalVectorVariable maps the orientation of the fault to nearby swarm particles
      - signedDistanceVariable carries the distance (positive means 'inside')  
      
      Unchecked error: if these variables are from different swarms 
      
    """
        
    sd, dnz = fault.compute_signed_distance(gSwarm.particleCoordinates.data)
    signedDistanceVariable.data[dnz] = sd[dnz]
    
    return


def mask_materials(fault, material, materialVariable, proximityVariable, normalVectorVariable, signedDistanceVariable):

    """
    set the key fault-related swarm variabels to zero, based on material typr
    """
    
    fptsMask1 = (materialVariable.data[:,0] != material)
    fptsMask2 = (proximityVariable.data[:,0] == fault.ID )

    fptsMaskOut = fptsMask1 & fptsMask2

    normalVectorVariable.data[fptsMaskOut,:] = [0.0,0.0]
    proximityVariable.data[fptsMaskOut] = 0
    signedDistanceVariable.data[fptsMaskOut] = 0.0
    
    
def fault_strainrate_fns(fault_list, velocityField, faultNormalVariable, proximityproVariable):

    ## This is a quick / short cut way to find the resolved stress components.
    
    strainRateFn = fn.tensor.symmetric( velocityField.fn_gradient )

    
    _edotn_SFn = (        directorVector[0]**2 * strainRateFn[0]  + 
                    2.0 * directorVector[1]    * strainRateFn[2] * directorVector[0] + 
                          directorVector[1]**2 * strainRateFn[1]                          
                ) 

    # any non-zero proximity requires the computation of the above
    
    _edotn_SFn_Map    = { 0: 0.0 }
    for f in fault_list:
        _edotn_SFn_Map[f.ID] =  _edotn_SFn



   
    _edots_SFn = (  directorVector[0] *  directorVector[1] *(strainRateFn[1] - strainRateFn[0]) +
                    strainRateFn[2] * (directorVector[0]**2 - directorVector[1]**2)
                 )
    
  
    _edots_SFn_Map = { 0: 1.0e-15 }
    
    for f in fault_list:
        _edots_SFn_Map[f.ID] =  _edots_SFn
        
        
    #Finally, map the resolved strain rate to the proximity variable, 
    #which also makes resolved strain rate zero whereever proximity variable is zero 


    edotn_SFn =     fn.branching.map( fn_key = proximityVariable, 
                                      mapping = _edotn_SFn_Map)


    edots_SFn =     fn.branching.map( fn_key = proximityVariable, 
                                      mapping = _edots_SFn_Map )

    
    return edotn_SFn, edots_SFn


## Time update

def faults_advance_in_time(faults,proximityVariable, directorVector, signedDistanceVariable, materialVariable, mmask):
    for f in faults:
        f.advection(dt)


    update_swarm_from_faults(faults, proximityVariable, directorVector, signedDistanceVariable)
    mask_materials(materialV, materialVariable, proximityVariable, directorVector, signedDistanceVariable)


# In[122]:




# In[80]:

###########
#Initial Coordinates for inerfaces and faults
###########

#subduction fault
introPoint = ndp.subzone - abs(ndp.subzone - ndp.lRidge)/2. #half way between ridge and Sz
faultthickness = ndp.MANTLETOCRUST/2. #initiale fault at half-depth of crust
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



#interface that tracks the shape and position of the slab
midthickness = 20e3/dp.LS #initialize tracking swarm at ~ mid lithosphere depth

if dp.sense == 'Right':
    introPoint = ndp.lRidge + midthickness #
else:
    introPoint = ndp.rRidge - midthickness #
nfault = 200
slabCoords =np.zeros((nfault, 2))

reducedRocM = ndp.roc  - midthickness
xlimslab = reducedRocM*math.cos(math.pi*(90. - dp.theta)/180)
slabCoords[:, 0] = np.linspace(introPoint, ndp.subzone + xlimslab, nfault) #note SZ location is hardcoded here 
for index, xval in np.ndenumerate(slabCoords[:,0]):
    #print index, xval
    #swarmCoords[index[0], 1] = 1. - isodepthFn.evaluate((xval, 0.)) #This bit for the plate 
    if  xval < ndp.subzone:
        slabCoords[index[0], 1] = MAXY - midthickness #This bit for the plate 
        
    else:
        slabCoords[index[0], 1] = (MAXY - (midthickness) - (reducedRocM - ( math.sqrt((reducedRocM**2 - (xval-ndp.subzone)**2)))))
        
slabCoords = slabCoords[slabCoords[:,1] > (MAXY - ndp.maxDepth)] #kill any deeper than cutoff


# In[113]:

#Initiaze the swarms in a 
fault_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], faultthickness, 0.0, 0.0, crustIndex)
surface_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], ndp.StALS, 0.0, 0.0, airIndex)
slab_seg  = marker2D.markerLine2D(mesh, velocityField, [], [], 1e9/dp.LS, 0.0, 0.0, crustIndex)   #Note very large fault thickness 


#These lists are used to checkpoint the marker lines, similar to the swarm variables.        
interfaces = []
interfaces.append(fault_seg)
interfaces.append(surface_seg)
interfaces.append(slab_seg)
interfacenames = ['fault_seg', 'surface_seg', 'slab_seg']

#If restarting, load the swarms from file Interfaces are just swarms, so should be fine to rely on parallel h5 machinery here
if checkpointLoad:
    for ix in range(len(interfaces)):
        tempname = interfacenames[ix]
        interfaces[ix].swarm.load(os.path.join(checkpointLoadDir,  tempname + ".h5"))

#otherwise add the point from at the initial locations
else:
    fault_seg.add_points(faultCoords[:, 0], faultCoords[:, 1])
    surface_seg.add_points(surfaceCoords[:, 0], surfaceCoords[:, 1])
    slab_seg.add_points(slabCoords[:, 0], slabCoords[:, 1]) 
    


# In[84]:

#Add the necessary swarm variables

proximityVariable      = gSwarm.add_variable( dataType="int", count=1 )
signedDistanceVariable = gSwarm.add_variable( dataType="float", count=1 )
directorVector   = gSwarm.add_variable( dataType="double", count=2)

directorVector.data[:,:] = 0.0
proximityVariable.data[:] = 0
signedDistanceVariable.data[:] = 0.0



# Call the Fault helper functions to initialize this info on the main material swarm
    
update_swarm_from_faults(surface_seg, proximityVariable, directorVector, signedDistanceVariable)
mask_materials(surface_seg, airIndex, materialVariable, proximityVariable, directorVector, signedDistanceVariable)


update_swarm_from_faults(fault_seg, proximityVariable, directorVector, signedDistanceVariable)
mask_materials(fault_seg, crustIndex, materialVariable, proximityVariable, directorVector, signedDistanceVariable)

#Also switch off proximity beneath ndp.CRUSTVISCUTOFF depth
proximityVariable.data[gSwarm.particleCoordinates.data[:,1]  < (1. - ndp.CRUSTVISCUTOFF)] = 0. 

# These should be general enough not to need updating when the faults move etc..
#ie they should update as the fields/functions/swarm variables they are built on update
edotn_SFn, edots_SFn = fault_strainrate_fns(interfaces, velocityField, directorVector, proximityVariable)


# In[79]:

#Have to add these new swarm variable to our variable lists if we want them to get checkpointed

varlist.append(proximityVariable )
varlist.append(signedDistanceVariable)
varlist.append(directorVector)

varnames.append('proximityVariable')
varnames.append('signedDistanceVariable')
varnames.append('directorVector')


if checkpointLoad:                              #Reload all swarm variables 
    for ix in range(len(varlist)):
        varb = varlist[ix]
        varb.load(os.path.join(checkpointLoadDir,varnames[ix] + ".h5"))
        
 

 


# In[78]:

## Take a look at the locations of the materials

#Note we only use this mesh director for visualizing the directorVector (vector-on-swarm viewing not suppoted yet)
#meshDirector = uw.mesh.MeshVariable( mesh, 2 )
#projectDirector = uw.utils.MeshVariable_Projection( meshDirector, directorVector, type=1 )
#projectDirector.solve()    

#figMaterials = glucifer.Figure( figsize=(1200,400), boundingBox=((-2.0, -0.0, 0.0), (2.0, 1.0, 0.0)) )

#Plot swarm associated with each fault
#for f in faults:
#    figMaterials.append( glucifer.objects.Points(f.swarm, colours="Black Black", pointSize=2.0, colourBar=False) )


#plot mesh director viz. guy
#figMaterials.append( glucifer.objects.VectorArrows(mesh, meshDirector, scaling=.08, 
#                                               resolutionI=100, resolutionJ=20, opacity=0.25) )


#Proximity variable - this is the colour
#figMaterials.append( glucifer.objects.Points(gSwarm, proximityVariable, 
#                                             pointSize=5.0,  opacity=0.75) )


#signedDistanceVariable - this variable goes to zero where the proximity cutoff is
#note that it's signed...faults have a direction
#figMaterials.append( glucifer.objects.Points(gSwarm, signedDistanceVariable, 
#                                             pointSize=2.0))

#Add mesh
#figMaterials.append( glucifer.objects.Mesh(mesh, opacity=0.1) )

#figMaterials.show()
#figMaterials.save_database('test.gldb')


# Rheology
# -----
# 
# 

# In[55]:

##############
#Set up any functions required by the rheology
##############
strainRate_2ndInvariant = fn.tensor.second_invariant( 
                            fn.tensor.symmetric( 
                            velocityField.fn_gradient ))

def safe_visc(func, viscmin=ndp.eta_min, viscmax=ndp.eta_max):
    return fn.misc.max(viscmin, fn.misc.min(viscmax, func))


# In[56]:

#strainRate_2ndInvariant = fn.misc.constant(ndp.SR) #dummy fucntion to check which mechanisms are at active are reference strain rate


# In[57]:

#ndp.crust_cohesion_fac


# In[58]:

############
#Rheology: create UW2 functions for all viscous mechanisms
#############
omega = fn.misc.constant(1.) #this function can hold any arbitary viscosity modifications 


##Diffusion Creep
diffusion = fn.misc.min(ndp.eta_max, fn.math.exp(-1*ndp.Edf + ndp.Edf / (temperatureField + 1e-8)))


##Define the Plasticity
ys =  ndp.cm + (depthFn*ndp.fcmd)
ysMax = 10e4*1e6*sf.stress
ysf = fn.misc.min(ys, ysMax)
yielding = ysf/(2.*(strainRate_2ndInvariant))

##Crust rheology
#crustys =  ndp.cohesion*ndp.crust_cohesion_fac + (depthFn*ndp.fcd*ndp.crust_fc_fac)
crustys =  ndp.cc + (depthFn*ndp.fccd) #only weakened cohesion is discussed, not fc
crustvisc = crustys/(2.*(strainRate_2ndInvariant)) 


##Interface rheology
interfaceys =  ndp.ci + (depthFn*ndp.fcid) #only weakened cohesion is discussed, not fc
interfacevisc = interfaceys/(2.*(strainRate_2ndInvariant))


# In[59]:

############
#Rheology: combine viscous mechanisms in various ways 
#harmonic: harmonic average of all mechanims
#min: minimum effective viscosity of the mechanims
#mixed: takes the minimum of the harmonic and the plastic effective viscosity
#############

#linear rheology 
linearviscosityFn = safe_visc(diffusion)



interfaceCond = operator.and_((depthFn < ndp.CRUSTVISCUTOFF), (depthFn > ndp.MANTLETOCRUST))    


#combined rheology    
finalviscosityFn  = fn.branching.conditional([(depthFn < ndp.LOWMANTLEDEPTH, safe_visc(fn.misc.min(diffusion, yielding))),
                                  (True, safe_visc(safe_visc(diffusion*ndp.low_mantle_visc_fac)))])

#crust rheology    
#finalcrustviscosityFn = safe_visc(fn.misc.min(ndp.eta_max_crust, 
#                                              crustyielding)) #cohesion weakening factor also applies to eta_0

crustviscosityFn = safe_visc(fn.misc.min(linearviscosityFn, crustvisc), ndp.eta_max_crust)
interfaceviscosityFn = safe_visc(fn.misc.min(linearviscosityFn, interfacevisc), ndp.eta_max_interface)

if ndp.eta_max_crust == ndp.eta_min_crust: #If these are equal, set to constant visc. 
    crustviscosityFn = fn.misc.constant(ndp.eta_min_crust)
    
if ndp.eta_max_interface == ndp.eta_min_interface: #If these are equal, set to constant visc. 
    interfaceviscosityFn = fn.misc.constant(ndp.eta_min_interface)
    


finalcrustviscosityFn  = fn.branching.conditional([(depthFn < ndp.MANTLETOCRUST, crustviscosityFn),
                                                     (interfaceCond, interfaceviscosityFn), #
                                                     (True, finalviscosityFn)])


# Stokes system setup
# -----
# 

# In[60]:

buoyancyFn =  ndp.RA*temperatureField


# In[61]:

densityMapFn = fn.branching.map( fn_key = materialVariable,
                         mapping = {airIndex:ndp.StRA,
                                    crustIndex:buoyancyFn, 
                                    mantleIndex:buoyancyFn,
                                    harzIndex:buoyancyFn} )


# In[62]:


# Define our vertical unit vector using a python tuple (this will be automatically converted to a function).
gravity = ( 0.0, 1.0 )

# Now create a buoyancy force vector using the density and the vertical unit vector. 
buoyancyFn = densityMapFn * gravity


# In[63]:

stokesPIC = uw.systems.Stokes(velocityField=velocityField, 
                              pressureField=pressureField,
                              conditions=[freeslipBC,],
                              fn_viscosity=linearviscosityFn, 
                              fn_bodyforce=buoyancyFn )


# In[64]:

solver = uw.systems.Solver(stokesPIC)
if not checkpointLoad:
    solver.solve() #A solve on the linear visocisty is unhelpful unless we're starting from scratch


# In[ ]:




# In[65]:

viscosityMapFn1 = fn.branching.map( fn_key = materialVariable,
                         mapping = {crustIndex:finalcrustviscosityFn,
                                    mantleIndex:finalviscosityFn,
                                    harzIndex:finalviscosityFn,
                                    airIndex:ndp.Steta_n} )


delta_Steata2 = ndp.Steta_n - ndp.Steta_s
delta_eta_fault = 0.

if md.subductionFault:  
    if ndp.eta_min_fault == ndp.eta_min_fault:#Transverse rheology is isoviscous
        delta_eta_fault = fn.misc.min(0.999, fn.misc.max (0.,   ndp.eta_min_crust - ndp.eta_min_fault))       
        
    else:
        #Transverse viscosity is related the Mohr-Coulomb criterion
        delta_eta_fault = fn.misc.min(0.999, fn.misc.max (0., 
        viscosityMapFn1 - ((edotn_SFn*viscosityMapFn1 + ndp.fcfd * pressureField)  + ndp.cf)/edots_SFn))
   


# In[66]:

# This one maps to my fault-proximity variable (which also picks only materialV)
viscosityMapFn2    = { 0: 0.0, 
                           1: delta_eta_fault, 
                           3: delta_Steata2
                       }
    

viscosityMapFn2  = fn.branching.map( fn_key = proximityVariable, 
                                           mapping = viscosityMapFn2)


# In[67]:

#orientation = -1.*90. * math.pi / 180.0  #vertical
#math.cos(orientation), math.sin(orientation)


# In[68]:

#stickyAir


# In[69]:

#Add the non-linear viscosity to the Stokes system
stokesPIC.fn_viscosity = viscosityMapFn1

if md.stickyAir or md.subductionFault:
    stokesPIC.fn_viscosity2 = viscosityMapFn2
    stokesPIC._fn_director   = directorVector


# In[70]:

solver.set_inner_method("mumps")
solver.options.scr.ksp_type="cg"
solver.set_penalty(1.0e7)
solver.options.scr.ksp_rtol = 1.0e-4
solver.solve(nonLinearIterate=True)
solver.print_stats()


# In[71]:

#Check which particles are yielding
#yieldingCheck.data[:] = 0

#yieldconditions = [ ( finalviscosityFn < Visc , 1), 
#               ( True                                           , 0) ]

# use the branching conditional function to set each particle's index
#yieldingCheck.data[:] = fn.branching.conditional( yieldconditions ).evaluate(gSwarm)


# In[72]:

#velocityFieldIso       = uw.mesh.MeshVariable( mesh=mesh,         nodeDofCount=2 )
#velocityFieldIso.data[:] = velocityField.data.copy()



#strainRate_2ndInvariantIso = fn.tensor.second_invariant( 
#                            fn.tensor.symmetric( 
#                            velocityFieldIso.fn_gradient ))


# Advection-diffusion System setup
# -----

# In[75]:

advDiff = uw.systems.AdvectionDiffusion( phiField       = temperatureField, 
                                         phiDotField    = temperatureDotField, 
                                         velocityField  = velocityField,
                                         fn_sourceTerm    = 0.0,
                                         fn_diffusivity = 1.0, 
                                         #conditions     = [neumannTempBC, dirichTempBC] )
                                         conditions     = [ dirichTempBC] )

passiveadvector = uw.systems.SwarmAdvector( swarm         = gSwarm, 
                                     velocityField = velocityField, 
                                     order         = 1)


# In[76]:

population_control = uw.swarm.PopulationControl(gSwarm,deleteThreshold=0.2,splitThreshold=1.,maxDeletions=3,maxSplits=0, aggressive=True, particlesPerCell=ppc)


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

# In[ ]:




# In[54]:

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
tempMM = fn.view.min_max(temperatureField)
tempMM.evaluate(mesh)
TMAX = tempMM.max_global()
mantleconditions = [ (                                  temperatureField < 0.9*TMAX, 1.),
                   (                                                   True , 0.) ]
lithRestFn = fn.branching.conditional(mantleconditions)
lithRestFn*=rockRestFn #Add next level up in heirarchy


#Level 4. lower plate - upper plate:


#This whole section simply builds a restriction Fn that separates the upper and lower plate 
#It's pretty cumbersome, and will need to advected, rebuilt
#can YOU think of a better way?
   
lowerPlateRestFn = gSwarm.add_variable( dataType="double", count=1 )
lowerPlateRestFn.data[:] = 0.0

update_swarm_from_line(slab_seg, lowerPlateRestFn )
lowerPlateRestFn.data[np.where(lowerPlateRestFn.data != 0.)] += midthickness #assume instances of 0. are actually where no fault on proc
lowerPlateRestFn.data[np.where(lowerPlateRestFn.data > 0.*midthickness)] = 1.
lowerPlateRestFn.data[np.where(lowerPlateRestFn.data <= 0.*midthickness)] = 0. 
lowerPlateRestFn *= lithRestFn #Add next level up in heirarchy

#Also see if we can stick the the Velocity and coords on to this swarm as well
tipVar = uw.swarm.SwarmVariable(slab_seg.swarm, dataType='double', count=4)
tipVar.data[:,:2] = velocityField.evaluate(slab_seg.swarm)
tipVar.data[:,2:] = xFn.evaluate(slab_seg.swarm)
tipVar.data[:,3:] = yFn.evaluate(slab_seg.swarm)


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


# In[78]:

respltconditions = [ 
                    (                                  hinge60RestFn*2. > rockRestFn*1., 1.),
                    (                                  lowerPlateRestFn*3. > hinge60RestFn*2. , 3.),
                    (                                  lithRestFn*5. > lowerPlateRestFn*3. , 4.),
                   (                                                   True , 0.) ]

respltFn = fn.branching.conditional(respltconditions )


# In[95]:

#fig= glucifer.Figure()
#fig.append( glucifer.objects.Points(gSwarm,respltFn))
#fig.append( glucifer.objects.Points(gSwarm,lithRestFn))
#fig.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
#fig.append( glucifer.objects.Points(gSwarm,hinge180RestFn))
#fig.append( glucifer.objects.Points(gSwarm,interfaceRestFn))
#fig.show()
#fig.save_database('test.gldb')


# In[80]:

###################
#Surface Restriction functions
###################

def platenessFn(val = 0.1):
    normgradV = fn.math.abs(velocityField.fn_gradient[0]/fn.math.sqrt(velocityField[0]*velocityField[0])) #[du*/dx]/sqrt(u*u)



    srconditions = [ (                                  normgradV < val, 1.),
                   (                                                   True , 0.) ]


    return fn.branching.conditional(srconditions)

srRestFn = platenessFn(val = 0.1)


# In[81]:

###################
#Setup any Functions to be integrated
###################

sqrtv2 = fn.math.sqrt(fn.math.dot(velocityField,velocityField))
vx = velocityField[0]
v2x = fn.math.dot(velocityField[0],velocityField[0])
sqrtv2x = fn.math.sqrt(fn.math.dot(velocityField[0],velocityField[0]))
dw = temperatureField*velocityField[1]
sinner = fn.math.dot( strainRate_2ndInvariant, strainRate_2ndInvariant )
vd = 2.*viscosityMapFn1*sinner
dTdZ = temperatureField.fn_gradient[1]



# In[82]:

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
    
   
    


# In[83]:

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


# In[84]:

#Setup surface integrals

_surfLength = surfint()
_rmsSurf = surfint(v2x)
_nuTop = surfint(dTdZ)
_nuBottom = surfint(dTdZ, surfaceIndexSet=mesh.specialSets["MinJ_VertexSet"])
_plateness = surfint(srRestFn)


# In[85]:

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


# In[86]:

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

# In[89]:

#viscVariable = gSwarm.add_variable( dataType="float", count=1 )
#viscVariable.data[:] = viscosityMapFn1.evaluate(gSwarm)


# In[90]:

if figures == 'gldb':
    #Pack some stuff into a database as well
    figDb = glucifer.Figure()
    #figDb.append( glucifer.objects.Mesh(mesh))
    figDb.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.0005))
    #figDb.append( glucifer.objects.Points(gSwarm,tracerVariable, colours= 'white black'))
    figDb.append( glucifer.objects.Points(gSwarm,materialVariable))
    #figDb.append( glucifer.objects.Points(gSwarm,viscMinVariable))
    #figDb.append( glucifer.objects.Points(gSwarm,fnViscMin))
    figDb.append( glucifer.objects.Points(gSwarm, viscosityMapFn1, logScale=True))
    figDb.append( glucifer.objects.Points(gSwarm, strainRate_2ndInvariant, logScale=True))
    figDb.append( glucifer.objects.Points(gSwarm,temperatureField))
    
    
    figRestrict= glucifer.Figure()
    #figRestrict.append( glucifer.objects.Points(gSwarm,respltFn))
    figRestrict.append( glucifer.objects.Points(gSwarm,lithRestFn))
    figRestrict.append( glucifer.objects.Points(gSwarm,lowerPlateRestFn))
    figRestrict.append( glucifer.objects.Points(gSwarm,hinge180RestFn))
    figRestrict.append( glucifer.objects.Points(gSwarm,interfaceRestFn))
    figRestrict.append( glucifer.objects.Points(interfaces[0].swarm, colours="Blue Blue", pointSize=2.0, colourBar=False) )
    figRestrict.append( glucifer.objects.Points(interfaces[1].swarm, colours="Red Red", pointSize=2.0, colourBar=False) )
    figRestrict.append( glucifer.objects.Points(slab_seg.swarm, colours="Black Black", pointSize=2.0, colourBar=False) )

elif figures == 'store':
    fullpath = os.path.join(outputPath + "gldbs/")
    store = glucifer.Store(fullpath + 'subduction.gldb')

    figTemp = glucifer.Figure(store,figsize=(300*np.round(md.aspectRatio,2),300))
    figTemp.append( glucifer.objects.Points(gSwarm,temperatureField))

    figVisc= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
    figVisc.append( glucifer.objects.Points(gSwarm,viscosityMapFn1, logScale=True, valueRange =[1e-3,1e5]))
    
    figSr= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
    figSr.append( glucifer.objects.Points(gSwarm,strainRate_2ndInvariant, logScale=True))
    figSr.append( glucifer.objects.VectorArrows(mesh,velocityField, scaling=0.0005))

    #figMech= glucifer.Figure(store, figsize=(300*np.round(md.aspectRatio,2),300))
    #figMech.append( glucifer.objects.Points(gSwarm,fnViscMin))


# **Miscellania**

# In[236]:

##############
#Create a numpy array at the surface to get surface information on (using parallel-friendly evaluate_global)
##############

surface_xs = np.linspace(mesh.minCoord[0], mesh.maxCoord[0], mesh.elementRes[0] + 1)
surface_nodes = np.array(zip(surface_xs, np.ones(len(surface_xs)*mesh.maxCoord[1]))) #For evaluation surface velocity
normgradV = velocityField.fn_gradient[0]/fn.math.sqrt(velocityField[0]*velocityField[0])

tempMM = fn.view.min_max(temperatureField)
dummy = tempMM.evaluate(mesh)



# In[ ]:




# In[114]:

##############
#These functions handle checkpointing
##############


#Subzone = ndp.subzone


def checkpoint1(step, checkpointPath,filename, filewrites):
    path = checkpointPath + str(step) 
    os.mkdir(path)
    ##Write and save the file, if not already a writing step
    if not step % filewrites == 0:
        f_o.write((28*'%-15s ' + '\n') % (areaintRock, tempintRock, rmsintRock, dwintRock, vdintRock,
                                  areaintLith, tempintLith,rmsintLith, dwintLith, vdintLith,
                                  areaintLower, tempintLower, rmsintLower, dwintLower, vdintLower, 
                                  areaintHinge180,vdintHinge180, areaintHinge60, vdintHinge60, 
                                  areaintInterface, vdintInterface, vdintInterface,
                                  rmsSurf, nuTop, nuBottom, plateness, ndp.subzone, realtime))
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
    
    


# In[ ]:




# In[1]:

##############
#Simple function to return info about location of plate boundaries
##############

def getnearpos(array,value):
    idx = (np.abs(array-value)).argmin()
    return idx 

def plate_info(srfilename, minx, maxx,  searchdx, oldszloc = 0.0):
    """
    Use the surface strain rate field to find the location of the subduction zone in 2d
    
    """
    if type(srfilename) == str: #read surface strain rate points from file
        sr = np.load(srfilename)
    else:
        sr =  srfilename        #read surface strain rates directly from array
    xs = np.linspace(minx,maxx,sr.shape[0] )
    #infs at the ends of the SR data...replace with adjacent values
    sr[0] = sr[1] 
    sr[-1] = sr[2]
    #Normalize
    srx = (sr- sr.mean()) /(sr.max() - sr.min())
    #reduce the search domain, to near the previous PB location
    lx, rx = getnearpos(xs, oldszloc - searchdx),  getnearpos(xs, oldszloc + searchdx)
    red_xs, red_sr = xs[lx:rx], srx[lx:rx]
    #return the minima
    newszLoc = red_xs[np.argmin(red_sr)]
    return newszLoc


# In[116]:

# initialise timer for computation
start = time.clock()


# In[117]:

#max_vx_surf(velocityField, mesh)


# Main simulation loop
# -----
# 

# In[190]:

#while step < 6:
while realtime < 0.002:

    # solve Stokes and advection systems
    solver.solve(nonLinearIterate=True)
    dt = advDiff.get_max_dt()
    if step == 0:
        dt = 0.
    advDiff.integrate(dt)
    passiveadvector.integrate(dt)
    for f in interfaces:
        f.advection(dt)
        

    
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
    # Calculate the Metrics
    ################
    if (step % metric_output == 0):
        ###############
        #Rebuild the restriction functions where necessary
        ###############
        lowerPlateRestFn = gSwarm.add_variable( dataType="double", count=1 )
        lowerPlateRestFn.data[:] = 0.0
        update_swarm_from_line(slab_seg, lowerPlateRestFn )
        lowerPlateRestFn.data[np.where(lowerPlateRestFn.data >= -1.*midthickness)] = 1.
        lowerPlateRestFn.data[np.where(lowerPlateRestFn.data < -1.*midthickness)] = 0. 
        lowerPlateRestFn *= lithRestFn #Add next level up in hierarchy
        
        #Hinge
        
        hinge60Spatialconditions = [ (           operator.and_( (depthFn < MAXY - (60e3/dp.LS)),  (xFn > ndp.subzone - 60e3/dp.LS)), 1.),
                   (                                                   True , 0.) ]
        hinge60RestFn = fn.branching.conditional(hinge60Spatialconditions)
        hinge60RestFn*=lowerPlateRestFn #Add next level up in hierarchy



        hinge180Spatialconditions = [ (           operator.and_( (depthFn < MAXY - (180e3/dp.LS)),  (xFn > ndp.subzone - 180e3/dp.LS)), 1.),
                           (                                                   True , 0.) ]
        hinge180RestFn = fn.branching.conditional(hinge180Spatialconditions)
        hinge180RestFn*=lowerPlateRestFn #Add next level up in hierarchy
        
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
        # output to summary text file
        if uw.rank()==0:
            f_o.write((28*'%-15s ' + '\n') % (areaintRock, tempintRock, rmsintRock, dwintRock, vdintRock,
                                  areaintLith, tempintLith,rmsintLith, dwintLith, vdintLith,
                                  areaintLower, tempintLower, rmsintLower, dwintLower, vdintLower, 
                                  areaintHinge180,vdintHinge180, areaintHinge60, vdintHinge60, 
                                  areaintInterface, vdintInterface, vdintInterface,
                                  rmsSurf, nuTop, nuBottom, plateness, ndp.subzone, realtime))
    ################
    #Also repopulate entire swarm periodically
    ################
    #if step % swarm_repop == 0:
    population_control.repopulate()   
    ################
    #Gldb output
    ################ 
    if (step % gldbs_output == 0): 
        if figures == 'gldb':
            #Remember to rebuild any necessary swarm variables
            fnamedb = "dbFig" + "_" + str(step) + ".gldb"
            fullpath = os.path.join(outputPath + "gldbs/" + fnamedb)
            figDb.save_database(fullpath)
            
            #Temp figure
            fnamedb = "restrictFig" + "_" + str(step) + ".gldb"
            fullpath = os.path.join(outputPath + "gldbs/" + fnamedb)
            figRestrict.save_database(fullpath)
        elif figures == 'store':      
            fullpath = os.path.join(outputPath + "gldbs/")
            store.step = step
            #Save figures to store
            figVisc.save( fullpath + "Visc" + str(step).zfill(4))
            #figMech.save( fullPath + "Mech" + str(step).zfill(4))
            figTemp.save( fullpath + "Temp"    + str(step).zfill(4))
            figSr.save( fullpath + "Str_rte"    + str(step).zfill(4))
    ################
    #Files output
    ################ 
    if (step % files_output == 0):

        vel_surface = velocityField.evaluate_global(surface_nodes)
        norm_surface_sr = normgradV.evaluate_global(surface_nodes)
        if uw.rank() == 0:
            fnametemp = "velsurface" + "_" + str(step)
            fullpath = os.path.join(outputPath + "files/" + fnametemp)
            np.save(fullpath, vel_surface)
            fnametemp = "norm_surface_sr" + "_" + str(step)
            fullpath = os.path.join(outputPath + "files/" + fnametemp)
            np.save(fullpath, norm_surface_sr)
            
        #Save the slab_seg and tipswarm coords 
        fnametemp1 = "midSwarm" + "_" + str(step)
        fullpath1 = os.path.join(outputPath + "files/" + fnametemp1)
        slab_seg.swarm.save(fullpath1)
        #tipVar.data[:,:2] = velocityField.evaluate(tipSwarm)
        #tipVar.data[:,2:] = xFn.evaluate(tipSwarm)
        #tipVar.data[:,3:] = yFn.evaluate(tipSwarm)
        #comm.barrier()
        #fnametemp2 = "tipSwarm" + "_" + str(step)
        #fullpath2 = os.path.join(outputPath + "files/" + fnametemp2)
        #tipVar.save('fullpath2')
            
    ################
    #Update the subduction zone / plate information
    ################ 
    
    comm.barrier()
    if (step % files_output == 0):
        
        if uw.rank() == 0:
            fnametemp = "norm_surface_sr" + "_" + str(step) + ".npy"
            fullpath = os.path.join(outputPath + "files/" + fnametemp)
            ndp.subzone = plate_info(fullpath, MINX, MAXX,  800e3/dp.LS, oldszloc = ndp.subzone)
            
        else:
            ndp.subzone = None
        
        comm.barrier()    
        #send out the updated info for sz location
        
        ndp.subzone = comm.bcast(ndp.subzone, root=0)

        #Has the polarity reversed?

        if dp.sense == 'right':
            tempop = operator.lt
            szoffet *= -1
        else:
            tempop = operator.gt
            
        #Update the relevant parts of the material graph
        #Remove and rebuild edges related to crust
        DG.remove_edges_from([(mantleIndex,crustIndex)])
        DG.add_edges_from([(mantleIndex,crustIndex)])
        DG.remove_edges_from([(harzIndex,crustIndex)])
        DG.add_edges_from([(harzIndex,crustIndex)])

        #... to crust
        DG.add_transition((mantleIndex,crustIndex), depthFn, operator.lt, 0.5)
        DG.add_transition((mantleIndex,crustIndex), xFn, tempop , ndp.subzone) #No crust on the upper plate
        DG.add_transition((mantleIndex,crustIndex), ageVariable, operator.gt, 0.2)

        DG.add_transition((harzIndex,crustIndex), depthFn, operator.lt, ndp.MANTLETOCRUST)
        DG.add_transition((harzIndex,crustIndex), xFn, tempop, ndp.subzone) #This one sets no crust on the upper plate
        DG.add_transition((harzIndex,crustIndex), ageVariable, operator.gt, crustageCond)
        
        comm.barrier()
                   
    
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
        
        #Apply any materialVariable changes
        for i in range(2): #go through twice
            materialVariable.data[:] = fn.branching.conditional(DG.condition_list).evaluate(gSwarm)
        
        #Also update any information related to faults / interfaces:
        update_swarm_from_faults(surface_seg, proximityVariable, directorVector, signedDistanceVariable)
        mask_materials(surface_seg, airIndex, materialVariable, proximityVariable, directorVector, signedDistanceVariable)
        
        update_swarm_from_faults(fault_seg, proximityVariable, directorVector, signedDistanceVariable)
        mask_materials(fault_seg, crustIndex, materialVariable, proximityVariable, directorVector, signedDistanceVariable)
        
        proximityVariable.data[gSwarm.particleCoordinates.data[:,1]  < (1. - ndp.CRUSTVISCUTOFF)] = 0.
        
        #And add extra particles to interfaces as necessary
        #subduction fault
        introPoint = ndp.subzone - abs(ndp.subzone - ndp.lRidge)/2. #half way between ridge and Sz
        fault_seg.add_points([introPoint],[MAXY - faultthickness])
        #Slab / mid swarm
        if dp.sense == 'Right':
            introPoint = ndp.lRidge + midthickness #
        else:
            introPoint = ndp.rRidge - midthickness #
        slab_seg.add_points([introPoint],[MAXY - midthickness])
        
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


# In[120]:




# In[ ]:




# In[98]:

#vxTi = velocityField[0].evaluate(surface_nodes)
#vxIso = velocityFieldIso[0].evaluate(surface_nodes)
#%pylab inline
#fig, axes = plt.subplots(figsize=(16,4))
#axes.plot(vxTi)
#axes.plot(vxIso)


#fig, axes = plt.subplots(figsize=(16,4))
#axes.plot(vxTi)
#axes.plot(surface_xs, vxTi - vxIso)
#fig.savefig('Ti_minus_Iso.png')
#plt.title('surface velocity residual - T. Iso minus Iso. weak zone')


# In[ ]:




# In[249]:




# In[102]:

#figRestrict.show()


# In[91]:

#figDb.show()


# In[92]:

figDb.save_database('test.gldb')


# In[ ]:

#slab_seg.swarm.save()


# In[ ]:



