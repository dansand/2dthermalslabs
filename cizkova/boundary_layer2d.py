#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""

Function for generating the thermal profile of (oceanic) tectonic plates for Underworld2

module_name, package_name, ClassName, function_name, method_name, ExceptionName, propertyName GLOBAL_CONSTANT_NAME, globalVarName, instanceVarName, functionParameterName, localVarName

:license: GNU General Public License, Version 3
    (http://www.gnu.org/copyleft/gpl.html)
"""

import slippy2 as sp
from slippy2 import unit_conversions
import math
class LithosphereTemps(object):

    def __init__(self, mesh, temperatureField,lengthScale, SZ, MOR=None,tint = 0.8, tsurf = 0.0, vel= 100e3, diffs = 1e-6):
        self.mesh = mesh
        self.dim = mesh.dim
        self.maxCoord = mesh.maxCoord
        self.minCoord = mesh.minCoord
        self.meanTemp = temperatureField.data.mean()
        self.temperatures = temperatureField.data
        self.periodic = mesh.periodic[0]
        self.lengthScale = lengthScale
        self.SZ = SZ
        self.MOR = MOR
        self.vel = vel
        self.diffs = diffs
        self.tint = tint
        self.tsurf = tsurf

    def agefunc(self, x):
        """
        Create a (linear) age function based on ridges being the sides of the model,
        and subduction zone location at the dimesnionaless (model) position "szloc"
        Args:
            x (float): dimenisionless value to get age (horizontal coordinate)
            SZ (float): dimensionless location of trench / subduction zone
            MOR(float): dimensionless location of ridge (for periodic meshes)
            vel(float): in m/my
        Returns:
            the age in millions of years
        Raises:
            TypeError: not implemented
            ValueError: not implemented
        """
        #Translate the domain so it begins at x=0:
        if self.minCoord[0] < 0:
            xt = x + abs(self.minCoord[0])
            SZt = self.SZ + abs(self.minCoord[0])
            xmax = abs(self.maxCoord[0] + abs(self.minCoord[0]))
            if self.MOR:
                MORt = self.MOR + abs(self.minCoord[0])
        else:
            xt = x - abs(self.minCoord[0])
            SZt = SZ +- abs(self.minCoord[0])
            xmax = abs(self.maxCoord[0] - abs(self.minCoord[0]))
            if self.MOR:
                MORt = self.MOR - abs(self.minCoord[0])
        if not self.periodic:
            #print('1')
            if xt >= SZt:
                dx = (xmax - xt)
            else:
                dx = xt
        elif not self.MOR:
            #print('2')
            if xt >= SZt:
                dx = (xmax - xt)
            else:
                dx = xt
        else:
            #print('3')
            if SZt > MORt:
                #print('1')
                if xt >= SZt:
                    dx = ((xmax - xt) + (MORt))
                elif xt >= MORt:
                    dx = (xt - MORt)
                else:
                    dx = MORt - xt
            else:
                #print('2')
                if xt >= MORt:
                    dx = ((xt - MORt))
                elif xt > SZt:
                    dx = MORt - xt
                else:
                    dx = (xt + (xmax - MORt))

        age = abs(dx* self.lengthScale )/self.vel
        return age

    def tempfunc(self, age, depth):
        """
        return dimensionless halfspace cooling model fucntion,
        Args:
            age (float): millions or years
            depth (float): metres
            t0 (float): the initial temperature of the halfspace
        Returns:
            dimensionless temp (0. - 1.)
        Raises:
            TypeError: not implemented
            ValueError: not implemented
        """
        #1000 in line below to convert back to km
        secs = unit_conversions.myts(age)
        temp = (self.tint - self.tsurf)*math.erf((depth)/(2*math.sqrt(secs*self.diffs))) + self.tsurf
        return temp

    def lithdepthfunc(self, age):
        """
        returns depth or thermal lithosphere in kilometers
        Args:
            age (float): millions or years
            kappa (float): diffusivity, m**2/s
        Returns:
            depth meters
        Raises:
            TypeError: not implemented
            ValueError: not implemented
        """
        depth_metres = 2.32*math.sqrt(unit_conversions.myts(age)*self.diffs)
        return depth_metres
