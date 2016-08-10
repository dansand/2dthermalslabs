#!/usr/bin/env python
# -*- coding: utf-8 -*

def mpy_to_kmpma(mpy):
    """
    convert meters per year to kilometers per year
    Args:
        mpy (float): velocity in meters per year
    Returns:
        velocity in kilometers per year
    Raises:
        TypeError: not implemented
        ValueError: not implemented
    """
    return 1e3*mpy

def myts(mys):
    """
    convert millions of years to seconds
    Args:
        mys (float): millions or years
    Returns:
        seconds
    Raises:
        TypeError: not implemented
        ValueError: not implemented
    """
    secs = mys*(3600.*24*365)*1e6
    return secs

def stmy(secs):
    """
    convert millions of years to seconds
    Args:
        mys (float): millions or years
    Returns:
        seconds
    Raises:
        TypeError: not implemented
        ValueError: not implemented
    """
    myr = secs/((3600.*24*365)*1e6)
    return myr
