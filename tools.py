import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.stats import mode
from PIL import Image
from scipy.optimize import curve_fit
from dataclasses import dataclass


def local_max(arr, N = 2, strict = False):
    '''find local maximums of an array where local is defined as M points on either side, if strict is true
    then it will follow this process exactly if strict is false it will also count local maxes that are at least
    one space from the edge if they satisfy the requirement within the remaining array'''
    local_maxs = []
    M = int(N/2)
    
    #loop through the array
    if not strict:
        i = 1

    else:
        i = M

    indexes = []
    
    while i < len(arr) - 1:
        
        iterate = 1
        #flag
        local_max = True
        
        for j in range(M):
            try:
                #will make index error when your with M of the edges so except index error
                if arr[i] < arr[i + j]:
                    local_max = False
                    iterate = j
                    break

            except IndexError:
                if strict:
                    #reproduce old behaviour
                    local_max = False
                    break
                #other wise search in the other direction
                
            try:
                if arr[i] < arr[i - j]:
                    local_max = False
                    break

            except IndexError:
                if strict:
                    local_max = False
                    break

            
        if local_max:
            local_maxs.append(arr[i])
            indexes.append(i)
            
        i += iterate
        
    return np.array(local_maxs), np.array(indexes)


def gaussian(x, mu, sigma):
    return 1/(sigma*np.sqrt(2*np.pi))*np.exp(-1/2*((x-mu)/sigma)**2)


def error_prop(f,args,errors,ind_var = [None],**kwargs):
    errors = np.array(errors)
    
    #array for storing derivatives
    d_arr = np.ones(len(args))
    prop_arr = []
    #
    
        
    for x in ind_var:
        for i,arg in enumerate(args):
            #these are the smallest ratios that still see improvement in estimate
            lower_ratio = 0.99999999999
            upper_ratio = 1.00000000001
            #take a linspace of area surrounding point abs accounts for negative
            if arg != 0:
                arg_space = np.linspace(lower_ratio*abs(arg),upper_ratio*abs(arg),2)
                #reintroduce negative if it was removed
                arg_space *= abs(arg)/arg
            else:
                arg_space = np.linspace(lower_ratio - 1, upper_ratio - 1,2)

            #call the function with x inserted in correct position
            if x is None:
                y = f(*args[:i], arg_space, *args[i+1:],**kwargs)
            else:
                y = f(x,*args[:i], arg_space, *args[i+1:],**kwargs)

            #get partial derivative of the function with respect to arg
            d_arr[i] = np.gradient(y,arg_space)[0]
#             print(f"arg_space = {arg_space}\nderivative = {np.gradient(y,arg_space)[0]}")
        
        prop_err = np.sqrt(np.sum(d_arr**2*errors))
        prop_arr.append(prop_err)
    
    return prop_arr



def line_intensity(filename, rows = [], cols = [], conversion = None, norm = False):
    '''takes filename and the rows and cols of the image you wish to inspect and returns the line intensity
    averaged across the columns, conversion is the mm/px and setting a value will return a corresponding x array 
    with the line profile in form x, line'''
    #get image and change to np.array
    img = np.array(Image.open(filename))

    #handle case where no rows or cols are specified
    if len(cols) == 0:
        cols = [0, len(img[0])]
    
    if len(rows) == 0:
        rows = [0, len(img)]

    

    
    selected_rectangle = img[rows[0]:rows[1],cols[0]:cols[1]]
    line_intensity = np.mean(selected_rectangle, axis = 0)

    if norm:
        line_intensity /= np.max(line_intensity)

    if conversion is not None:
        x  = np.arange(*cols, dtype = float)
        x += conversion*rows[0]
        x *= conversion
        return x, line_intensity

    else:
        return line_intensity

@dataclass
class Extracted_Data:
    '''stores raw, the rows averages, the x in mm, the profile - the minimum and the norm_profile'''
    raw: np.array = np.array([])
    rows: int = 0
    x: np.array = np.array([])
    profile: np.array = np.array([])
    norm_profile: np.array = np.array([])
    err: np.array = np.array([])

def extract_data(filename, conversion, rows = [], cols = [],sigmaRaw = 0.0315, norm = 0):
    '''extracts data and returns the data in Extracted_Data class'''

    #load in greyscale image
    raw = np.array(Image.open(filename))
    
    #handle case where no rows or cols are specified
    if len(cols) == 0:
        cols = [0, len(raw[0])]
    
    if len(rows) == 0:
        rows = [0, len(raw)]
    
    #select rectangle
    selected_rectangle = raw[rows[0]:rows[1],cols[0]:cols[1]]

    #get profile of rectangle
    profile = np.mean(selected_rectangle, axis = 0)
    profileMin = np.min(profile)
    rowsAveraged = rows[1] - rows[0]
    
    _, index = local_max(profile, N = len(profile))

    #get the position from center of sensor
    x  = np.arange(*cols, dtype = float)
    x += conversion*rows[0] - np.median(x[index])
    x *= conversion

    # Calculate error in profile (ydatea)
    GreyValMax = 255
    SensorWellMax = 15585 # according to http://softwareservices.flir.com/BFS-PGE-16S2/latest/EMVA/EMVA.html
    alpha_1 = np.sqrt( profile*GreyValMax/(SensorWellMax*rowsAveraged) ) # poisson statistics of pixels capturing N photons
    alpha_0 = np.ones(len(x))*sigmaRaw                             # baseline dark + ambient uncertainty
    y_error = alpha_0 + alpha_1

    # subtract minimum baseline light from profile
    profile -= profileMin

    if norm == 'abs': # in 8 bit
        norm_profile = profile/GreyValMax
    else: # on axis normalization
        norm_profile = profile/np.max(profile)


    

    return Extracted_Data(raw = raw, profile = profile, norm_profile = norm_profile, x = x, rows = rowsAveraged, err = y_error)

