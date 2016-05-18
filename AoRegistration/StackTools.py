# author: Tom Wright
# website: tomwright.ca

import numpy as np
import scipy
import scipy.signal
import cv2
import logging
import multiprocessing
import FrameStack
import ImageTools

logger = logging.getLogger(__name__)  
    
def apply_displacements(framestack,displacements):
    """Apply displacements to an image stack
    Resizes the framestack to the largest displacement
    displacements is expected to be a list of dicts
    [{'frameid':id in original input},
     {'shift':(x,y)}]
     framestack should be a framestack object
    """
    assert isinstance(framestack,FrameStack.FrameStack), 'FrameStack object expected'
    
    #check all displacements are in the framestack
    displacement_ids = [displacement['frameid'] for displacement in displacements]
    
    nrows, ncols = framestack.frameHeight, framestack.frameWidth
    nframes = len(displacements)
    if not all(np.in1d(displacement_ids, framestack.frameIds)):
        raise ValueError('Invalid displacements supplied')
    
    #calculate the minimum and maximum displacements
    x_shifts = [displacement['shift'][0] for displacement in displacements]
    y_shifts = [displacement['shift'][1] for displacement in displacements]
    left=0
    top=0
    right=0
    bottom=0
    if min(x_shifts) < 0:
        left = abs(min(x_shifts))
    if min(y_shifts) < 0:
        top = abs(min(y_shifts))
    if max(x_shifts) > 0:
        right = max(x_shifts)
    if max(y_shifts) > 0:
        bottom = max(y_shifts)
        
    output = np.ones((nframes,
                      nrows + top + bottom+1,
                      ncols + left + right+1),
                      dtype = np.float) * -1
    outputIdx = -1
    for displacement in displacements:
        outputIdx = outputIdx + 1
        currentFrameId = displacement['frameid']
        xShift = displacement['shift'][0]
        yShift = displacement['shift'][1]
        if xShift < 0:
            leftIndex = left - abs(xShift)
        else:
            leftIndex = left + xShift
            
        if yShift < 0:
            topIndex = top - abs(yShift)
        else:
            topIndex = top + yShift
        
        output[outputIdx,
               topIndex:topIndex + nrows,
               leftIndex : leftIndex + ncols] = framestack.get_frame_by_id(currentFrameId)       
        #find the points where all images overlap
    map = output.min(0)
    [r,c] = np.where(map>-1)
    output = output[:,min(r):max(r)+1,min(c):max(c)+1]  #crop the output to just the overlapping region
    output = FrameStack.FrameStack(output,frameIds=displacement_ids)
    output.templateFrameId = framestack.templateFrameId
    return output

def compute_lucky_image(stack):
    '''Use the 'lucky' algorithm to compute an average frame from an aligned stack

    Original Code Gang Huang, Indiana University, 2010
       revised to use convolution by Steve Burns, 2012
       Indiana University
    this code may be freely distributed and used in publications with attribution to the original paper
    reference Huang, G., Zhong, Z. Y., Zou, W. Y., and Burns, S. A., Lucky averaging: quality improvement of adaptive optics scanning laser ophthalmoscope images, Optics Letters 36, 3786-3788 (2011).
    '''
    nFrames, height, width = stack.shape
    
    numFinal = 15 
    covStack = np.empty_like(stack) #stack to hold covariance matrices
    imageStackSorted = np.empty((numFinal,height,width))
    
    for iFrame in range(nFrames):
        covStack[iFrame,:,:] = ImageTools.comatrix(stack[iFrame,:,:])
        
    covStackSortIndex = covStack.argsort(0)
    covStackSortIndex[:] = covStackSortIndex[::-1] #reverse the sort
    
    #now resort the image stack so that each pixel in the stack is sorted with the highest contrast pixels in the first frame
    for iRow in range(height):
        for iCol in range(width):
            imageStackSorted[:,iRow,iCol] = stack[covStackSortIndex[0:numFinal,iRow,iCol],iRow,iCol]
            
    finalImage = imageStackSorted.mean(axis=0)
    return finalImage

def getMaskedStack(stack):
    '''returns the stack as an np.maskedarray
    pixels with values of -1 in any frame are masked
    '''
    
    mask = stack.min(axis=0) < 0
    mask = np.tile(mask,[stack.shape[0],1,1])
    return np.ma.masked_array(stack,mask)
def interlaceStack(stack):
    """Attempt to fix poor interlacing in a frame stack
    """
    logger.info('Fixing interlace')
    nFrames, nRows, nCols = stack.shape
    shifts = [ImageTools.getInterlaceShift(stack[iFrame,:,:]) for iFrame in range(nFrames)]
    
    # find the modal shift value
    shift = int(max(set(shifts), key=shifts.count))
    logger.debug('Shifting interlace by %s pixels',shift)
    #allocate a new imageStack
    newStack = np.zeros((nFrames,nRows,nCols+abs(shift)))
    even_rows = np.arange(0,nRows,2)
    odd_rows = np.arange(1,nRows,2)
    
    if shift < 0:
        newStack[:,even_rows,abs(shift):newStack.shape[2]] = stack.data[:,even_rows,:]
        newStack[:,odd_rows,0:nCols] = stack.data[:,odd_rows,:]
    elif shift > 0:
        newStack[:,even_rows,0:nCols] = stack.data[:,even_rows,:]
        newStack[:,odd_rows,abs(shift):newStack.shape[2]] = stack.data[:,odd_rows,:]
    else:
        newStack=stack.data
        
    stack.data=newStack
    