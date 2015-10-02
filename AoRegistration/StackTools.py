# author: Tom Wright
# website: tomwright.ca

import numpy as np
import scipy
import scipy.signal
import cv2
import logging

import ImageTools

logger = logging.getLogger(__name__)

def find_frame_shear(im1,im2):
    '''{shear,correlvals} = check_frame_shear(im1,im2)
    Compares two images at the top and bottom to figure out the displacement occuring
    Original matlab code copyright 2012, Steve Burns, IU
    '''

    correlvals = []
    
    [numRows,numCols] = im1.shape[0],im2.shape[1]
    # want to find vertical locations 1/4 and 3/4
    topMidRow = int(numRows / 4)
    if topMidRow % 2 == 1:
        topMidRow = topMidRow - 1
    midCol = int(numCols / 2)
    if midCol % 2 == 1:
        midCol = midCol - 1

    # not going to use the very edges in this
    boarderRows = int(numRows / 6) - 2
    boarderCols = int(numCols / 3) - 2
    shears = []
    for indx in range(2):
        indx_multiplier = (indx * 2) + 1 #1 for top, 3 for bottom
        rowStart = (indx_multiplier * topMidRow) - boarderRows
        rowEnd = (indx_multiplier * topMidRow) + boarderRows
        colStart = midCol - boarderCols
        colEnd = midCol + boarderCols
        image_std = im1[rowStart:rowEnd,colStart:colEnd]
        image_cmp = im2[rowStart:rowEnd,colStart:colEnd]
        
        #I think fft convolve is good enough here as we are only interested in the amount of shift, not the value of the match
        results = find_frame_shift(image_std, image_cmp,method='fft',applyBlur=False,attemptSubPixelAlignment=False) 
        
        #this xcorr method takes twice as long
        
        #image_std = ImageTools.padd_image(image_std, [image_std.shape[1],
                                                      #image_std.shape[0],
                                                      #image_std.shape[1],
                                                      #image_std.shape[0]])        
        #results = calc_frame_shift(image_std, image_cmp,method='xcorr') 
        correlvals.append(results['maxcorr'])
        shears.append(results['coords'])
        
    shearval = ((shears[0][0] - shears[1][0])**2 + (shears[0][1] - shears[1][1])**2)**0.5
    return {'shearval':shearval,
            'correlvals':correlvals}


def find_frame_shift(im1,im2,topLeft=[(0,0),(0,0)],method='fft', applyBlur = False, attemptSubPixelAlignment = False):
    '''find_frame_shift(im1,im2) - Perform correlation between two images
    PARAMS:
    im1 - NxM ndarray
    im2 - NxM ndarray
    topLeft - Original coordinates of topLeft corner of images [(col,row),(col,row)] 
    [method] - 'fft' | 'xcorr'
    [applyFilter] - boolean
    RETURNS:
    {coords: (x,y), maxcorr}
    coords - tuple containing x,y displacement
    maxcorr - correlation at the maximal point
    
    Note:
    Expects im1 and im2 to be the same size if performing fft correlation
    method = fft - default, performs convolution, 
        'xcorr' uses a normalised x correlation - this won't work if the images are the same size
        
    applyFilter - default False, if true apply a mild blurring filter
    if top left is specified it need to be a two element list, each element is a tuple (col,row). 
        First element is top left coordinates for Im1, second topleft coordinates for Im2
    '''
    assert len(im1.shape)==2, "Image 1 is invalid size expects NxM"
    assert len(im2.shape)==2, "Image 2 is invalid size expects NxM"
    assert len(topLeft) == 2, 'If topLeft coords are provided needs to have len 2 [im1 coords, im2 coords]'
    assert len(topLeft[0]) == 2 and len(topLeft[1]) == 2, 'Top left coordinates should be (col,row)'
    assert method in ['xcorr','fft']
    
    if applyBlur:
        im1 = cv2.GaussianBlur(im1,(3,3),0)
        im2 = cv2.GaussianBlur(im2,(3,3),0)

    if method == 'fft':
        assert im1.shape==im2.shape,'im1 and im2 must be the same shape'
    
        #need to subtract mean brightness from each image
        im1 = im1 - im1.mean()
        im2 = im2 - im2.mean()    
        
        # going to use fftconvolve as it's faster than correlate2d
        # first need to rotate im2 180 degrees
        im2 = np.flipud(np.fliplr(im2))
        x_im = scipy.signal.fftconvolve(im1, im2, mode='same') 
        indices = np.unravel_index(x_im.argmax(), x_im.shape)
    elif method == 'xcorr':
        assert im1.shape[0] >= im2.shape[0], 'Image 2 must be smaller than image 1'
        assert im1.shape[1] >= im2.shape[1], 'Image 2 must be smaller than image 1'
        
        x_im = cv2.matchTemplate(np.float32(im1), np.float32(im2), cv2.cv.CV_TM_CCORR_NORMED)
        indices = np.unravel_index(x_im.argmax(), x_im.shape)
        
    if attemptSubPixelAlignment:
        indices = quadrant_detect(x_im)
        
        
    correlval = x_im.max()
    
    if method=='fft':
        shift_x = indices[1] - np.floor(x_im.shape[1]/2)
        shift_y = indices[0] - int(x_im.shape[0]/2)        
    elif method=='xcorr':
        shift_x = indices[1] - (topLeft[1][1] - topLeft[0][1])
        shift_y = indices[0] - (topLeft[1][0] - topLeft[0][0])
        
    return {'coords':(shift_x,shift_y),
            'maxcorr':correlval}

def quadrant_detect(corrMap):
    '''Performs subpixel alignment using the xcorr map from matchTemplate
    Only uses the 9x9 pixel region in the center of the correlation map
    Returns (rowcent,colcent)
    
    ## I'm loosing two pixels in this function!!!
    EXAMPLE:
    np.random.seed(1)
    x=np.random.rand(9,9)
    row,col = quadrant_detect(x)
    
    '''
    width = 11
    offset = (width - 1) / 2
    center = offset + 1
    
    [row,col] = np.unravel_index(corrMap.argmax(), corrMap.shape)
    try:
        #extract the region we are interested in
        corrMapPeak = corrMap[row - offset:row + offset + 1,col - offset:col + offset + 1]
        #seems the previous code doesn't raise an error, lets create my own...
        if not corrMapPeak.any():
            raise IndexError
        #make some cumulative sums across rows and columns
        cumulativeRow = corrMapPeak.sum(axis=1).cumsum() / corrMapPeak.sum()
        cumulativeCol = corrMapPeak.sum(axis=0).cumsum() / corrMapPeak.sum()
   
        peakRow = _get_midpoint(cumulativeRow)
        peakCol = _get_midpoint(cumulativeCol)
        
        rowcent = row + (peakRow - center) + 1  #+1 to deal with zero indexing
        colcent = col + (peakCol - center) + 1
        
    except IndexError:
        #if we are out of bounds return basic values without interpolation
        rowcent = row
        colcent = col
    return (rowcent,colcent)

def _get_midpoint(norm_sum):
    """Finds the midpoint value from a (normalised) cumulative sum array by making a linear fit and interpolating the 0.5 point
    """
    assert len(norm_sum) >= 6,'Too few points supplied for accurate interpolation'
    
    idxMidPoint = np.argmax(norm_sum > 0.5)
    idxMidPoint = max(idxMidPoint,2) #incase the midpoint is skewed to the start
    x = np.arange(idxMidPoint-2, idxMidPoint+2)
    y = norm_sum[x]
    
    p=np.polyfit(x,y,1)
    midpoint = (0.5 - p[1]) / p[0]
    midpoint = midpoint + 1 #deal with 0 indexing
    
    return midpoint
    
    
    
def apply_displacements(framestack,displacements):
    """Apply displacements to an image stack
    Resizes the framestack to the largest displacement
    """

    nframes, nrows, ncols = framestack.shape
    
    x_shifts = [val[0] for val in displacements]
    y_shifts = [val[1] for val in displacements]
    
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
    
    for idx in range(framestack.shape[0]):
        if displacements[idx][0] < 0:
            leftIndex = left - abs(displacements[idx][0])
        else:
            leftIndex = left + displacements[idx][0]
            
        if displacements[idx][1] < 0:
            topIndex = top - abs(displacements[idx][1])
        else:
            topIndex = top + displacements[idx][1]
        
        output[idx,topIndex:topIndex + nrows, leftIndex : leftIndex + ncols] = framestack[idx,:,:]
        
        #find the points where all images overlap
    map = output.min(0)
    [r,c] = np.where(map>-1)
    output = output[:,min(r):max(r)+1,min(c):max(c)+1]  #crop the output to just the overlapping region
        
    return output


    
    
    