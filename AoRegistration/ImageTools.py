# author: Tom Wright
# website: tomwright.ca

# ClickCropper based on code by:
# author:    Adrian Rosebrock
# website:   http://www.pyimagesearch.com


import cv2
import numpy as np
import scipy.signal

class ClickCropper:
    '''ClickCropper
    Object to support function click_and_crop()'''
    def __init__(self, image):
        self.image = image
        self.image_rectangle = image.copy()
        self.positions = []
        self.state = None

    def mouse_callback(self, event, x, y, flags, param):
        # Saving the click coordinates
        if event == cv2.EVENT_LBUTTONUP:
            if not self.state:
                self.state = "clicked"
                self.positions.append((x, y))
            elif self.state == "clicked":
                self.positions.append((x, y))
                self.state = "double_clicked"

        # Showing the selected area in real time
        # with green rectangle
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.state == "clicked":
                self.image_rectangle = self.image.copy()
                x1, y1 = self.positions[0]
                cv2.rectangle(self.image_rectangle, (x1, y1), (x, y), (0, 255, 0), 2)

        # Canceling the selection if right clicked
        elif event == cv2.EVENT_RBUTTONUP:
            if self.state == "double_clicked":
                self.image_rectangle = self.image.copy()
                self.positions = []
                self.state = None

    def crop(self):
        coords = self.roi()
        x1, y1 = coords[0]
        x2, y2 = coords[1]

        return self.image[y1:y2, x1:x2]
    
    def mask(self):
        coords = self.roi()
        x1, y1 = coords[0]
        x2, y2 = coords[1]        
        
        mask = np.zeros(self.image.shape)
        mask[y1:y2, x1:x2] = 1
        return mask
    
    def roi(self):
        '''Returns roi coordinates'''
        if len(self.positions) != 2:
            return None
        
        x1, y1 = self.positions[0]
        x2, y2 = self.positions[1]
    
        # Swapping coordinates if selected from
        # right-to-left or bottom-to-top
        if x1 > x2:
            x1, x2 = x2, x1
    
        if y1 > y2:
            y1, y2 = y2, y1        
            
        return [(x1,y1), (x2,y2)]

def click_and_crop(image, types='roi'):
    '''Displays image allowing user to select a rectangle ROI
    returns the cropped image
    USAGE:
    
    mask=tools.click_and_crop(np.ones((500,500)),types=['mask'])
    cv2.imshow(mask['mask'])
    
    PARAMS:
    image - an ndarray
    type ['roi' | 'mask' | 'crop'] - what to return, see details
    
    DETAILS:
    Left click to start selecting the region, left click again to finish
    Right click to reset
    'c' key to return
    'q' key to quit
    type param can take values:
    roi - returns the coordinates [(x1,y1),(x2,y2)]
    mask - returns a ndarray of same size as image with 1's in the selected region
    crop - returns the cropped image
    '''
    cropper = ClickCropper(image)
    cv2.namedWindow("Cropper")
    cv2.setMouseCallback("Cropper", cropper.mouse_callback)

    while True:
        cv2.imshow("Cropper", cropper.image_rectangle)

        key = cv2.waitKey(1) & 0xFF

        if key is ord('q'):
            cv2.destroyWindow("Cropper")
            return image
        elif key is ord('c'):
            cropped_image = cropper.crop()

            if cropped_image is not None:
                cv2.destroyWindow("Cropper")
                output = {}
                for _type in types:
                    if _type == 'roi':
                        output['roi'] = cropper.roi()
                    elif _type == 'mask':
                        output['mask'] = cropper.mask()
                    elif _type == 'crop':
                        output['crop'] = cropper.crop()

                return output

def write_image(filename,image,normalise=True):
    assert len(image.shape) < 3, 'Only support for grayscale images is enabled'
    if normalise:
        if not image.dtype == np.float:
            image=image.astype(np.float)
        image = image + image.min()
        image = image / image.max()
        image = image * 255
        image = image.astype('uint8')
    cv2.imwrite(filename,image)
            
def padd_image(image,padding):
    """Padds an image with random noise with similar mean + deviation
    PARAMS:
    image - an NxM image
    padding (left,top,right,bottom) in pixels
    """
    assert len(image.shape) ==2, "Invalid image size, expect NxM"
    left,top,right,bottom = padding
    output = np.random.randn(image.shape[0] + top + bottom,
                             image.shape[1] + left + right)
    output = image.std() * output + image.mean()
    
    output[top:top+image.shape[0],
           left:left+image.shape[1]] = image
    
    return output

def comatrix(image):
    """Calculate the 2D covariace matrix for an image
    """
    height, width = image.shape
    image = np.ma.MaskedArray(data = image,
                                     mask = (image==0))
    imageWorking = image
    nullKernel = np.zeros((7,7))
    smallAverageKernel = np.ones((7,7))
    
    k1 = np.zeros((7,7))
    np.copyto(nullKernel,k1)
    k1[3,3] = 1
    k1[0,6] = -1
    
    k2 = np.zeros((7,7))
    np.copyto(nullKernel,k2)
    k2[3,3] = 1
    k2[0,3] = -1
    
    k3 = np.zeros((7,7))
    np.copyto(nullKernel,k3)    
    k3[3,3] = 1
    k3[3,6] = -1

    k4 = np.zeros((7,7))
    np.copyto(nullKernel,k4)    
    k4[3,3] = 1
    k4[3,0] = -1

    k5 = np.zeros((7,7))
    np.copyto(nullKernel,k5)
    k5[3,3] = 1
    k5[6,6] = -0.14
    k5[5,5] = -0.86

    k6 = np.zeros((7,7))
    np.copyto(nullKernel,k6)    
    k6[3,3] = 1
    k6[0,0] = -0.14
    k6[1,1] = -0.86

    k7 = np.zeros((7,7))
    np.copyto(nullKernel,k7)    
    k7[3,3] = 1
    k7[6,0] = -0.14
    k7[5,1] = -0.86

    k8 = np.zeros((7,7))
    np.copyto(nullKernel,k8)    
    k8[3,3] = 1
    k8[0,6] = -0.14
    k8[1,5] = -0.86
    
    contrastM = scipy.signal.convolve2d(imageWorking,k1,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k2,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k3,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k4,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k5,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k6,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k7,'same')**2
    contrastM = contrastM + scipy.signal.convolve2d(imageWorking,k8,'same')**2
    olderr = np.seterr(divide='ignore')
    try:
        imageWorking = image / scipy.signal.convolve2d(imageWorking,np.ones((7,7)),'same')
    except RuntimeWarning:
        pass
    contrastM = contrastM / imageWorking**2
    np.seterr(**olderr)    
    cov=scipy.signal.convolve2d(contrastM, np.ones((5,5)), 'same')
    
    return cov

def unsharp(image,size,stdDevScale):
    '''Perform an unsharp mask on an image
    '''
    if not type(image)==np.ndarray:
        image=np.array(image)
    assert len(image.shape)==2, 'Expected an height x width image'
    image[image<0]=0
    
    h = np.ones((size,size))
    h = h / size**2
    imLow = scipy.signal.convolve2d(image,h,'same')
    imAvg = image.mean()
    result = (image - imLow) + imAvg
    avgResult = result.mean()
    stdResult = result.std()
    
    result = result - (avgResult - (stdResult * stdDevScale))
    result = result / (avgResult + (stdResult * stdDevScale))
    
    
    #I think this is the same as the matlab function imadjust(result,[0 1],[0 1])
    result[result < 0] = 0
    result[result > 1] = 2
    
    return result

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
    
def getInterlaceShift(image):
    """Calculate the row shift in pixels for poorly interlaced images
    """
    assert len(image.shape)==2, "Image must be NxM"
    nrows = image.shape[0]
    image_odd = image[np.arange(0,nrows,2)]
    image_even = image[np.arange(1,nrows,2)]
    
    shifts = find_frame_shift(image_odd,
                              image_even)
    
    return shifts['coords'][0]