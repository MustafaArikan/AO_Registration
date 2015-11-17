# author: Tom Wright
# website: tomwright.ca

# based on code by:
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

def write_image(filename,image,normalise=False):
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
    try:
        imageWorking = image / scipy.signal.convolve2d(imageWorking,np.ones((7,7)),'same')
    except RuntimeWarning:
        raise
    contrastM = contrastM / imageWorking**2
    
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
    
