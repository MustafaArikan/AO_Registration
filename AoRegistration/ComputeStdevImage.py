import numpy as np
import multiprocessing as mp
import logging
import StackTools
import ImageTools
import cv2

logger = logging.getLogger(__name__)

results = None

def compute_variance_image(image,medImage, idx = None):
    """Compute the variance image of a frame
    medImage - an ndarray of same size as image with stack median values for each pixel, contains np.nan values outside of the mask region
    """
    assert len(image.shape)==2, 'Expected an height x width image'
    assert np.all(np.equal(image.shape,medImage.shape))
    logger.debug('calulating variance for frame: {}'.format(idx))
    image = image.astype(np.float32) #medianBlur function only works on 32bit floats
    image = cv2.medianBlur(image,3)
    
    #mask the image
    medMask = image > 0
    medMask = cv2.erode(medMask.astype(np.float32),np.ones((3,3)),1)
    image = image * medMask
    image = image + (np.logical_not(medMask) * medImage)
    
    image = ImageTools.unsharp(image,21,3)
    if idx is not None:
        return [idx, image]
    else:
        return image

def _compute_stdev_image_callback(result):
    logger.debug('recieved {}'.format(result[0]))
    results[result[0],:,:] = result[1]
    

def compute_stdev_image(stack):
    '''Compute the standard deviation image'''
    nFrames, height, width = stack.shape
    if nFrames < 10:
        logger.error('stdev image requires at least 10 frames')
        return None
    
    global results
    results = np.empty(stack.shape)
    
    stack = StackTools.getMaskedStack(stack)
    medianImage = np.median(stack,axis=0)

    #pool = mp.Pool()
    for idxFrame in range(stack.shape[0]):
        srcImg = stack[idxFrame,:,:]
        #pool.apply_async(compute_variance_image,
                         #args=(srcImg,
                               #medianImage,
                               #idxFrame),
                         #callback=_compute_stdev_image_callback)
        results[idxFrame,:,:] = compute_variance_image(srcImg,medianImage,idxFrame)[1]
    #pool.close
    #pool.join
        
    #newStack = np.ma.array(results)
    imageStdev = results.std(axis=0)
    imageAvg = results.mean(axis=0)
    #some values in imageAvg can be 0, create this as a masked array
    imageAvg = np.ma.MaskedArray(data = imageAvg,
                                 mask = imageAvg == 0)
    imageNorm = imageAvg / imageAvg.max()
    
    imageRatio = imageStdev / imageNorm
    imageRatio[imageRatio > 0.3] = 0.3
    imageRatio = cv2.medianBlur(imageRatio.astype(np.float32),3)
    
    return imageRatio
    
    
    