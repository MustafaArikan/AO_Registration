import multiprocessing as mp
import numpy as np
import logging
import StackTools
import ImageTools
import FrameStack

logger = logging.getLogger(__name__)

results = []
timetics = []

sharedTargetFrameData = None
sharedSmallRowStart = None
sharedSmallColStart = None
sharedLargeRowStart = None
sharedLargeColStart = None
sharedSmallSzRow = None
sharedLargeSzRow = None
sharedSmallSzCol = None
sharedLargeSzCol = None

def _complete_align_parallel_callback(result):
    logger.debug('callback called')
    results.append(result)

def _complete_align_frame(image,frameid):
    """Perform strip alignment on a frame
    Expects to be called from complete_align_parallel
    Uses some objects placed in shared memory
    """
    #start building the output list
    result = {'frameid':frameid,'stripResults':[]}
    #rebuild the target image from the shared data
    targetFrameData = np.asarray(sharedTargetFrameData).reshape(image.shape) #assume that target frame and procFrame are the same shape
    smallRowStart = np.asarray(sharedSmallRowStart)
    largeRowStart = np.asarray(sharedLargeRowStart)
    smallColStart = np.asarray(sharedSmallColStart)[0]
    largeColStart = np.asarray(sharedLargeColStart)[0]
    largeSzRow = np.asarray(sharedLargeSzRow)[0]
    largeSzCol = np.asarray(sharedLargeSzCol)[0]
    smallSzRow = np.asarray(sharedSmallSzRow)[0]
    smallSzCol = np.asarray(sharedSmallSzCol)[0]
    
    
    #apply a random mask to the processed frame
    mask = np.zeros(image.shape,dtype=np.bool)
    mask[image > 0] = 1
    image = np.ma.array(image,
                        mask=~mask)
    randomData = image.std() * np.random.standard_normal(image.shape) + image.mean()
    image = (image.data * ~image.mask) + (randomData * image.mask) #no longer a masked array    
    
    for idxStrip in range(len(sharedSmallRowStart)):
        #loop through the strips here
        #print('from:{}, to:{}'.format(smallColStart[0],smallColStart[0] + smallSzCol))
        smallStrip = image[smallRowStart[idxStrip]:smallRowStart[idxStrip]+smallSzRow,
                           smallColStart:smallColStart + smallSzCol]
       
        largeStrip = targetFrameData[largeRowStart[idxStrip]:largeRowStart[idxStrip]+largeSzRow,
                                     largeColStart:largeColStart + largeSzCol]
        
        displacement = ImageTools.find_frame_shift(largeStrip, 
                                            smallStrip,
                                            topLeft=[(largeRowStart[idxStrip],largeColStart),
                                                     (smallRowStart[idxStrip],smallColStart)],
                                            method='xcorr',
                                            applyBlur=True,
                                            attemptSubPixelAlignment=True)
        #the offsets returned here are for the small strip within the large strip
        #coords = displacement['coords']
        #displacement['coords'] = (coords[0] + largeRowStart[idxStrip],
                                  #coords[1] + largeColStart)
        result['stripResults'].append(displacement)
    logger.debug('done parallel frame{}'.format(frameid))
    return result


def complete_align_parallel(alignedData,rowStarts,colStarts,rowSizes,colSizes):
    """Takes a roughly aligned stack and performs a complete alignment
    Params:
    alignedData - FrameStack.FrameStack object
    rowStarts - ([smallRowStarts], largeRowStart)
    colStarts - ([smallColStarts], largeColStart)
    rowSizes - (smallRowSize,largeRowSize)
    colSizes - (smallColSize,largeColSize)
    numberPointsToAlign - number of strips in each frame
    newCoords - height x width ndarray
    """        
    nFrames = alignedData.frameCount
    nrows = alignedData.frameHeight
    ncols = alignedData.frameWidth
    if nFrames <= 1:
        raise RuntimeError('Requires more than one frame')
    
    
    global results
    global timetics
    global sharedTargetFrameData
    global sharedSmallRowStart
    global sharedSmallColStart
    global sharedLargeRowStart
    global sharedLargeColStart
    global sharedSmallSzRow
    global sharedLargeSzRow
    global sharedSmallSzCol
    global sharedLargeSzCol
    #convert from original stack index to new stack index
    #localTargetFrameIdx = goodFrames.index(targetFrameIdx)
    targetFrameData = alignedData.templateFrame

    framesToProcess = [frameId for frameId in alignedData.frameIds if not frameId == alignedData.templateFrameId]
    #apply a mask to the target frame
    mask = np.zeros(targetFrameData.shape,dtype=np.bool)
    mask[targetFrameData > 0] = 1
    
    #convert the targetFrameData to a masked array for simple calculation of means
    targetFrameData = np.ma.array(targetFrameData,
                                  mask=~mask)
    randomData = targetFrameData.std() * np.random.standard_normal(targetFrameData.shape) + targetFrameData.mean()
    
    targetFrameData = (targetFrameData.data * ~targetFrameData.mask) + (randomData * targetFrameData.mask) #no longer a masked array
          
    #target frame and index vectors are also the same in every loop, put them in shared memory too
    sharedTargetFrameData = mp.Array('d',targetFrameData.ravel(),lock=False)    
    sharedSmallRowStart = mp.Array('i',rowStarts[0],lock=False)
    sharedSmallColStart = mp.Array('i',colStarts[0],lock=False)
    sharedLargeRowStart = mp.Array('i',rowStarts[1],lock=False)
    sharedLargeColStart = mp.Array('i',colStarts[1],lock=False)
    sharedSmallSzRow = mp.Value('i',rowSizes[0],lock=False)
    sharedLargeSzRow = mp.Value('i',rowSizes[1],lock=False)
    sharedSmallSzCol = mp.Value('i',colSizes[0],lock=False)
    sharedLargeSzCol = mp.Value('i',colSizes[1],lock=False)
    
    results = []
    pool = mp.Pool()
    for frameId in framesToProcess:
        pool.apply_async(_complete_align_frame,
                         args = (alignedData.get_frame_by_id(frameId),frameId),
                         callback=_complete_align_parallel_callback)
    pool.close()
    pool.join()    

