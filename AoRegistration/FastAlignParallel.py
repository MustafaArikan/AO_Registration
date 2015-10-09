import multiprocessing as mp
import numpy as np
import logging

logger = logging.getLogger(__name__)

outstack = None

def fast_align_parallel(imageStack,alignmentSplines,goodFrames,targetFrameNo,newCoords):
    """apply image alignment splines to an input image stack
    Params:
    imageStack nFrames x height x width ndarray
    alignmentSplines - list of splines, each item is a dict with entries 'frameid' - frame number in original input. 'ppx','ppy'- splines for x & y coordinates
    goodFrames - List of positions of the original frames in the imageStack
    targetFrameNo - Original frame number of the template frame in the original input
    newCoords - height x width ndarray containing pixel sample times
    """   
    global outstack
    
    outputmargin = 30
    assert len(alignmentSplines) == (imageStack.shape[0]-1), 'alignmentSplines should be list of length imageStack.shape[0]-1'
    nframes, nrows, ncols = imageStack.shape

    nframes = len(alignmentSplines)
       
    templateFrameIdx = goodFrames.index(targetFrameNo)        
    
    outputSizeRows = nrows + 2*outputmargin
    outputSizeCols = ncols + 2*outputmargin
    
    outstack = np.ones((nframes + 1, outputSizeRows, outputSizeCols))
    outstack = outstack * -1
    #insert the template frame unchanged
    outstack[templateFrameIdx,
             outputmargin:outputmargin+nrows,
             outputmargin:outputmargin+ncols] = imageStack[templateFrameIdx,:,:]
    
    interiorMask = outstack[templateFrameIdx,:,:] > -0.001 #there has to be a better way to do this.
    
    mask = imageStack[templateFrameIdx,:,:] > 0   #this mask is the size of the rough aligned images,true over the region of the template image, we will use it to ensure we only sample valid points
    
    
    pool=mp.Pool()

    for idx in range(nframes):
        frameid = alignmentSplines[idx]['frameid']
        frameIdx = goodFrames.index(frameid)
        
        srcImg = imageStack[frameIdx,:,:]
        
        pool.apply_async(_align_frame,args=(srcImg,
                                            alignmentSplines[idx],
                                            frameIdx,
                                            mask,
                                            interiorMask,
                                            newCoords,
                                            outputmargin),
                                      callback = _align_frame_parallel_callback)
                                      

    pool.close()
    pool.join()        

def _align_frame(srcImg,splines,frameIdx,mask,interiorMask,newCoords,outputmargin):
    """function to align a frame using splines
    frameIdx here is the index of the frame in the aligned imageStack"""
    logger.debug('aligning frame:{}'.format(frameIdx))
    nrows, ncols = srcImg.shape
    outputSizeRows, outputSizeCols = interiorMask.shape
    
    
    srcImg = srcImg * mask
    srcImg = srcImg + ((srcImg==0) * -1)
    
    tmpFrame = np.ones(interiorMask.shape) * -1
    tmpFrame = tmpFrame + interiorMask

    times = newCoords['times'].ravel()

    newx = splines['ppx'](times).reshape(nrows,ncols)
    newy = splines['ppy'](times).reshape(nrows,ncols)    
    
    finalCols = np.round(newCoords['collocs'] + newx + outputmargin)
    finalRows = np.round(newCoords['rowlocs'] + newy + outputmargin)    
    
    mask2 = mask * (finalRows > 0.5)
    mask2 = mask2 * (finalRows < outputSizeRows)
    mask2 = mask2 * (finalCols > 0.5)
    mask2 = mask2 *(finalCols < outputSizeCols)

    validRows, validCols = np.where(mask2)    
    
    for idx in range(len(validRows)):
        #for each valid pixel, take it from the source image and place it in the new location
        tmpFrame[finalRows[validRows[idx],validCols[idx]],
                 finalCols[validRows[idx],validCols[idx]]] = srcImg[validRows[idx],
                                                                    validCols[idx]]    
        
    return (frameIdx,tmpFrame)

def _align_frame_parallel_callback(result):
    outstack[result[0],:,:]=result[1]
