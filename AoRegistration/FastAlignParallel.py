import multiprocessing as mp
import numpy as np
import logging
import FrameStack

logger = logging.getLogger(__name__)

outstack = None

def fast_align_parallel(imageStack,alignmentSplines,newCoords):
    """apply image alignment splines to an input image stack
    Params:
    imageStack FrameStack.FrameStack object
    alignmentSplines - list of splines, each item is a dict with entries 'frameid' - frame number in original input. 'ppx','ppy'- splines for x & y coordinates
    newCoords - height x width ndarray containing pixel sample times
    """
    global outstack

    outputmargin = 30
    assert len(alignmentSplines) == (imageStack.frameCount-1), 'alignmentSplines should be list of length imageStack.shape[0]-1'


    nrows = imageStack.frameHeight
    ncols = imageStack.frameWidth

    nframes = len(alignmentSplines)


    templateFrameIdx = imageStack.get_idx_from_id(imageStack.templateFrameId)

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

    for alignmentSpline in alignmentSplines:
        frameId = alignmentSpline['frameid']
        frameIdx = imageStack.get_idx_from_id(frameId)

        srcImg = imageStack.get_frame_by_id(frameId)

        pool.apply_async(_align_frame,args=(srcImg,
                                            alignmentSpline,
                                            frameId,
                                            frameIdx,
                                            mask,
                                            interiorMask,
                                            newCoords,
                                            outputmargin),
                                      callback = _align_frame_parallel_callback)


    pool.close()
    pool.join()

    return FrameStack.FrameStack(outstack,
                                 frameIds = imageStack.frameIds,
                                 templateFrame = imageStack.templateFrameId)

def _align_frame(srcImg,splines,frameId,frameIdx,mask,interiorMask,newCoords,outputmargin):
    """function to align a frame using splines
    frameIdx here is the index of the frame in the aligned imageStack"""
    logger.debug('aligning frame:{}'.format(frameId))
    nrows, ncols = srcImg.shape
    outputSizeRows, outputSizeCols = interiorMask.shape


    srcImg = srcImg * mask
    srcImg = srcImg + ((srcImg==0) * -1)

    tmpFrame = np.ones(interiorMask.shape) * -1
    tmpFrame = tmpFrame + interiorMask

    times = newCoords['times'].ravel()

    newx = splines['ppx'](times).reshape(nrows,ncols)
    newy = splines['ppy'](times).reshape(nrows,ncols)

    finalCols = np.int64(np.round(newCoords['collocs'] + newx + outputmargin))
    finalRows = np.int64(np.round(newCoords['rowlocs'] + newy + outputmargin))

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
