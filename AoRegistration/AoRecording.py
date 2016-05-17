import numpy as np
import os
import cv2
import logging
import scipy
import scipy.interpolate
import ImageTools
import StackTools
import CompleteAlignParallel
import FastAlignParallel
import ComputeStdevImage
import FrameStack

logger = logging.getLogger(__name__)

class AoRecording:
    """Class for information about a single AO recording.
    """
    #default values for xcorr frame alignment
    templateSize = 128  
    largeFrameSize = 231
    smallFrameSize = 91
    
    #default values for continous alignment
    smallSzRow = 25
    largeSzRow = 125
    smallSzCol = 151
    largeSzCol = 175
    numberPointsToAlign = 55
    
    maxStdDist = 49 # max allowable variance (std) in distances moved
    maxDist = 20    # max allowable distance of deviation in pixels
    
    

    def __init__(self,filepath=None):
        """Initialse an AoRecording object
        PARAMS: 
        [filepath] = full path to the source file
        filepath
        """
        self.filepath=filepath
        self.nframes=0
        self.frameheight=0
        self.framewidth=0
        self.data=[]
        self.mask=None
        self.goodFrames = None
        self.templateFrame = None
        self.filterResults = None
        self.timeTics = None
        self.currentStack = None
        self.currentStdevFrame = None
        self.currentAverageFrame = None
        self.b_continue = 1 # flag to check if an error in processing has occurred
    def get_masked(self):
        #returns an np.maskedarray type
        if self.mask is None:
            masked_data = np.ma.array(self.data,
                               mask = np.ones((self.data.shape),dtype=np.bool))
        else:
            masked_data = np.ma.array(self.data,
                                      mask = np.tile(~self.mask,[self.data.frameCount,1,1]))
                                      
        return masked_data
                
    def set_mask(self,roi=None):
        """Create a mask for the image
        Params:
        roi - [(x1,y1),(x2,y2)]
        
        If roi is None user is prompted to draw a mask, otherwise mask is created from roi"""
        if roi is None:
            mask = ImageTools.click_and_crop(self.data[0:,:,],types=['mask'])
            self.mask = mask['mask']
        else:
            x1,y1 = roi[0]
            x2,y2 = roi[1]

            assert x1 >= 0 
            assert x2 >= x1 and x2 <= self.data.frameWidth
            assert y1 >= 0
            assert y2 >= y1 and y2 <= self.data.frameHeight
            
            mask = np.zeros((self.data.frameHeight,self.data.frameWidth),dtype=np.bool)
            mask[y1:y2, x1:x2] = 1
            self.mask = mask
    def write_video(self,filename):
        """Write the current framestack to an avi"""
        self.data.write_stack(filename)
        
    def load_video(self, cropInterlace = True):
        """Loads an AO video
        Loads the video identified by filepath into a nframes height x width numpy array
        PARAMS:
        cropInterlace - boolean attempt to crop interlace bars at the sides of the video
        """
        RGB=False #indicator is video is in RGB format, in which case only use G channel
        
        cap = cv2.VideoCapture(self.filepath)
        if not cap.isOpened():
            logger.warning('Failed opening video: %s',self.filepath)
            return

        nframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        frameheight = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        framewidth = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        
        #preallocate a numpy array
        data = np.empty([nframes, frameheight, framewidth],dtype=np.uint8)
      
        ret, frame = cap.read() #get the first frame
        if len(frame.shape)>2:
            #frames are RGB format, using only G channel
            logger.debug('Video is in RGB format, using only G channel')
            RGB=True
            
        while(ret):
            frame_idx = cap.get(cv2.cv.CV_CAP_PROP_POS_FRAMES)
            if RGB:
                data[frame_idx - 1,:,:] = frame[:,:,1]
            else:
                data[frame_idx -1 ,:,:] = frame[:,:]
            ret, frame = cap.read()        
        cap.release()
        if cropInterlace:
            frameSums = data.sum(0)
            midRow = frameSums.shape[1]/2
            r,c = np.where(frameSums[:,0:midRow]==0)
            left = max(c) + 1
            r,c = np.where(frameSums[:,midRow:]==0)
            right = (min(c) - 1) + midRow
            #limit the interlace to a maximum of 200 pixels, otherwise can cause problems with v dark videos
            left = min([left,200])
            right = max([right,framewidth-200])
            data = data[:,:,left:right]
            
        self.data = FrameStack.FrameStack(data)
      
    def write_average_frame(self,filename):
        if self.currentAverageFrame is None:
            logger.error('Average frame not created')
            raise ValueError
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        ImageTools.write_image(filename, self.currentAverageFrame)
        
    def write_frame(self,filename,frameTypes):
        #check to see if an error has occured before this
        if self.data is None:
            return
        if isinstance(frameTypes,str):
            frameTypes = [frameTypes]
        for frameType in frameTypes:
            assert frameType in ['average','stdev']
            if frameType == 'average':
                self.write_average_frame(filename)
            if frameType == 'stdev':
                if self.currentStdevFrame is not None:
                    ImageTools.write_image(filename,self.currentStdevFrame)
        
    def filter_frames(self, minCorr=0.38):
        '''Perform an initial filtering on a frame set
        filterFrames()
        minCorr default 0.38 is ok for cones, other structures require lower values
        '''
        #check to see if an error has occured before this
        if not self.b_continue:
            return
        
        framestack = self.data
        
        # calculate mean brightness for each frame, 
        #  if framestack is a masked array masked values are ignored
        frame_brightnesses = np.apply_over_axes(np.mean, framestack, [1,2]).flatten()
        max_bright = frame_brightnesses.max()
        
        #find frame index for frames >50% max_bright
        #  frames not in this list will be excluded
        good_brights = np.array(frame_brightnesses > max_bright * 0.5, dtype=np.bool)
        #good_brights = [i for i, val in enumerate(frame_brightnesses) if val > 0.5* max_bright]
        brightestFrames = np.array(frame_brightnesses > max_bright * 0.85, dtype=np.bool)
        
        framelist = np.where(good_brights)[0]
        framestack.filter_frames_by_idx(good_brights) # only going to use good frames from here on.

        if len(good_brights) < 1:
            logger.error("No good frames found")
            self.data = None
            raise RuntimeError("No good frames found:Brightness too low")
    
        results = []

        midRow = int(framestack.frameWidth / 2)
        midCol = int(framestack.frameHeight / 2)        

        for iFrame in np.arange(1,len(framelist)):
            currImage = framestack[iFrame - 1,:,:] #target frame
            tempImage = framestack[iFrame,:,:] #template frame
            
            shear = StackTools.find_frame_shear(currImage, tempImage)
            tempImage = tempImage[midRow - self.templateSize / 2 : midRow + self.templateSize / 2,
                                  midCol - self.templateSize / 2 : midCol + self.templateSize / 2]            
                
            displacement = StackTools.find_frame_shift(currImage,
                                                       tempImage,
                                                       topLeft=[(0,0),
                                                                (midRow - self.templateSize / 2,midCol - self.templateSize / 2)],
                                                       method='xcorr',
                                                       applyBlur=True,
                                                       attemptSubPixelAlignment=False)
            motion = (displacement['coords'][0]**2 + displacement['coords'][0]**2)**0.5
            results.append({'frameid':framestack.frameIds[iFrame],
                            'shear':shear['shearval'],
                            'correlation':displacement['maxcorr'],
                            'shift':displacement['coords'],
                            'motion':motion})
        #filter frames where sheer > 20
        #results = [r for r in results if r['motion'] <= 20]
        #if len(results) < 1:
            #logger.error("No good frames found")
            #self.data = None
            #raise RuntimeError("No good frames found:Shear too high")

        #data for frame 0 is missing, use the data from the first remaining frame
        #r=[r for r in results if r['frameid'] == 1]
        if not results:
            raise RuntimeError('Could not get displacements')
        r =dict(results[0])    #make a copy of this item
        r['frameid']=framelist[0]
        r['shift']=(0,0)
        r['motion']=0
        results.append(r)
            
        maxCorr = max([result['correlation'] for result in results])
        
        if maxCorr < minCorr:
            #all correlations are crummy, just bail
            #TODO
            logger.warning('No good frames found')
            raise RuntimeError("No good frames found:Correlation too low")
        else:
            goodFrames = [result['frameid'] for result in results if result['shear'] < 20 and result['correlation'] > 0.5 * maxCorr and result['motion'] < 50 ]
            badFrames = [frameid for frameid in self.data.frameIds if frameid not in goodFrames]
            if not goodFrames:
                logger.warning('No good frames found')
                raise RuntimeError('No good frames found:Group criteria (Shear, Correlation, Motion) not met.')
                
            logger.info('Removing frames {} due to brightness or shear'.format(badFrames))
            self.data.filter_frames_by_id(goodFrames)
            self.data.templateFrameId = goodFrames[frame_brightnesses[goodFrames].argmax()] #return the brightest of the remaining frames as a potential template
            self.filterResults = results    #store this for debugging
            
        
    def fixed_align_frames(self,maxDisplacement=50):
        '''perform fixed alignment on the framestack
        maxDisplacement=50 - maximum allowed displacement, frames with > than this will be removed from the stack'''
        if self.data is None:
            logger.warning('No frames found')
            return
            
        if self.data.templateFrame is None:
            logger.warning('template frame not set')
            return
        
        framesToProcess = [i for i in self.data.frameIds if i != self.data.templateFrameId]
        
        midRow = int(self.data.frameWidth / 2)
        midCol = int(self.data.frameHeight / 2)        
        
        targetFrame = self.data.templateFrame
        targetFrame = targetFrame[midRow - self.largeFrameSize : midRow + self.largeFrameSize,
                                  midCol - self.largeFrameSize : midRow + self.largeFrameSize]
        
        results = []
        #ensure the target frame is included in the output
        results.append({'frameid':self.data.templateFrameId,
                        'correlation':1,
                        'shift':(0,0)})
        
        for iFrame in framesToProcess:
            templateFrame = self.data.get_frame_by_id(iFrame)
            templateFrame = templateFrame[midRow - self.smallFrameSize : midRow + self.smallFrameSize,
                                          midCol - self.smallFrameSize : midCol + self.smallFrameSize]
        
            displacement = StackTools.find_frame_shift(targetFrame,
                                                       templateFrame,
                                                       topLeft=[(midRow - self.largeFrameSize,midCol - self.largeFrameSize),
                                                                (midRow - self.smallFrameSize,midCol - self.smallFrameSize)],
                                                       applyBlur=True,
                                                       method='xcorr',
                                                       attemptSubPixelAlignment=False)
            
            results.append({'frameid':iFrame,
                            'correlation':displacement['maxcorr'],
                            'shift':displacement['coords']})
        #Check displacement is les than 50 pixels
        good_results = [result for result in results 
                        if abs(result['shift'][1])<=maxDisplacement 
                        and abs(result['shift'][0]) <= maxDisplacement]
        bad_results = [result['frameid'] for result in results 
                        if abs(result['shift'][1]) > maxDisplacement 
                        or abs(result['shift'][0]) > maxDisplacement]
        logger.info('Removing frames {} for too large displacements'.format(bad_results))
        if not good_results:
            #no good frames found
            logger.warning('frame displacements are too large')
            raise RuntimeError('frame displacements are too large')
        
        alignedData = StackTools.apply_displacements(self.data,good_results)
        self.data = alignedData
        
        self.currentStack = alignedData
        
        
    def complete_align(self,minCorr = 0.38):
        """Takes a roughly aligned stack and performs a complete alignment
        minCorr (default 0.38, minimum correlation for inclusion in the output stack)
        """        
        if self.data is None:
            logger.warning('Aborting:No good frames found')
            return
        nrows,ncols = self.data.frameHeight, self.data.frameWidth
        
        targetFrameData = self.data.templateFrame
        framesToProcess = [frameid for frameid in self.data.frameIds if not frameid == self.data.templateFrameId]
        #apply a mask to the target frame
        mask = np.zeros(targetFrameData.shape,dtype=np.bool)
        mask[targetFrameData > 0] = 1
        
        #convert the targetFrameData to a masked array for simple calculation of means
        targetFrameData = np.ma.array(targetFrameData,
                                      mask=~mask)
        randomData = targetFrameData.std() * np.random.standard_normal(targetFrameData.shape) + targetFrameData.mean()
        
        targetFrameData = (targetFrameData.data * ~targetFrameData.mask) + (randomData * targetFrameData.mask) #no longer a masked array
        
        #setup the row indices
        defaultStep = int((nrows - self.smallSzRow + 1) / (self.numberPointsToAlign))
        smallRowStart = np.array(range(self.numberPointsToAlign)) * defaultStep
        
        #the large rows should be centered on the small rows
        halfDifference = int((self.largeSzRow - self.smallSzRow) / 2)
        largeRowStart = smallRowStart - halfDifference # this gives some values out of bounds
        largeRowStart[largeRowStart < 0] = 0
        maxRowStart = nrows - self.largeSzRow
        largeRowStart[largeRowStart > maxRowStart] = maxRowStart
        
        smallColStart = (ncols / 2) - (self.smallSzCol / 2)
        largeColStart = (ncols / 2) - (self.largeSzCol / 2)
        
        results = []
        for frameId in framesToProcess:
            #loop through all the frames here
            #need to generate a new mask for each frame
            image = self.data.get_frame_by_id(frameId)
            mask = np.zeros(image.shape,dtype=np.bool)
            mask[image > 0] = 1
            image = np.ma.array(image,
                                mask=~mask)
            randomData = image.std() * np.random.standard_normal(image.shape) + image.mean()
            image = (image.data * ~image.mask) + (randomData * image.mask) #no longer a masked array
            results.append({'frameid':frameId,'stripResults':[]})
            for idxStrip in range(len(smallRowStart)):
                #loop through the strips here
                stripResults = [result['stripResults'] for result in results if result['frameid'] == frameId][0]
                smallStrip = image[smallRowStart[idxStrip]:smallRowStart[idxStrip]+self.smallSzRow,
                                   smallColStart:smallColStart + self.smallSzCol]
            
                largeStrip = targetFrameData[largeRowStart[idxStrip]:largeRowStart[idxStrip]+self.largeSzRow,
                                             largeColStart:largeColStart + self.largeSzCol]
                
                displacement = StackTools.find_frame_shift(largeStrip, 
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
                stripResults.append(displacement)
                
        newCoords = self._get_coords(nrows, ncols)
        timetics=[]
        for jndx in range(self.numberPointsToAlign):
            timetics.append(newCoords['times'][(smallRowStart[jndx]+int(self.smallSzRow/2)),
                                               (smallColStart+int(self.smallSzCol/2)-1)])
            
        self.timeTics = np.array(timetics)
        self.times = newCoords['times']
        alignmentSplines = self._make_valid_points(results,minCorr)
        self.data = self.fast_align(alignmentSplines)

    def complete_align_parallel(self,minCorr = 0.38):
        """Takes a roughly aligned stack and performs a complete alignment
            minCorr (default 0.38, minimum correlation for inclusion in the output stack)
            """                
        #check to see if an error has occured before this
        if self.data is None:
            logger.warning('Aborting:No good frames found')
            return
        nrows,ncols = self.data.frameHeight, self.data.frameWidth

        newCoords = self._get_coords(nrows, ncols)

        #setup the row indices
        defaultStep = int((nrows - self.smallSzRow + 1) / (self.numberPointsToAlign))
        smallRowStart = np.array(range(self.numberPointsToAlign)) * defaultStep
    
        #the large rows should be centered on the small rows
        halfDifference = int((self.largeSzRow - self.smallSzRow) / 2)
        largeRowStart = smallRowStart - halfDifference # this gives some values out of bounds
        largeRowStart[largeRowStart < 0] = 0
        maxRowStart = nrows - self.largeSzRow
        largeRowStart[largeRowStart > maxRowStart] = maxRowStart
    
        smallColStart = (ncols / 2) - (self.smallSzCol / 2)
        largeColStart = (ncols / 2) - (self.largeSzCol / 2)
        logging.debug('Starting parallel alignment')
        CompleteAlignParallel.complete_align_parallel(self.data,
                                                     (smallRowStart,largeRowStart),
                                                     (smallColStart,largeColStart),
                                                     (self.smallSzRow,self.largeSzRow), 
                                                     (self.smallSzCol,self.largeSzCol))

        
            
        timetics=[]
        for jndx in range(self.numberPointsToAlign):
            timetics.append(newCoords['times'][(smallRowStart[jndx]+int(self.smallSzRow/2)),
                                               (smallColStart+int(self.smallSzCol/2)-1)])
    
        self.timeTics = np.array(timetics)
        self.times = newCoords['times']
        alignmentSplines = self._make_valid_points(CompleteAlignParallel.results,minCorr)
        self.data = self.fast_align_parallel(alignmentSplines)


    def _make_valid_points(self,displacements,minCorr):
        """Takes the displacements created by complete_align() and converts them into a series of fitted splines
        returns a list of dicts, one dict for each frame
        {'frameid':original frame number
         'ppx':splines for generating x coords
         'ppy':splines for generating y coords
         }
         N.B. there may not be a dict entry for every frame
         """

        if self.timeTics is None:
            logger.debug('Complete alignment not completed')
            raise ValueError            
        
        #convert the complete alignment results to arrays
        correls = np.empty((len(displacements),self.numberPointsToAlign),dtype=np.float32)    #frames by strips N.B. template frame is excluded
        xShifts = np.empty((len(displacements),self.numberPointsToAlign),dtype=np.float32)
        yShifts = np.empty((len(displacements),self.numberPointsToAlign),dtype=np.float32)
        frameids = [frame['frameid'] for frame in displacements]
        frameIdx = -1
        for frameDisplacement in displacements:
            frameIdx = frameIdx + 1
            stripResults = frameDisplacement['stripResults']
            correls[frameIdx,:] = [stripResult['maxcorr'] for stripResult in stripResults]
            xShifts[frameIdx,:] = [stripResult['coords'][0] for stripResult in stripResults]
            yShifts[frameIdx,:] = [stripResult['coords'][1] for stripResult in stripResults]
            
            
        #this trims the first and last strips
        #ugly code...
        correls[:,0:2]=0
        correls[:,(correls.shape[1]-2):(correls.shape[1])]=0
        
        goodCorrels = correls > minCorr

        dists = np.sqrt(xShifts**2 + yShifts**2)
        stdDist = np.std(dists,axis=1)   #deviation of displacements per frame
        goodStdDists = stdDist < self.maxStdDist
        goodDists = dists < self.maxDist
        
        goodPoints = np.logical_and(goodCorrels,goodDists)
        #require at least 10 good points per frame
        goodFrames = goodPoints.sum(axis=1) > 9
        goodFrames = np.logical_and(goodFrames,goodStdDists)
        output = []
        goodFrame_list = np.where(goodFrames)[0].tolist()
        badFrames = np.array(frameids)[~goodFrames].tolist()
        if len(badFrames)>0:
            logger.info('Removing frames {} for bad strip alignments'.format(badFrames))
            self.data.delete_frame_by_id(badFrames)
        else:
            logger.info('All frames have good strip alignments')
            
        for iFrame in goodFrame_list:
            #work through the list of good frames
            displaceX = xShifts[iFrame,goodPoints[iFrame,:]]
            displaceY = yShifts[iFrame,goodPoints[iFrame,:]]
            times = self.timeTics[goodPoints[iFrame,:]]
            #going to apply a moving average of size 5, need to padd the sequences with 2 extra values at each end
            #while we are at it, padd with an extra value for time = 0 and time = maxTime for the spline fitting
            nrep = 3
            displaceX = np.insert(displaceX, 0, [displaceX[0]] * nrep )
            displaceX = np.insert(displaceX, len(displaceX), [displaceX[-1]] * nrep)
            displaceY = np.insert(displaceY, 0, [displaceY[0]] * nrep)
            displaceY = np.insert(displaceY, len(displaceY), [displaceY[-1]] * nrep)            
            times = np.insert(times,[0,-1],[0,self.times.max()])
            
            displaceX = self._smooth(displaceX)
            displaceY = self._smooth(displaceY)
            
            # resample at a higher frequency
            freqFactor = 20
            displaceX = np.interp(np.linspace(0,len(displaceX),num=len(displaceX)*freqFactor),
                                  np.linspace(0,len(displaceX),num=len(displaceX)),
                                  displaceX)
            displaceY = np.interp(np.linspace(0,len(displaceY),num=len(displaceY)*freqFactor),
                                  np.linspace(0,len(displaceY),num=len(displaceY)),
                                  displaceY)
            times = np.interp(np.linspace(0,len(times),num=len(times)*freqFactor),
                                  np.linspace(0,len(times),num=len(times)),
                                  times)
            displaceX = scipy.interpolate.UnivariateSpline(times, 
                                                           displaceX)
            displaceY = scipy.interpolate.UnivariateSpline(times, 
                                                           displaceY)
            output.append({'frameid':frameids[iFrame],
                           'ppx':displaceX,
                           'ppy':displaceY})
        
        return output


    def _smooth(self,seq,n=5):
        """Apply a moving average to an input sequence
        returns a smoothed sequence of length len(seq) - n + 1"""
        ret = np.cumsum(seq, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n        

    def _get_coords(self,numrows,numcols):
        """return the timing of pixels in a padded frame in terms of the original samples
        """
        pixPerLineOrig = np.float(1000)
        #pixPerLineOrig = np.float(5)   #debug code
        linesPerFrameOrig = np.float(1024)
        sampRate = np.float(23) #Hz
        
        sampTime = np.float(1 / (pixPerLineOrig * linesPerFrameOrig * sampRate))  #sample time per pixel
        #sampTime = 1 #debug code
        lineTime = np.float(pixPerLineOrig * sampTime)
        frameTime = np.float(numrows * lineTime)
        
        #i think this is an error in sburns code        
        #rows = np.array(range(numrows)) 
        #cols = np.array(range(numcols))        
        
        rows = np.array(range(numrows)) 
        cols = np.array(range(numcols))
        
        coltime = np.atleast_2d(cols * sampTime)
        rowtime = np.atleast_2d(rows * lineTime)
        
        [rowlocs,collocs] = np.meshgrid(rows,cols);
        [rowlocs,collocs]= [a.T for a in [rowlocs, collocs]]
        times = rowtime + coltime.T
        times = times.T
        return {'rowlocs':rowlocs,'collocs':collocs,'times':times,'FrameTimeIncrement':frameTime}
        
    def fast_align_parallel(self,alignmentSplines):
        newCoords = self._get_coords(self.data.frameHeight,
                                     self.data.frameWidth) 
        alignedData = FastAlignParallel.fast_align_parallel(self.data,
                                              alignmentSplines,
                                              newCoords)
        return alignedData
        
        
    def fast_align(self,alignmentSplines):
        outputmargin = 30
        
        nrows = self.data.frameHeight
        ncols = self.data.frameWidth
        
        nframes = len(alignmentSplines)
        
        templateFrameIdx = self.data.get_idx_from_id(self.data.templateFrameId)
        
        outputSizeRows = nrows + 2*outputmargin
        outputSizeCols = ncols + 2*outputmargin
        
        outstack = np.ones((nframes + 1, outputSizeRows, outputSizeCols))
        outstack = outstack * -1
        #insert the template frame unchanged in first position
        outstack[templateFrameIdx,
                 outputmargin:outputmargin+nrows,
                 outputmargin:outputmargin+ncols] = self.data.templateFrame
        
        interiorMask = outstack[templateFrameIdx,:,:] > -0.001 #there has to be a better way to do this.
        
        mask = self.data.templateFrame > 0   #this mask is the size of the rough aligned images,true over the region of the template image, we will use it to ensure we only sample valid points
        
        newCoords = self._get_coords(self.data.frameHeight,
                                     self.data.frameWidth)
        times = newCoords['times']
        times = times.ravel()
        
        for frame in alignmentSplines:
            frameIdx = self.data.get_idx_from_id(frame['frameid'])

            logging.debug('Aligning frame{}'.format(frame['frameid']))
            
            srcImg = self.data.get_frame_by_id(frame['frameid']) * mask
            srcImg = srcImg + ((srcImg==0) * -1)
            
            tmpFrame = np.ones(interiorMask.shape) * -1
            tmpFrame = tmpFrame + interiorMask
            
            newx = frame['ppx'](times).reshape(nrows,ncols)
            newy = frame['ppy'](times).reshape(nrows,ncols)
            
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
                
            outstack[frameIdx,:,:] = tmpFrame
            
        return FrameStack.FrameStack(outstack,
                                     frameIds = self.data.frameIds,
                                     templateFrame = self.data.templateFrameId)
        
    def create_average_frame(self,type='mean'):
        assert type in ['lucky','mean']
        #check to see if an error has occured before this
        if self.data is None:
            logger.debug('')
        if type == 'lucky':
            #creating a lucky average
            if self.data.frameCount > 20:
                self.currentAverageFrame = StackTools.compute_lucky_image(self.data)
            else:
                self.currentAverageFrame = None
                logger.warning('Too few frames to create lucky average')
                
        else:
            self.currentAverageFrame = self.data.mean(axis=0)
            
    def create_stdev_frame(self):
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        self.currentStdevFrame = ComputeStdevImage.compute_stdev_image(self.data)