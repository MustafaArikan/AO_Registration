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
    
    minCorrel = 0.38 # Used by make_valid_points .38 is OK for cones, other structures need lower values
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
                                      mask = np.tile(~self.mask,[self.nframes,1,1]))
                                      
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
            assert x2 >= x1 and x2 <= self.framewidth
            assert y1 >= 0
            assert y2 >= y1 and y2 <= self.frameheight
            
            mask = np.zeros((self.frameheight,self.framewidth),dtype=np.bool)
            mask[y1:y2, x1:x2] = 1
            self.mask = mask
            
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

        self.nframes = cap.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT)
        self.frameheight = cap.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        self.framewidth = cap.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        
        #preallocate a numpy array
        data = np.empty([self.nframes, self.frameheight, self.framewidth],dtype=np.uint8)
      
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
            left = max(c)
            r,c = np.where(frameSums[:,midRow:]==0)
            right = min(c) + midRow
            data = data[:,:,left:right]
            
        self.data=data
        self.currentStack = data
        

    def write_video(self,fpath):
        '''Write the current output to an avi'''
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        nframes, height,width = self.completeAlignedData.shape
        #fourcc = cv2.cv.CV_FOURCC(*'XVID')
        fourcc = cv2.cv.CV_FOURCC(*'I420')
        vid = cv2.VideoWriter(fpath,fourcc,10,(width,height))
        for idx in range(nframes):
            frame = np.uint8(self.completeAlignedData[idx,:,:])
            #frame = self.completeAlignedData[idx,:,:]
            frame = np.tile(frame,(3,1,1))
            frame = np.transpose(frame, (1,2,0))
            
            vid.write(frame)
        
        vid.release()              
        
    def write_average_frame(self,filename):
        assert self.currentAverageFrame is not None,'Average frame not created'
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        ImageTools.write_image(filename, self.currentAverageFrame)
        
    def write_frame(self,filename,frameTypes):
        #check to see if an error has occured before this
        if not self.b_continue:
            return      
        if isinstance(frameTypes,str):
            frameTypes = [frameTypes]
        for frameType in frameTypes:
            assert frameType in ['average','stdev']
            if frameType == 'average':
                self.write_average_frame(filename)
            if frameType == 'stdev':
                assert self.currentStdevFrame is not None, 'Stdev frame not created'
                ImageTools.write_image(filename,self.currentStdevFrame)
        
    def filter_frames(self):
        '''Perform an initial filtering on a frame set
        [goodframes,template] = filterFrames(framestack)
        returns the number of good frames, this can be used for error checking later
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
        framestack = framestack[good_brights,:,:] # only going to use good frames from here on.
    
        results = []

        midRow = int(self.framewidth / 2)
        midCol = int(self.frameheight / 2)        

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
            results.append({'frameid':iFrame,
                            'shear':shear['shearval'],
                            'correlation':displacement['maxcorr'],
                            'shift':displacement['coords'],
                            'motion':motion})
        if len(results) < 1:
            logger.debug("No good frames found")
            self.b_continue = 0
            return
        #data for frame 0 is missing, use the data from frame 1
        r=[r for r in results if r['frameid'] == 1]
        r=dict(r[0])    #make a copy of this item
        r['frameid']=0
        r['shift']=(0,0)
        r['motion']=0
        results.append(r)
            
        maxCorr = max([result['correlation'] for result in results])
        
        if maxCorr < self.minCorrel:
            #all correlations are crummy, just bail
            #TODO
            logger.debug('No good frames found')
            self.b_continue = 0
        else:
            self.goodFrames = [result['frameid'] for result in results if result['shear'] < 20 and result['correlation'] > 0.5 * maxCorr and result['motion'] < 50 ]
            self.templateFrame = self.goodFrames[frame_brightnesses[self.goodFrames].argmax()] #return the brightest of the remaining frames as a potential template
            self.filterResults = results
            
        
    def fixed_align_frames(self):
        '''perform fixed alignment on the framestack'''
        if self.goodFrames is None or self.templateFrame is None:
            logging.warning('filterframes not run first')
        
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        
        framesToProcess = [i for i in self.goodFrames if i is not self.templateFrame]
        
        midRow = int(self.framewidth / 2)
        midCol = int(self.frameheight / 2)        
        
        targetFrame = self.data[self.templateFrame,:,:]
        targetFrame = targetFrame[midRow - self.largeFrameSize : midRow + self.largeFrameSize,
                                  midCol - self.largeFrameSize : midRow + self.largeFrameSize]
        
        results = []
        #ensure the target frame is included in the output
        results.append({'frameid':self.templateFrame,
                        'correlation':1,
                        'shift':(0,0)})
        
        for iFrame in framesToProcess:
            templateFrame = self.data[iFrame,:,:]
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
        #results =[result for result in results if abs(result['shift'][1])<=50 and abs(result['shift'][0]) <= 50]        
        
        #sort the results array
        def byFrame_key(result):
            return result['frameid']
        
       
        results = sorted(results,key = byFrame_key)
        self.fixedDisplacements = [result['shift'] for result in results] #
        self.goodFrames = [result['frameid'] for result in results] #
        
        self.alignedData = StackTools.apply_displacements(self.data[self.goodFrames,:,:],self.fixedDisplacements)
        self.currentStack = self.alignedData
        
    def complete_align(self):
        """Takes a roughly aligned stack and performs a complete alignment
        """        
        if self.goodFrames is None or self.templateFrame is None:
            logging.warning('filterframes not run first')        
            
        if self.alignedData is None:
            logging.warning('fixed align not not run first')        
            
        nframes, nrows, ncols = self.alignedData.shape
        
        targetFrame_idx = self.goodFrames.index(self.templateFrame) #wish I had not called it template frame!!!
        targetFrameData = self.alignedData[targetFrame_idx]
        framesToProcess = [frameidx for frameidx in self.goodFrames if not frameidx == self.templateFrame]
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
        for idxFrame in range(len(framesToProcess)):
            #loop through all the frames here
            #need to generate a new mask for each frame
            image = self.alignedData[idxFrame,:,:]
            mask = np.zeros(image.shape,dtype=np.bool)
            mask[image > 0] = 1
            image = np.ma.array(image,
                                mask=~mask)
            randomData = image.std() * np.random.standard_normal(image.shape) + image.mean()
            image = (image.data * ~image.mask) + (randomData * image.mask) #no longer a masked array
            results.append({'frameid':framesToProcess[idxFrame],'stripResults':[]})
            for idxStrip in range(len(smallRowStart)):
                #loop through the strips here
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
                results[idxFrame]['stripResults'].append(displacement)
                
        newCoords = self._get_coords(nrows, ncols)
        timetics=[]
        for jndx in range(self.numberPointsToAlign):
            timetics.append(newCoords['times'][(smallRowStart[jndx]+int(self.smallSzRow/2)),
                                               (smallColStart+int(self.smallSzCol/2)-1)])
            
        self.timeTics = np.array(timetics)
        self.times = newCoords['times']
        self.alignmentSplines = self._make_valid_points(results)
        self.fast_align()

    def complete_align_parallel(self):
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        nFrames, nrows, ncols = self.alignedData.shape
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
        CompleteAlignParallel.complete_align_parallel(self.alignedData, self.goodFrames, 
                                                     self.templateFrame,
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
        self.alignmentSplines = self._make_valid_points(CompleteAlignParallel.results)
        self.fast_align_parallel()


    def _make_valid_points(self,displacements):
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
        correls = np.empty((len(self.goodFrames)-1,self.numberPointsToAlign))    #frames by strips N.B. template frame is excluded
        xShifts = np.empty((len(self.goodFrames)-1,self.numberPointsToAlign))
        yShifts = np.empty((len(self.goodFrames)-1,self.numberPointsToAlign))
        frameids = [frame['frameid'] for frame in displacements]
        for iFrame in range(len(displacements)):
            result=displacements[iFrame]
            stripResults = result['stripResults']
            for iSlice in range(len(stripResults)):
                correls[iFrame,iSlice] = stripResults[iSlice]['maxcorr']
                xShifts[iFrame,iSlice] = stripResults[iSlice]['coords'][0]
                yShifts[iFrame,iSlice] = stripResults[iSlice]['coords'][1]
        
        #this trims the first and last strips
        #ugly code...
        correls[:,0:2]=0
        correls[:,(correls.shape[1]-2):(correls.shape[1])]=0
        
        goodCorrels = correls > self.minCorrel

        dists = np.sqrt(xShifts**2 + yShifts**2)
        stdDist = np.std(dists,axis=1)   #deviation of displacements per frame
        goodStdDists = stdDist < self.maxStdDist
        goodDists = dists < self.maxDist
        
        goodPoints = np.logical_and(goodCorrels,goodDists)
        #require at least 10 good points per frame
        goodFrames = goodPoints.sum(axis=1) > 9
        goodFrames = np.logical_and(goodFrames,goodStdDists)
        output = []
        for iFrame in np.where(goodFrames)[0]:
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
        
    def fast_align_parallel(self):
        newCoords = self._get_coords(self.alignedData.shape[1],
                                     self.alignedData.shape[2]) 
        FastAlignParallel.fast_align_parallel(self.alignedData,
                                              self.alignmentSplines,
                                              self.goodFrames,
                                              self.templateFrame,
                                              newCoords)
        self.completeAlignedData = FastAlignParallel.outstack
        self.currentStack = self.completeAlignedData
        
    def fast_align(self):
        outputmargin = 30
        
        nrows = self.alignedData.shape[1]
        ncols = self.alignedData.shape[2]
        nframes = len(self.alignmentSplines)
        
        templateFrameIdx = self.goodFrames.index(self.templateFrame)        
        
        outputSizeRows = nrows + 2*outputmargin
        outputSizeCols = ncols + 2*outputmargin
        
        outstack = np.ones((nframes + 1, outputSizeRows, outputSizeCols))
        outstack = outstack * -1
        #insert the template frame unchanged
        outstack[templateFrameIdx,
                 outputmargin:outputmargin+nrows,
                 outputmargin:outputmargin+ncols] = self.alignedData[templateFrameIdx,:,:]
        
        interiorMask = outstack[templateFrameIdx,:,:] > -0.001 #there has to be a better way to do this.
        
        mask = self.alignedData[templateFrameIdx,:,:] > 0   #this mask is the size of the rough aligned images,true over the region of the template image, we will use it to ensure we only sample valid points
        
        newCoords = self._get_coords(self.alignedData.shape[1],
                                     self.alignedData.shape[2])
        times = newCoords['times']
        times = times.ravel()
        
        for frame in self.alignmentSplines:
            frameIdx = self.goodFrames.index(frame['frameid'])
            logging.debug('Aligning frame{}'.format(frameIdx))
            srcImg = self.alignedData[frameIdx,:,:] * mask
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
                
            outstack[frameIdx] = tmpFrame
            
        self.completeAlignedData = outstack
        self.currentStack = self.completeAlignedData
        
    def create_average_frame(self,type='mean'):
        assert type in ['lucky','mean']
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        if type == 'lucky':
            #creating a lucky average
            self.currentAverageFrame = StackTools.compute_lucky_image(self.currentStack)
        else:
            self.currentAverageFrame = self.currentStack.mean(axis=0)
            
    def create_stdev_frame(self):
        #check to see if an error has occured before this
        if not self.b_continue:
            return        
        self.currentStdevFrame = ComputeStdevImage.compute_stdev_image(self.currentStack)