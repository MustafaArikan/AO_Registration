import numpy as np
import cv2
import logging

logger = logging.getLogger(__name__)

class FrameStack(object):
    """Wrapper class for ndarray that also stores the original frame ids
    frameids are the original frame number
    frameidx are the position in the current stack
    """
    def __init__(self,data,*args,**kargs):
        self.data = np.array(data)
        if len(self.data.shape) != 3:
            raise ValueError('Expected an nFrames x Height x Width array')
        
        if len(args) == 1:
            self.frameIdx = np.array(args[0])
        elif 'frameIds' in kargs.keys():
            self.frameIds = np.array(kargs['frameIds'])
        else:
            self.frameIds = np.array(range(self.data.shape[0]))
            
        if 'templateFrame' in kargs.keys():
            self._templateFrameId = kargs['templateFrame']
        else:
            self._templateFrameId = None
            
        assert self.frameIds.shape[0] == self.data.shape[0], "frame ids must be the same length as data"
        
    def __getattr__(self,name):
        """Delegate to NumPy array"""
        try:
            return getattr(self.data, name)
        except:
            raise AttributeError(
                "'Array' object has no attribute {}".format(name))

    def __getitem__(self,key):
        return self.data.__getitem__(key)
    
    def __setitem__(self,key,value):
        self.data.__setitem__(key,value)

    def __iter__(self):
        self._curpos = -1
        
    def next(self):
        if self._curpos < self.data.shape[0]:
            self._curpos = self._curpos + 1
            return (self.frameIds[self._curpos],
                    self.data[self._curpos,:,:])
        else:
            raise StopIteration
        
    
    def delete_frame_by_index(self,idx):
        """Remove a frame by it's index in the frame stack"""
        self.data = np.delete(self.data, idx, 0)
        self.frameIds = np.delete(self.frameIds, idx, 0)
        
    def delete_frame_by_id(self,frameId):
        """Remove a frame by it's id in the original file"""
        frameId = np.array(frameId)
        checkin = np.in1d(frameId,self.frameIds)
        if not all(checkin):
            raise ValueError("Frame id: {} not found".format(id))
        idx = np.in1d(self.frameIds,frameId,invert=True)
        self.frameIds = self.frameIds[idx]
        self.data = self.data[idx,:,:]

    def filter_frames_by_id(self,frameIds):
        """remove frames not in the list of ids"""
        frameIds = np.array(frameIds)
        checkIn = np.in1d(frameIds,self.frameIds)
        if not all(checkIn):
            raise ValueError('Invalid frame id(s):{} requested'.format(frameIds[checkIn]))
        checkIn = np.in1d(self.frameIds,frameIds)
        self.data = self.data[checkIn,:,:]
        self.frameIds = self.frameIds[checkIn]
        
    def filter_frames_by_idx(self,frameIdx):
        """Remove frames not in frameIdx"""
        frameIdx = np.array(frameIdx)
        self.data = self.data[frameIdx,:,:]
        self.frameIds = self.frameIds[frameIdx]
        
    def get_frame_by_id(self,frameid):
        """return a height x width array containing frame with id"""
        if frameid not in self.frameIds:
            raise ValueError('Invalid frame ID: {}'.format(frameid))
        return self.data[self.frameIds==frameid,:,:].squeeze()
                             
    def sort(self):
        #sort the data so it's output in the original frame order
        sortIdx = self.frameIds.argsort()
        self.frameIds = self.frameIds[sortIdx]
        self.data = self.data[sortIdx,:,:]        
        
    def write_stack(self,fpath,sort=True):
        """Write the current stack to an avi"""
        nframes, height,width = self.data.shape
        
        fourcc = cv2.cv.CV_FOURCC(*'I420')
        vid = cv2.VideoWriter(fpath,fourcc,10,(width,height))
        if sort:
            self.sort()
            
        for idx in range(nframes):
            frame = np.uint8(self.data[idx,:,:])
            frame = np.tile(frame,(3,1,1))
            frame = np.transpose(frame, (1,2,0))
            
            vid.write(frame)
        
        vid.release()              
        
    @property
    def frameHeight(self):
        return self.data.shape[1]
    
    @property
    def frameWidth(self):
        return self.data.shape[2]
    
    @property
    def frameCount(self):
        return len(self.frameIds)
    
    @property
    def templateFrame(self):
        if self._templateFrameId is None:
            logger.debug('Template frame not set')
            return None
        idx = np.in1d(self.frameIds,self._templateFrameId)
        return self.data[idx,:,:].squeeze()

    @property
    def templateFrameId(self):
        return self._templateFrameId
    
    @templateFrameId.setter   
    def templateFrameId(self,value):
        if value not in self.frameIds:
            raise ValueError('Invalid frame ID')
        self._templateFrameId = value
        
        
if __name__ == '__main__':
    data = np.random.random((3,5,5))
    fs = FrameStack(data)
    assert fs.templateFrameId is None
    fs.templateFrameId = 2
    assert fs.templateFrameId == 2
    assert len(fs.templateFrame.shape) == 2
    
    