# AO_Registration
Code for registering video output from high-resolution retinal imaging systems

## Notes
* Added parallel processing, reduced times for a 32 frame video (30 good frames) from 80 seconds to ~45 seconds.
* 26.10.2015 added a new FrameStack object to hold video frames, extends ndarray  object to keep track of frame ids

## Funding

Canadian Institute for Heath Research grant 106560

## Acknowledgements

This work is based on code kindly shared by:

  **Stephen A. Burns, Ph.D.**  
  *Indiana University*

  *funding sources:*
    - Foundation Fighting Blindness grant TA-CL-0613-0617-IND

    - NIH grant P20 EY019008


  ## Dependencies
    [python 2.7](http://www.python.com)
    [SciPy 0.14](http://www.SciPy.org)
    [OpenCV 2.4](http://opencv.org/)

  ## Example usage
```python
import AoRegistration.AoRecording as AoRecording

vid = AoRecording.AoRecording(filepath='data/sample.avi')
# load the video into memory
vid.load_video()
# Perform an initial filter, excludes frames that are too dim (blink)
#   or contain shear (saccades).
vid.filter_frames()
# Perform a rough (translation) alignment
vid.fixed_align_frames()
# Calculate the within frame splines used for strip alignment

#vid.complete_align() #for serial processing
vid.complete_align_parallel() # for parallel processing

# apply the within frame alignment splines to the frames
vid.fast_align()

# output the registered video (caution: fails silently if output folder doesn't exist)
vid.write_video('output/output.avi')
```

From the command line:
```
$ python example.py -v
```

## Testing
```
$ nosetests

```

## Observations
scipy.signal.correlate2d is slow as molases.
