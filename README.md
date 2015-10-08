# AO_Registration
Code for registering video output from high-resolution retinal imaging systems

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
    [SciPy](http://www.SciPy.org)
    [OpenCV](http://opencv.org/)

  ## Example usage
```python
import AoRegistration.AoRecording as AoRecording

vid = AoRecording.AoRecording(filepath='data/sample.avi')
vid.load_video()
vid.filter_frames()
vid.fixed_align_frames()
vid.complete_align()
vid.fast_align()
vid.write_video('output/output.avi')
```

## Testing
```
$ nosetests

```

## Notes
scipy.signal.correlate2d is slow as molases.
