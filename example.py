import AoRegistration.AoRecording as AoRecording
import AoRegistration.stackTools as stackTools
import timeit
import numpy as np
#x=AoRecording.AoRecording()
#x._get_coords(3,5)
vid = AoRecording.AoRecording(filepath='data/sample.avi')

tic=timeit.default_timer()
vid.filter_frames()
vid.fixed_align_frames()
#vid.write_video('output/output.avi')
vid.complete_align()
vid.fast_align()
toc = timeit.default_timer()
vid.write_video('output/output.avi')

print 'Process took {}:'.format(toc-tic)
