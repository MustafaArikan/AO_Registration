import AoRegistration.AoRecording as AoRecording
import AoRegistration.stackTools as stackTools
import timeit

vid = AoRecording.AoRecording(filepath='data/sample.avi')
vid.load_video()

tic=timeit.default_timer()
vid.filter_frames()
vid.fixed_align_frames()
vid.complete_align()
vid.create_average_frame()
toc = timeit.default_timer()
vid.write_video('output/output.avi')
vid.write_average_frame('output/lucky_average.png')

print 'Process took {}:'.format(toc-tic)
