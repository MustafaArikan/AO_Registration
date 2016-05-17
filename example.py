import AoRegistration.AoRecording as AoRecording
import timeit
import logging
import argparse

def main(filename,outfile,create=False):
    """
    """
    logging.info('Reading file:%s',filename)
    vid = AoRecording.AoRecording(filepath='data/sample.avi')
    vid.load_video()
    logging.info('Starting parallel processing')
    tic=timeit.default_timer()
    vid.filter_frames()
    vid.fixed_align_frames()
    vid.complete_align_parallel()
    vid.create_average_frame()
    vid.create_stdev_frame()
    toc = timeit.default_timer()
    print 'Parallel Process took {}:'.format(toc-tic)


    vid.create_stdev_frame()

    logging.info('writing output')
    vid.write_video('output/output_parallel.avi')
    vid.write_average_frame('output/lucky_average_parallel.png')
    vid.write_frame('output/lucky_stdev.png','stdev')

    
    logging.info('Starting serial processing')
    tic=timeit.default_timer()
    vid.filter_frames()
    vid.fixed_align_frames()
    vid.complete_align()
    vid.create_average_frame()
    toc = timeit.default_timer()
    print 'Serial Process took {}:'.format(toc-tic)
    
    logging.info('writing output')
    vid.write_video('output/output_serial.avi')
    vid.write_frame('output/lucky_average_serial.png','average')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Register frames from an AO video')
    parser.add_argument('-v','--verbose',help='Increase the amount of output', action='store_true')
    parser.add_argument('-f','--filename',help='full path to file to process', default='data/sample.avi')
    parser.add_argument('-o','--output',help='Full path to a directory to store the output')
    parser.add_argument('-c','--create',help='Create output directory if it doesnt exist',action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
        
    logging.info('started')
    main(filename=args.filename)