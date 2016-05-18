import AoRegistration.AoRecording as AoRecording
import timeit
import logging
import argparse
import os
import glob

def main(filename,out_path=None,create=False):
    """
    """

    if out_path is None:
        out_path = os.path.dirname(filename)
        logging.warn('output files will be written to input folder')
    else:
        # check if folder exists
        if not os.path.exists(out_path):
            if create:
                logging.info('Creating output path %s',out_path)
                os.makedirs(out_path)
            else:
                logging.warning('Path for output %s doesnt exist and create flag not set, exiting', out_path)
                exit()
        else:
            logging.info('Output path already exists, files will be overwritten')
        
        
        
    vid = AoRecording.AoRecording(filepath=filename)
    vid.load_video()
    vid.fixInterlace()
    vid.filter_frames()
    vid.fixed_align_frames()
    try:
        vid.complete_align_parallel()
    except RuntimeError:
        logging.warning('Failed to align movie, skipping')
        return
    logging.info('writing stabalised movie')
    vid.write_video(os.path.join(out_path,'stabalised.avi'))

    vid.create_average_frame()
    logging.info('writing mean frame')
    try:
        vid.write_average_frame(os.path.join(out_path,'average.png'))
    except ValueError:
        pass

    vid.create_average_frame(type='lucky')
    logging.info('writing lucky frame')
    try:
        vid.write_average_frame(os.path.join(out_path,'lucky.png'))
    except ValueError:
        pass


    vid.create_stdev_frame()
    logging.info('writing stdev frame')
    try:
        vid.write_frame(os.path.join(out_path,'stdev.png'),'stdev')
    except ValueError:
        pass
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Register frames from an AO video')
    parser.add_argument('-v','--verbose',help='Increase the amount of output', action='store_true')
    parser.add_argument('-f','--basepath',help='full path to file to process', default='data/sample.avi')
    parser.add_argument('-c','--create',help='Create output directory if it doesnt exist',action='store_true')
    args = parser.parse_args()
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARN)
    logging.info('started')
    
    files = glob.glob(os.path.join(args.basepath,'SLO_refl_video_*.avi'))
    for file in files:
        outpath = os.path.join(os.path.dirname(file),
                               'output',
                               os.path.splitext(os.path.basename(file))[0])
        logging.info('Processing file %s', file)
        main(filename = file,
             out_path = outpath,
             create = True)
        
    #main(filename=args.filename)