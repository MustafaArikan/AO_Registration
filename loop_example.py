# -*- coding: utf-8 -*-
import AoRegistration.AoRecording as AO
import logging

def proc_file(fname, idx):
    vid = AO.AoRecording(filepath=fname)
    vid.load_video()
    vid.filter_frames()
    vid.fixed_align_frames()
    vid.complete_align_parallel()
    vid.write_video('output/sample_{}.avi'.format(idx))

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    
    fnames = [(1, 'sample.avi'),(2, 'sample_2.avi')]
    for f in fnames:
        logging.info('Processing: {}'.format(f[1]))
        fpath = 'data/{}'.format(f[1])
        proc_file(fpath, f[0])