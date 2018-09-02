from concurrent.futures import ThreadPoolExecutor

import os
import uuid
import librosa as lrs
import numpy as np
from music.preprocess import mkdirp
import redis
import ntpath
from music.timer import Timer

base_sr = 22050
base_len = 30 * base_sr
redis_key = 'zlm:resampled'
rd = redis.Redis()


def resampling(file_path, lbl, out_dir):
    is_processed = rd.sismember(redis_key, file_path)
    if is_processed: return

    y, sr = lrs.load(file_path, sr=base_sr)
    n = (len(y) // base_len)
    if n > 0:
        for i in range(n):
            lrs.output.write_wav('%s/%i.%s.wav' % (out_dir, lbl, uuid.uuid4().hex), y[i*base_len:(i+1)*base_len], sr)

    rd.sadd(redis_key, file_path)


def resamplingdir(dir, lbl, out_dir):
    mkdirp(out_dir)
    files = os.listdir(dir)
    n = len(files)
    exe = ThreadPoolExecutor(max_workers=4)
    for _i, _file in enumerate(files):
        def f(i, file):
            Timer.start('Resampling file %i / %i' % (i + 1, n))
            resampling('%s/%s' % (dir, file), lbl, out_dir)
            Timer.end('Resampling file %i / %i' % (i + 1, n))

        exe.submit(lambda _file=_file, _i=_i: f(_i, _file))

    exe.shutdown(wait=True)


def is_int(s):
    try:
        int(s)
        return True
    except ValueError:
        return False


def count_dir(dir):
    files = os.listdir(dir)
    lbls = [f.split('.')[0] for f in files]
    count = dict([(i, 0) for i in range(10)])
    for lbl in lbls:
        if is_int(lbl): count[int(lbl)] += 1

    print(count)


if __name__ == '__main__':
    out_dir = '/Users/salm/Desktop/zalo_music/extra'
    count_dir(out_dir)

