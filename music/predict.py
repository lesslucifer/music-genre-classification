from concurrent.futures import ThreadPoolExecutor

import os
import librosa as lrs
import json
from music.preprocess import dist, make_file_data, hop, nfft, win_len, mkdirp
from music.resampling import base_len, base_sr
from music.timer import Timer
import tensorflow as tf
import keras
import numpy as np

music_exts = ['.mp3', '.wav']
stride = 10 * base_sr

def make_test_data(dir, out_dir, ext):
    mkdirp(out_dir)
    files = list(filter(lambda f: os.path.splitext(f)[1] in music_exts, os.listdir(dir)))
    packs = dist(files, len(files) // 300)
    n_pack = len(packs)
    for pi, pack in enumerate(packs):
        out_file = '%s/data_%i.%s' % (out_dir, pi, ext)
        if os.path.isfile(out_file): continue
        Timer.start('Make test data pack %i / %i' % (pi + 1, n_pack))
        exe = ThreadPoolExecutor(max_workers=4)
        file_names = []
        data = []
        for _i, _file in enumerate(pack):
            def f(i, file):
                Timer.start('Make test data for file: %i / %i' % (i + 1, len(pack)))
                y, sr = lrs.load('%s/%s' % (dir, file), sr=base_sr)
                n = 1 + ((len(y) - base_len) // stride)
                for j in range(n):
                    start = j * stride
                    end = start + base_len
                    chunk_data = make_file_data(y[start:end], sr, hop, nfft, win_len)
                    data.append(chunk_data)
                    file_names.append(file)

                Timer.end('Make test data for file: %i / %i' %(i + 1, len(pack)))

            exe.submit(lambda _file=_file, _i=_i: f(_i, _file))

        exe.shutdown(wait=True)
        np.asarray(data).tofile(out_file)
        with open('%s/meta_%i.json' % (out_dir, pi), 'w') as fp: json.dump(file_names, fp)

        Timer.end('Make test data pack %i / %i' % (pi + 1, n_pack))


def predict(model, dir, ext):
    files = list(filter(lambda f: os.path.splitext(f)[1] == '.%s' % ext, os.listdir(dir)))
    results = []
    Timer.start('Prediction')
    for file in files:
        Timer.start('Predicting file %s' % file)
        idx = int(file.split('.')[0].split('_')[1])
        meta_file = 'meta_%s.json' % idx
        with open('%s/%s' % (dir, meta_file), 'r') as fp: file_names = json.load(fp)
        x_predict = np.fromfile('%s/%s' % (dir, file))
        x_predict = x_predict.reshape(len(file_names), -1, 1)
        predictions = model.predict(x_predict)
        for file_name, prediction in zip(file_names, predictions):
            results.append({
                'file': file_name,
                'predict': prediction.tolist()
            })
        Timer.end('Predicting file %s' % file)

    Timer.end('Prediction')
    return results


def run():
    test_dir = '/Users/salm/Desktop/zalo_music/test'
    ext = 'pdt3'
    run_id = 1
    out_dir = '%s_%s_%i' % (test_dir, ext, run_id)
    make_test_data(test_dir, out_dir, ext)
    model_file = '/Users/salm/Desktop/zalo_music/model_3.pdt3.mdl'
    model = keras.models.load_model(model_file)
    predictions = predict(model, out_dir, ext)
    with open('%s/predictions.json' % out_dir, 'w') as fp: json.dump(predictions, fp)


if __name__ == '__main__':
    run()
