from concurrent.futures import ThreadPoolExecutor
import json
import numpy as np
import pathlib
import librosa
import os
import hashlib
from music.timer import Timer

audio_exts = ['.mp3', '.wav']
hop = 512
nfft = 512
win_len = 512

def mkdirp(dir):
    pathlib.Path(dir).mkdir(parents=True, exist_ok=True)


def dist(files, n):
    packs = [[] for i in range(n)]
    # print(files)
    for f in files:
        h = int(hashlib.md5(f.encode('ascii')).hexdigest(), 16)
        hi = h % n
        packs[hi].append(f)

    return packs


def make_file_data(y, sr, hop, nfft, win_len):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr, hop_length=hop)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, n_fft=nfft)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=win_len, hop_length=hop)
    onsetLog = np.log1p(onset_env)
    ac = librosa.autocorrelate(onsetLog)
    mfcc = librosa.feature.mfcc(y, sr, n_mfcc=13)

    data = np.vstack([chroma, spectral_contrast, mfcc, onset_env, zcr, ac]).transpose(1, 0)
    return data


def extract_features(signal, sample_rate, frame_size, hop_size):
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=signal, frame_length=frame_size, hop_length=hop_size)
    spectral_centroid = librosa.feature.spectral_centroid(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_contrast = librosa.feature.spectral_contrast(y=signal, sr=sample_rate, n_fft=frame_size,
                                                          hop_length=hop_size)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=signal, sr=sample_rate, n_fft=frame_size,
                                                            hop_length=hop_size)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)
    mfccs = librosa.feature.mfcc(y=signal, sr=sample_rate, n_fft=frame_size, hop_length=hop_size)

    return [

        np.mean(zero_crossing_rate),
        np.std(zero_crossing_rate),
        np.mean(spectral_centroid),
        np.std(spectral_centroid),
        np.mean(spectral_contrast),
        np.std(spectral_contrast),
        np.mean(spectral_bandwidth),
        np.std(spectral_bandwidth),
        np.mean(spectral_rolloff),
        np.std(spectral_rolloff),

        np.mean(mfccs[1, :]),
        np.std(mfccs[1, :]),
        np.mean(mfccs[2, :]),
        np.std(mfccs[2, :]),
        np.mean(mfccs[3, :]),
        np.std(mfccs[3, :]),
        np.mean(mfccs[4, :]),
        np.std(mfccs[4, :]),
        np.mean(mfccs[5, :]),
        np.std(mfccs[5, :]),
        np.mean(mfccs[6, :]),
        np.std(mfccs[6, :]),
        np.mean(mfccs[7, :]),
        np.std(mfccs[7, :]),
        np.mean(mfccs[8, :]),
        np.std(mfccs[8, :]),
        np.mean(mfccs[9, :]),
        np.std(mfccs[9, :]),
        np.mean(mfccs[10, :]),
        np.std(mfccs[10, :]),
        np.mean(mfccs[11, :]),
        np.std(mfccs[11, :]),
        np.mean(mfccs[12, :]),
        np.std(mfccs[12, :]),
        np.mean(mfccs[13, :]),
        np.std(mfccs[13, :]),
    ]


def make_data(ext, in_dir, out_dir, n_pack):
    mkdirp(out_dir)
    allfiles = os.listdir(in_dir)
    packs = dist(allfiles, n_pack)
    print([len(p) for p in packs])
    for pi, files in enumerate(packs):
        Timer.start('Making data for pack %i / %i' % (pi + 1, len(packs)))
        out_file = '%s/data_%i.%s' % (out_dir, pi, ext)
        if os.path.exists(out_file): continue
        exe = ThreadPoolExecutor(max_workers=16)
        records = {}
        meta_dict = {
            'hop': hop,
            'nfft': nfft,
            'win_len': win_len,
            'records': records,
            'labels': []
        }
        n = len(files)
        pack_data = []
        for idx, _file in enumerate(files):
            def f(full_file, i):
                file_with_lbl, file_ext = os.path.splitext(full_file)
                if file_ext not in audio_exts: return

                lbl, file = os.path.splitext(file_with_lbl)
                lbl = int(lbl)
                file = file[1:]

                Timer.start('Making data for %s(%i) (%i/%i)' % (file, pi + 1, i + 1, n))
                meta = {'file': file, 'label': lbl}

                y, sr = librosa.load('%s/%i.%s%s' % (in_dir, lbl, file, file_ext))
                meta['sr'] = sr
                # data = make_file_data(y, sr, hop, nfft, win_len)
                data = extract_features(y, sr, nfft, hop)
                pack_data.append(data)
                records[file] = meta
                meta_dict['labels'].append(lbl)
                Timer.end('Making data for %s(%i) (%i/%i)' % (file, pi + 1, i + 1, n))

            exe.submit(lambda _file=_file,idx=idx: f(_file, idx))

        exe.shutdown(wait=True)

        nppack = np.asarray(pack_data)
        print(nppack.shape)
        meta_dict['rows'] = len(meta_dict['labels'])
        meta_dict['nfets'] = nppack.shape[2]
        with open('%s/meta_%i.json' % (out_dir, pi), 'w') as fp:
            json.dump(meta_dict, fp)

        nppack.tofile(out_file)
        Timer.end('Making data for pack %i / %i' % (pi + 1, len(packs)))


def rename_mp3(csv, dir):
    rows = np.genfromtxt(csv, delimiter=',', dtype=None)
    files = [r[0][:-4].decode('ascii') for r in rows]
    labels = [int(r[1]) - 1 for r in rows]

    for file, lbl in zip(files, labels):
        path = '%s/%s.mp3' % (dir, file)
        if os.path.isfile(path):
            os.rename(path, '%s/%s.%s.mp3' % (dir, lbl, file))

if __name__ == '__main__':
    train_dir = '/Users/salm/Desktop/zalo_music/extra'
    ext = 'pdt4'
    make_data(ext, train_dir, '%s_%s' % (train_dir, ext), 10)
