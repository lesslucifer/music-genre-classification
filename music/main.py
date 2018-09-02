import numpy as np, scipy, matplotlib.pyplot as plt
import music.preprocess as pp
from music.timer import Timer
import librosa
import tensorflow as tf
import keras
import json
import os
import redis

print(tf.VERSION)
print(keras.__version__)

num_cores = 4
GPU = True
rd = redis.Redis()

if GPU:
    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores, \
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True, \
                            device_count={'CPU': 1, 'GPU': 1})
    session = tf.Session(config=config)
    keras.backend.set_session(session)


def load_file(dir, file):
    idx = int(file.split('.')[0].split('_')[1])
    meta_file = 'meta_%s.json' % idx
    with open('%s/%s' % (dir, meta_file), 'r') as fp: meta = json.load(fp)
    rows = meta['rows']
    data = np.fromfile('%s/%s' % (dir, file))
    x = data.reshape(rows, -1, 1)
    y = np.asarray(meta['labels'])
    return x, y, meta


def train_file(model, dir, file, x_val, y_val):
    nb_epoch = 15
    batch_size = 200

    print('Training file %s' % file)
    Timer.start('Training file %s' % file)
    x_train, y_train, meta = load_file(dir, file)
    # early_stopping_callback = keras.callbacks.EarlyStopping(monitor='acc', min_delta=0.001, patience=200, verbose=1, mode='auto', baseline=None)
    # save_model = keras.callbacks.ModelCheckpoint(model_file, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=10)
    model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, validation_data=(x_val, y_val), verbose=2)
    Timer.end('Training file %s' % file)


def get_model(nfets, size, nclass):
    Timer.start('Building model')
    model = keras.Sequential()
    model.add(keras.layers.BatchNormalization(batch_size=300))
    model.add(keras.layers.Conv1D(64, kernel_size=4 * nfets, strides=nfets, input_shape=(size, 1), activation='elu'))
    model.add(keras.layers.Dropout(0.005))
    model.add(keras.layers.MaxPooling1D(pool_size=6, strides=4))
    model.add(keras.layers.Conv1D(128, kernel_size=10, strides=2, activation='elu'))
    model.add(keras.layers.MaxPooling1D(pool_size=4, strides=2))
    model.add(keras.layers.Conv1D(128, kernel_size=2, strides=2, activation='elu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.Conv1D(128, kernel_size=2, strides=2, activation='elu'))
    model.add(keras.layers.MaxPooling1D(pool_size=2, strides=2))
    model.add(keras.layers.GRU(32, return_sequences=True))
    model.add(keras.layers.GRU(32, return_sequences=False))
    model.add(keras.layers.Dropout(0.001))
    # model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(units=1000, activation='elu'))
    model.add(keras.layers.Dropout(0.005))
    model.add(keras.layers.Dense(units=nclass, activation='softmax'))
    Timer.end('Building model')

    return model


def run():
    ext = 'pdt3'
    run_id = 3
    run_key = '%s:%i' % (ext, run_id)
    dir = '/Users/salm/Desktop/zalo_music/extra_%s' % ext
    model_file = '/Users/salm/Desktop/zalo_music/model_%s.%s.mdl' % (run_id, ext)
    files = list(filter(lambda f: os.path.splitext(f)[1] == '.%s' % ext, os.listdir(dir)))
    val_file = files[len(files) - 1]
    files = files[:len(files) - 1]
    classes = ['cai_luong', 'cach_mang', 'dan_ca', 'dance', 'khong_loi', 'thieu_nhi', 'trinh', 'tru_tinh', 'rap', 'rock']
    x_val, y_val, meta = load_file(dir, val_file)
    nfets = meta['nfets']

    if os.path.isfile(model_file):
        model = keras.models.load_model(model_file)
    else:
        model = get_model(nfets, x_val.shape[1], len(classes))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    nb_runs = 10
    r = int(rd.get('zlm:%s:runs' % run_key) or 0)
    while r < nb_runs:
        Timer.start('Training run %i / %i' % (r + 1, nb_runs))
        i = int(rd.hget('zlm:%s:file' % run_key, r) or 0)
        while i < len(files):
            file = files[i]
            train_file(model, dir, file, x_val, y_val)
            model.save(model_file)
            i = rd.hincrby('zlm:%s:file' % run_key, r)

        r = rd.incr('zlm:%s:runs' % run_key)
        Timer.end('Training run %i / %i' % (r + 1, nb_runs))

if __name__ == '__main__':
    run()
