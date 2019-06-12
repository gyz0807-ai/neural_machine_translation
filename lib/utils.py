import os
import re
import numpy as np
from tensorflow import keras
from datetime import datetime
from sklearn.utils import shuffle

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_model_dir(root_model_dir):
    datetime_now = datetime.utcnow().strftime('%Y%m%d%H%M%S')
    model_dir = os.path.join(root_model_dir, 'model-{}/'.format(datetime_now))
    return model_dir

class Dataset(object):
    def __init__(self, encoder_ds, decoder_ds, num_epochs, shuffle=True):
        self.encoder_ds = encoder_ds
        self.decoder_ds = decoder_ds
        self.num_epochs = num_epochs
        self.shuffle = shuffle
        self.epochs_done = 0
        self.current_idx = 0

    def batch(self, batch_size):
        if self.epochs_done <= self.num_epochs:
            enc_batch = self.encoder_ds[self.current_idx:(self.current_idx+batch_size)]
            dec_batch = self.decoder_ds[self.current_idx:(self.current_idx+batch_size)]
            self.current_idx += batch_size

            if len(enc_batch) < batch_size:
                self.encoder_ds, self.decoder_ds = shuffle(self.encoder_ds, self.decoder_ds)
                self.current_idx = batch_size - len(enc_batch)
                enc_batch += self.encoder_ds[0:self.current_idx]
                dec_batch += self.decoder_ds[0:self.current_idx]
                self.epochs_done += 1
        else:
            enc_batch, dec_batch = [], []

        enc_batch = keras.preprocessing.sequence.pad_sequences(enc_batch, padding='post', value=0)
        dec_batch = keras.preprocessing.sequence.pad_sequences(dec_batch, padding='post', value=0)
        dec_out_batch = np.roll(dec_batch, -1)
        dec_out_batch[:, -1] = 0
        return enc_batch, dec_batch, dec_out_batch

class FlagsParser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.flags = {}

    def parse_flags(self):
        with open(self.file_path, 'r') as f:
            flags_txt = f.readlines()
        flags_txt_processed = [
            re.sub('\n|-', '', flag).split('=') for flag in flags_txt]
        return flags_txt_processed

    def load_flags(self):
        flags_parsed = self.parse_flags()
        for flag in flags_parsed:
            if len(flag) == 1:
                setattr(self, '{}'.format(flag[0]), True)
            else:
                try:
                    flag_1 = eval(flag[1])
                except (NameError, SyntaxError):
                    flag_1 = flag[1]
                setattr(self, '{}'.format(flag[0]), flag_1)
