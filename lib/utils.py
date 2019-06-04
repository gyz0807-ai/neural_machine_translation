import os
import numpy as np
from tensorflow import keras

def create_path(path):
    if not os.path.exists(path):
        os.makedirs(path)

class Dataset(object):
    def __init__(self, encoder_ds, decoder_ds, num_epochs):
        self.encoder_ds = encoder_ds
        self.decoder_ds = decoder_ds
        self.num_epochs = num_epochs
        self.epochs_done = 0
        self.current_idx = 0

    def batch(self, batch_size):
        if self.epochs_done <= self.num_epochs:
            enc_batch = self.encoder_ds[self.current_idx:(self.current_idx+batch_size)]
            dec_batch = self.decoder_ds[self.current_idx:(self.current_idx+batch_size)]
            self.current_idx += batch_size

            if len(enc_batch) < batch_size:
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
