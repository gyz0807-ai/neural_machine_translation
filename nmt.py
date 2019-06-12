import os
import numpy as np
import tensorflow as tf
from absl import logging
from tensorflow import keras

from lib.preprocessing import VocabHub, en_txt_preproc
from lib.utils import Dataset, create_path, generate_model_dir, FlagsParser

class Attention(keras.Model):
    def __init__(self, flags):
        super(Attention, self).__init__()
        self.W_att = tf.Variable(
            tf.random.truncated_normal(
                [flags.enc_hidden_dim, flags.dec_hidden_dim], seed=flags.seed))

    def call(self, enc_hidden_states, dec_hidden_states):
        dec_hidden_splitted = tf.split(
            dec_hidden_states, dec_hidden_states.shape[1], 1)
        enc_hidden_processed = tf.tensordot(enc_hidden_states, self.W_att, [2, 0])
        att_logits = tf.map_fn(
            lambda dec_h: tf.matmul(enc_hidden_processed, tf.transpose(dec_h, [0, 2, 1])),
            tf.stack(dec_hidden_splitted))
        att_logits_reshaped = tf.transpose(tf.squeeze(att_logits, 3), [1, 0, 2])
        att_softmax = keras.activations.softmax(att_logits_reshaped)
        att_vector = tf.matmul(att_softmax,enc_hidden_states)
        return att_vector

class Encoder(keras.Model):
    def __init__(self, flags, enc_vocab_size, masking_val=0):
        super(Encoder, self).__init__()
        self.flags = flags
        self.enc_vocab_size = enc_vocab_size
        self.masking = keras.layers.Masking(masking_val)
        self.embedding = keras.layers.Embedding(
            self.enc_vocab_size, self.flags.enc_embedding_dim)
        self.gru = keras.layers.GRU(
            flags.enc_hidden_dim, return_sequences=True, return_state=True)

    def call(self, enc_input):
        masking_out = self.masking(enc_input)
        embedding_out = self.embedding(masking_out)
        gru_seq_out, gru_h_state = self.gru(embedding_out)
        return gru_seq_out, gru_h_state

class Decoder(keras.Model):
    def __init__(self, flags, dec_vocab_size, masking_val=0):
        super(Decoder, self).__init__()
        self.flags = flags
        self.dec_vocab_size = dec_vocab_size
        self.masking = keras.layers.Masking(masking_val)
        self.embedding = keras.layers.Embedding(
            self.dec_vocab_size, self.flags.dec_embedding_dim)
        self.gru = keras.layers.GRU(
            self.flags.dec_hidden_dim, return_sequences=True, return_state=True)

    def call(self, dec_input, initial_state):
        masking_out = self.masking(dec_input)
        embedding_out = self.embedding(masking_out)
        gru_seq_out, gru_h_state = self.gru(
            embedding_out, initial_state=initial_state)
        return gru_seq_out, gru_h_state

class EncDecTransition(keras.Model):
    def __init__(self, flags):
        super(EncDecTransition, self).__init__()
        self.enc_h_proj = tf.Variable(tf.random.truncated_normal(
            [flags.enc_hidden_dim, flags.dec_hidden_dim], seed=flags.seed))

    def call(self, enc_gru_h_state):
        enc_gru_h_proj = tf.tensordot(enc_gru_h_state, self.enc_h_proj, [1, 0])
        return enc_gru_h_proj

class NMT(keras.Model):
    def __init__(self, flags, enc_vocab_size, dec_vocab_size):
        super(NMT, self).__init__()
        self.flags = flags
        self.enc_vocab_size = enc_vocab_size
        self.dec_vocab_size = dec_vocab_size
        self.encoder = Encoder(flags, enc_vocab_size)
        self.decoder = Decoder(flags, dec_vocab_size)
        self.attention = Attention(flags)
        self.enc_dec_trans = EncDecTransition(flags)
        self.dec_fc_logit = keras.layers.Dense(dec_vocab_size)

    def call(self, enc_input, dec_input, dec_output_true):
        enc_gru_seq_out, enc_gru_h_state = self.encoder(enc_input)
        enc_gru_h_proj = self.enc_dec_trans(enc_gru_h_state)
        dec_gru_seq_out, _ = self.decoder(
            dec_input, enc_gru_h_proj)

        attention_vect = self.attention(enc_gru_seq_out, dec_gru_seq_out)
        dec_out = tf.concat([dec_gru_seq_out, attention_vect], 2)
        dec_logit_out = self.dec_fc_logit(dec_out)
        return dec_logit_out

# class MaskedLoss(keras.Model):
#     def __init__(self):
#         super(MaskedLoss, self).__init__()
#         self.loss = keras.losses.SparseCategoricalCrossentropy(
#             from_logits=True, reduction='none')

#     def call(self, dec_inputs, preds):
#         mask = tf.math.logical_not(tf.math.equal(dec_inputs, 0))
#         loss_sub = self.loss(dec_inputs, preds)

#         mask = tf.cast(mask, dtype=loss_sub.dtype)
#         loss_sub *= mask
#         return tf.reduce_mean(loss_sub)

class Trainer(object):
    def __init__(self, flags, model):
        self.flags = flags
        self.model = model
        self.loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(flags.learning_rate)

    def train_step(self, enc_input, dec_input, dec_output_true):
        with tf.GradientTape() as tape:
            dec_logit_out = self.model(enc_input, dec_input, dec_output_true)
            batch_loss = self.loss(dec_output_true, dec_logit_out)

        gradients = tape.gradient(batch_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return batch_loss
    
    def eval(self, enc_input, dec_input, dec_output_true):
        dec_logit_out = self.model(enc_input, dec_input, dec_output_true)
        batch_loss = self.loss(dec_output_true, dec_logit_out)
        return batch_loss

    def train(self, train_ds, test_ds):
        """
        train_ds: [enc_train_ds, dec_train_ds]
        val_ds: [enc_val_ds, dec_val_ds]
        test_ds: [enc_test_ds, dec_test_ds]
        """
        root_model_dir = generate_model_dir(self.flags.root_model_dir)
        create_path(root_model_dir)
        checkpoint_dir = os.path.join(root_model_dir, 'training_checkpoints')
        checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
        checkpoint = tf.train.Checkpoint(
            model=self.model)

        # save absl flags
        flags_txt = (
            self.flags.flags_into_string() +
            '--enc_vocab_size={}\n'.format(self.model.enc_vocab_size) +
            '--dec_vocab_size={}\n'.format(self.model.dec_vocab_size))
        flags_file = open(os.path.join(root_model_dir, 'flags.txt'), 'w')
        flags_file.write(flags_txt)
        flags_file.close()

        batch_count = 0
        epoch_count = -1
        batch_manager = Dataset(train_ds[0], train_ds[1], self.flags.num_epochs)
        while batch_manager.epochs_done <= self.flags.num_epochs:
            batch_count += 1
            enc_input, dec_input, dec_output_true = batch_manager.batch(self.flags.batch_size)
            batch_loss = self.train_step(enc_input, dec_input, dec_output_true)

            if epoch_count != batch_manager.epochs_done:
                test_enc_input = keras.preprocessing.sequence.pad_sequences(
                    test_ds[0], padding='post', value=0)
                test_dec_input = keras.preprocessing.sequence.pad_sequences(
                    test_ds[1], padding='post', value=0)
                test_dec_true = np.roll(test_dec_input, -1)
                test_dec_true[:, -1] = 0
                test_loss = self.eval(test_enc_input, test_dec_input, test_dec_true)

                checkpoint.save(file_prefix=checkpoint_prefix)
                epoch_count += 1

            if batch_count % 100 == 0:
                logging.info('Epoch {}, Batch {}, Loss {}, Last test loss: {}'.format(
                    batch_manager.epochs_done, batch_count, batch_loss, test_loss))

def translate(input_sentence, model_path):
    """
    Translate from English to Chinese
    """
    
    flags_file_path = os.path.join(model_path, 'flags.txt')
    flags_parser = FlagsParser(flags_file_path)
    flags_parser.load_flags()

    vocab_hub = VocabHub()
    vocab_hub.load()
    input_processed = en_txt_preproc(input_sentence)
    input_processed = [s.split(' ') for s in input_processed]
    input_processed = [np.array(list(map(lambda x: vocab_hub.en.word_to_idx(x), s))) for s in input_processed]
    input_processed = keras.preprocessing.sequence.pad_sequences(input_processed, padding='post', value=0)

    keras_model = NMT(flags_parser, flags_parser.enc_vocab_size, flags_parser.dec_vocab_size)
    checkpoint = tf.train.Checkpoint(
        model=keras_model)
    latest_ckpt = tf.train.latest_checkpoint(os.path.join(model_path, 'training_checkpoints/'))
    status = checkpoint.restore(latest_ckpt)

    enc_seq_out, enc_h_state = keras_model.encoder(input_processed)
    dec_h_state = keras_model.enc_dec_trans(enc_h_state)
    dec_input = tf.expand_dims([vocab_hub.en.word_to_idx('<START>')], 0)
    end_idx = vocab_hub.en.word_to_idx('<END>')

    final_sentence = [vocab_hub.en.word_to_idx('<START>')]
    pred = dec_input
    while pred != end_idx:
        dec_seq_out, dec_h_state = keras_model.decoder(dec_input, dec_h_state)
        attention_vect = keras_model.attention(enc_seq_out, dec_seq_out)
        dec_out = tf.concat([dec_seq_out, attention_vect], 2)
        dec_logit_out = keras_model.dec_fc_logit(dec_out)
        dec_softmax_out = keras.activations.softmax(dec_logit_out)
        pred = np.argmax(dec_softmax_out.numpy()[0, -1, :])
        dec_input = tf.concat([dec_input, tf.expand_dims([pred], 0)], 1)
        final_sentence.append(pred)
    
    s_full = ''
    s = [vocab_hub.ch.idx_to_word(w) for w in final_sentence]
    for w in s:
        s_full += w
    s_full
    
    return s_full