import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

class Attention(tf.keras.Model):
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
        att_softmax = keras.activations.softmax(att_logits_reshaped) # CHECK IF SUM OF SM OF DIM -1 IS 1
        att_vector = tf.matmul(att_softmax,enc_hidden_states)
        return att_vector

class Encoder(keras.Model):
    def __init__(self, flags, enc_vocab_size):
        super(Encoder, self).__init__()
        self.flags = flags
        self.enc_vocab_size = enc_vocab_size
        self.embedding = keras.layers.Embedding(
            self.enc_vocab_size, self.flags.enc_embedding_dim)
        self.lstm = keras.layers.LSTM(
            flags.enc_hidden_dim, return_sequences=True, return_state=True)

    def call(self, enc_input):
        embedding_out = self.embedding(enc_input)
        lstm_seq_out, lstm_c_state, lstm_h_state = self.lstm(embedding_out)
        return lstm_seq_out, lstm_c_state, lstm_h_state

class Decoder(keras.Model):
    def __init__(self, flags, dec_vocab_size):
        super(Decoder, self).__init__()
        self.flags = flags
        self.dec_vocab_size = dec_vocab_size
        self.embedding = keras.layers.Embedding(
            self.dec_vocab_size, self.flags.dec_embedding_dim)
        self.lstm = keras.layers.LSTM(
            self.flags.dec_hidden_dim, return_sequences=True, return_state=True)

    def call(self, dec_input, initial_state):
        embedding_out = self.embedding(dec_input)
        lstm_seq_out, lstm_c_state, lstm_h_state = self.lstm(
            embedding_out, initial_state=initial_state)
        return lstm_seq_out, lstm_c_state, lstm_h_state

class EncDecTransition(keras.Model):
    def __init__(self, flags):
        super(EncDecTransition, self).__init__()
        self.enc_c_proj = tf.Variable(tf.random.truncated_normal(
            [flags.enc_hidden_dim, flags.dec_hidden_dim], seed=flags.seed))
        self.enc_h_proj = tf.Variable(tf.random.truncated_normal(
            [flags.enc_hidden_dim, flags.dec_hidden_dim], seed=flags.seed))

    def call(self, enc_lstm_c_state, enc_lstm_h_state):
        enc_lstm_c_proj = tf.tensordot(enc_lstm_c_state, self.enc_c_proj, [1, 0])
        enc_lstm_h_proj = tf.tensordot(enc_lstm_h_state, self.enc_h_proj, [1, 0])
        return enc_lstm_c_proj, enc_lstm_h_proj

class NMT(object):

    def __init__(self, flags, enc_vocab_size, dec_vocab_size):
        self.flags = flags
        self.encoder = Encoder(flags, enc_vocab_size)
        self.decoder = Decoder(flags, dec_vocab_size)
        self.attention = Attention(flags)
        self.enc_dec_trans = EncDecTransition(flags)

    def build_graph(self):
        enc_input = keras.layers.Input([10], self.flags.batch_size, 'enc_input', tf.float32)
        dec_input = keras.layers.Input([10], self.flags.batch_size, 'dec_input', tf.float32)

        enc_lstm_seq_out, enc_lstm_c_state, enc_lstm_h_state = self.encoder(enc_input)
        enc_lstm_c_proj, enc_lstm_h_proj = self.enc_dec_trans(enc_lstm_c_state, enc_lstm_h_state)
        dec_init_state = [enc_lstm_c_proj, enc_lstm_h_proj]
        dec_lstm_seq_out, dec_lstm_c_state, dec_lstm_h_state = self.decoder(
            dec_input, dec_init_state)

        attention_vect = self.attention(enc_lstm_seq_out, dec_lstm_seq_out)
        dec_out_logits = tf.concat([dec_lstm_seq_out, attention_vect], 2)

        print('enc_input', enc_input)
        print('lstm_seq_out', enc_lstm_seq_out)
        print('lstm_c_state', enc_lstm_c_state)
        print('lstm_h_state', enc_lstm_h_state)
        print('dec_lstm_seq_out', dec_lstm_seq_out)
        print('dec_lstm_c_state', dec_lstm_c_state)
        print('dec_lstm_h_state', dec_lstm_h_state)
        print('attention_vect', attention_vect)
        print('dec_out_logits', dec_out_logits)
