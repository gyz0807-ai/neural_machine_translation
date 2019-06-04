import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential

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
        self.encoder = Encoder(flags, enc_vocab_size)
        self.decoder = Decoder(flags, dec_vocab_size)
        self.attention = Attention(flags)
        self.enc_dec_trans = EncDecTransition(flags)
        self.dec_fc_logit = keras.layers.Dense(dec_vocab_size)
        self.loss = keras.losses.SparseCategoricalCrossentropy()

    def call(self, enc_input, dec_input, dec_output_true):
        enc_gru_seq_out, enc_gru_h_state = self.encoder(enc_input)
        enc_gru_h_proj = self.enc_dec_trans(enc_gru_h_state)
        dec_gru_seq_out, dec_gru_h_state = self.decoder(
            dec_input, enc_gru_h_proj)

        attention_vect = self.attention(enc_gru_seq_out, dec_gru_seq_out)
        dec_out = tf.concat([dec_gru_seq_out, attention_vect], 2)
        dec_logit_out = self.dec_fc_logit(dec_out)
        dec_softmax_out = keras.activations.softmax(dec_logit_out)

        loss = self.loss(dec_output_true, dec_softmax_out)

        print('enc_input', enc_input.shape)
        print('dec_input', dec_input.shape)
        print('dec_output_true', dec_output_true.shape)
        print('gru_seq_out', enc_gru_seq_out.shape)
        print('gru_h_state', enc_gru_h_state.shape)
        print('dec_gru_seq_out', dec_gru_seq_out.shape)
        print('dec_gru_h_state', dec_gru_h_state.shape)
        print('attention_vect', attention_vect.shape)
        print('dec_out', dec_out.shape)
        print('dec_logit_out', dec_logit_out.shape)
        print('dec_softmax_out', dec_softmax_out.shape)
        print('loss', loss.shape)

### TRY TO FEED LAST ATTENTION SCORE INSTEAD OF CURRENT
### TRY IF BI-DIRECTIONAL IS BETTER