import bs4
import pickle
import numpy as np
from absl import app
from absl import flags
from absl import logging
from sklearn.model_selection import train_test_split

from nmt import NMT, Trainer
from lib.preprocessing import VocabHub, read_tmx, VocabDataset, split_enc_dec_ds

FLAGS = flags.FLAGS
flags.DEFINE_integer('reparse_vocab', 1, 'Whether construct a new vocab dict')
flags.DEFINE_float('learning_rate', 1e-3, 'Learning rate')
flags.DEFINE_integer('num_epochs', 30, 'Number of training epochs')
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_integer('enc_hidden_dim', 1024, 'Size of encoder hidden layers')
flags.DEFINE_integer('dec_hidden_dim', 1024, 'Size of decoder hidden layers')
flags.DEFINE_integer('enc_embedding_dim', 256, 'Dimension of encoder embeddings')
flags.DEFINE_integer('dec_embedding_dim', 256, 'Dimension of decoder embeddings')
flags.DEFINE_integer('is_en_to_ch', 1, 'en->ch or ch->en')
flags.DEFINE_integer('seed', 666, 'Seed for operations involving randomness')
flags.DEFINE_string(
    'data_file_path', './dataset/en-zh.tmx', '(Optional) Parallel text directory')
flags.DEFINE_string(
    'root_model_dir', './trained_model', 'Root model directory')

def main(argv):

    if FLAGS.reparse_vocab:
        logging.info('Loading data...')
        en_full, ch_full = read_tmx(FLAGS.data_file_path)

    logging.info('Building/Loading vocab...')
    vocab_hub = VocabHub()
    if FLAGS.reparse_vocab:
        vocab_hub.build(en_full, ch_full)
        vocab_hub.save()
    else:
        vocab_hub.load()

    ch_num_vocab = len(vocab_hub.ch.word_dict)
    en_num_vocab = len(vocab_hub.en.word_dict)
    enc_vocab_size = en_num_vocab if FLAGS.is_en_to_ch else ch_num_vocab
    dec_vocab_size = ch_num_vocab if FLAGS.is_en_to_ch else en_num_vocab
    enc_type = 'English' if FLAGS.is_en_to_ch else 'Chinese'
    dec_type = 'Chinese' if FLAGS.is_en_to_ch else 'English'
    logging.info('({})Encoder vocab size: {}'.format(enc_type, enc_vocab_size))
    logging.info('({})Decoder vocab size: {}'.format(dec_type, dec_vocab_size))

    logging.info('Preprocessing datasets...')
    dataset_generator = VocabDataset()
    if FLAGS.reparse_vocab:
        dataset_generator.build(en_full, ch_full, vocab_hub)
        dataset_generator.save()
    else:
        dataset_generator.load()

    logging.info('Performing train/test/val split...')
    if FLAGS.is_en_to_ch:
        enc_processed, dec_processed = (
            dataset_generator.en_processed,
            dataset_generator.ch_processed)
    else:
        enc_processed, dec_processed = (
            dataset_generator.ch_processed,
            dataset_generator.en_processed)
    enc_dec_ds = list(zip(enc_processed, dec_processed))
    train_ds, test_ds = train_test_split(
        enc_dec_ds, test_size=0.003, random_state=FLAGS.seed)
    enc_train_ds, dec_train_ds = split_enc_dec_ds(train_ds)
    enc_test_ds, dec_test_ds = split_enc_dec_ds(test_ds)
    logging.info('Number of train obs: {}'.format(len(enc_train_ds)))
    logging.info('Number of test obs: {}'.format(len(enc_test_ds)))

    logging.info('Training neural network model...')
    nmt_model = NMT(FLAGS, enc_vocab_size, dec_vocab_size)
    keras_trainer = Trainer(FLAGS, nmt_model)
    keras_trainer.train(
        [enc_train_ds, dec_train_ds],
        [enc_test_ds, dec_test_ds])

if __name__ == '__main__':
    app.run(main)
