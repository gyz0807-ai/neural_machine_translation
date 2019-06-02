import bs4
from absl import app
from absl import flags
from absl import logging

from nmt import NMT
from lib.preprocessing import VocabHub, read_tmx, create_dataset

FLAGS = flags.FLAGS
flags.DEFINE_integer('reparse_vocab', 0, 'Whether construct a new vocab dict')
flags.DEFINE_float('learning_rate', 1e-4, 'Learning rate')
flags.DEFINE_integer('batch_size', 64, 'Size of training batches')
flags.DEFINE_integer('enc_hidden_dim', 50, 'Size of encoder hidden layers')
flags.DEFINE_integer('dec_hidden_dim', 60, 'Size of decoder hidden layers')
flags.DEFINE_integer('enc_embedding_dim', 300, 'Dimension of encoder embeddings')
flags.DEFINE_integer('dec_embedding_dim', 300, 'Dimension of decoder embeddings')
flags.DEFINE_string(
    'data_file_path', './dataset/en-zh.tmx', '(Optional) Parallel text directory')
flags.DEFINE_integer('is_en_to_ch', 1, 'en->ch or ch->en')
flags.DEFINE_integer('seed', 666, 'Seed for operations involving randomness')

def main(argv):

    # logging.info('Loading data...')
    # en_full, ch_full = read_tmx(FLAGS.data_file_path)

    logging.info('Building/Loading vocab...')
    if FLAGS.reparse_vocab:
        vocab_hub = VocabHub()
        vocab_hub.build(en_full, ch_full)
        vocab_hub.save()
    else:
        vocab_hub = VocabHub()
        vocab_hub.load()

    ch_num_vocab = len(vocab_hub.ch.word_dict)
    en_num_vocab = len(vocab_hub.en.word_dict)
    enc_vocab_size = en_num_vocab if FLAGS.is_en_to_ch else ch_num_vocab
    dec_vocab_size = ch_num_vocab if FLAGS.is_en_to_ch else en_num_vocab
    # ch_type = 'Decoder' if FLAGS.is_en_to_ch else 'Encoder'
    # en_type = 'Encoder' if FLAGS.is_en_to_ch else 'Decoder'
    # logging.info('({})Chinese vocab size: {}'.format(ch_type, ch_num_vocab))
    # logging.info('({})English vocab size: {}'.format(en_type, en_num_vocab))

    # logging.info('Preprocessing datasets...')
    # en_processed, ch_processed = create_dataset(en_full, ch_full, vocab_hub)

    logging.info('Training neural network model...')
    nmt_model = NMT(FLAGS, enc_vocab_size, dec_vocab_size)
    nmt_model.build_graph()

if __name__ == '__main__':
    app.run(main)
