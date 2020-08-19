import tensorflow as tf
import sys

from data import DataHandler
from models import ACRegNet
from utils import random_pairs, read_config_file, create_dir

RUN_IN_GPU = False


def train_acregnet_model(config):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()

    if RUN_IN_GPU:
        tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)

    # Load training data
    train_ims, _ = DataHandler.load_images(config['train_ims_file'])
    train_lbs, _ = DataHandler.load_labels(config['train_lbs_file'])
    print('Loading training data...done')

    # Create model
    acregnet = ACRegNet(sess, config, 'ACRegNet', is_train=True)
    print('Building AC-RegNet model...done')

    print('Training...')
    # Train model
    for i in range(config['iterations']):
        batch = random_pairs(train_ims, train_lbs, size=config['batch_size'])
        mov_ims, fix_ims, mov_lbs, fix_lbs = batch

        cur_loss = acregnet.fit(mov_ims, fix_ims, mov_lbs, fix_lbs)

        print('Iteration {:>8d}/{}: Loss: {}'.format(
            i + 1, config['iterations'], cur_loss))

    # Save trained model
    acregnet.save(config['ckpt_dir'])
    print('Saving trained AC-RegNet model...done')

    print('Training done')

    tf.reset_default_graph()
    sess.close()


def main(dataset):
    file_path = './config/' + dataset + '/ACRegNet.cfg'
    config = read_config_file(file_path)
    create_dir(config['ckpt_dir'])

    print('Training AC-Regnet | Dataset: {}'.format(dataset))
    print('\nLoading configuration file {}...done'.format(file_path))
    train_acregnet_model(config)


if __name__ == "__main__":
    dataset_list = ['JSRT', 'Montgomery', 'Shenzhen']
    if len(sys.argv) == 2 and sys.argv[1] in dataset_list:
        dataset = sys.argv[1]
        main(dataset)
    else:
        print('Usage: {} [{}]'.format(
            sys.argv[0], ', '.join(map(str, dataset_list))))
        exit(1)
