import tensorflow as tf
import numpy as np
import sys

from data import DataHandler
from models import AENet
from utils import read_config_file, create_dir

RUN_IN_GPU = False


def train_aenet_model(config):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()

    if RUN_IN_GPU:
        tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)

    # Load training data
    train_lbs, _ = DataHandler.load_labels(config['train_lbs_file'])
    print('Loading training data...done')

    # Create model
    aenet = AENet(sess, config, 'AENet', is_train=True)
    print('Building AE-Net model...done')

    print('Training...')
    # Train model
    n_data = len(train_lbs)
    for i in range(config['iterations']):
        idxs = np.random.choice(n_data, config['batch_size'], replace=False)

        cur_loss = aenet.fit(train_lbs[idxs])

        print('Iteration {:>8d}/{}: Loss: {}'.format(
            i + 1, config['iterations'], cur_loss))

    # Save trained model
    aenet.save(config['ckpt_dir'])
    print('Saving trained AE-Net model...done')

    print('Training done')

    tf.reset_default_graph()
    sess.close()


def main(dataset):
    file_path = './config/' + dataset + '/AENet.cfg'
    config = read_config_file(file_path)
    create_dir(config['ckpt_dir'])

    print('Training AE-Net | Dataset: {}'.format(dataset))
    print('\nLoading configuration file {}...done'.format(file_path))
    train_aenet_model(config)


if __name__ == "__main__":
    dataset_list = ['JSRT', 'Montgomery', 'Shenzhen']
    if len(sys.argv) == 2 and sys.argv[1] in dataset_list:
        dataset = sys.argv[1]
        main(dataset)
    else:
        print('Usage: {} [{}]'.format(
            sys.argv[0], ', '.join(map(str, dataset_list))))
        exit(1)
