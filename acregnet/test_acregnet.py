from data import DataHandler
from models import ACRegNet
import tensorflow as tf
from utils import get_random_batch, read_config_file, create_dir


def test_acregnet_model(config):
    tf.reset_default_graph()
    sess = tf.Session()

    test_ims, _ = DataHandler.load_images(config['test_ims_file'])
    print('Loading test data...done')

    config['batch_size'] = test_ims.shape[0] * 2
    config['image_size'] = [256, 256]

    acregnet = ACRegNet(sess, config, 'ACRegNet', is_train=False)
    print('Building AC-RegNet model...done')
    acregnet.restore(config['ckpt_dir'])
    print('Loading trained AC-RegNet model...done')

    batch_ims_x, batch_ims_y = get_random_batch(test_ims, config['batch_size'])

    print('Testing...')
    acregnet.deploy(config['result_dir'], batch_ims_x, batch_ims_y, True)

    print('Testing...done')


if __name__ == "__main__":
    config = read_config_file('./config/JSRT/ACRegNet.cfg')
    create_dir(config['result_dir'])
    test_acregnet_model(config)
