from data import DataHandler
from models import AENet
import numpy as np
import tensorflow as tf
from utils import get_random_batch, read_config_file, create_dir

 
def test_aenet_model(config):
    tf.reset_default_graph()
    sess = tf.Session()

    test_lbs, _ = DataHandler.load_labels(config['test_lbs_file'])
    print('Loading test data...done')

    config['batch_size'] = test_lbs.shape[0]

    aenet = AENet(sess, config, 'AENet', is_train=False)
    print('Building AENet model...done')
    aenet.restore(config['ckpt_dir'])
    print('Loading trained AENet model...done')

    print('Testing...')
    aenet.deploy(config['result_dir'], test_lbs)
    
    print('Testing...done')


if __name__ == "__main__":
    config = read_config_file('./config/JSRT/AENet.cfg')
    create_dir(config['result_dir'])
    test_aenet_model(config)