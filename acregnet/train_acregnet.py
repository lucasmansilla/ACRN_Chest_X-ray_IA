from data import DataHandler
from models import ACRegNet
import tensorflow as tf
from utils import get_random_batch, read_config_file, create_dir
import numpy as np


RUN_IN_GPU = False

def train_acregnet_model(config):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()
    
    if RUN_IN_GPU:
        tf_config.gpu_options.allow_growth = True
    
    sess = tf.Session(config=tf_config)

    train_ims, _ = DataHandler.load_images(config['train_ims_file'])
    train_lbs, _ = DataHandler.load_labels(config['train_lbs_file'])
    print('Loading training data...done')
     
    acregnet = ACRegNet(sess, config, 'ACRegNet', is_train=True)
    print('Building AC-RegNet model...done')
    
    print('Training...')
    for i in range(config['iterations']):
        batch_ims_x, batch_ims_y, batch_lbs_x, batch_lbs_y = get_random_batch(train_ims, 
                                                                              config['batch_size'],
                                                                              train_lbs)
        cur_loss = acregnet.fit(batch_ims_x, batch_ims_y, batch_lbs_x, batch_lbs_y)
        print('Iteration {:>8d}/{}: Loss: {}'.format(i + 1, config['iterations'], cur_loss))
    
    acregnet.save(config['ckpt_dir'])
    print('Saving current AC-RegNet model...done')

    print('Training...done')
    
    tf.reset_default_graph()
    sess.close()


if __name__ == "__main__":
    config = read_config_file('./config/JSRT/ACRegNet.cfg')
    create_dir(config['ckpt_dir']) 
    train_acregnet_model(config)

