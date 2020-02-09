from data import DataHandler
from models import AENet
import tensorflow as tf
from utils import get_random_batch, read_config_file, create_dir


RUN_IN_GPU = False


def train_aenet_model(config):
    tf.reset_default_graph()
    tf_config = tf.ConfigProto()

    if RUN_IN_GPU:
        tf_config.gpu_options.allow_growth = True

    sess = tf.Session(config=tf_config)

    train_lbs, _ = DataHandler.load_labels(config['train_lbs_file'])
    print('Loading training data...done')

    aenet = AENet(sess, config, 'AENet', is_train=True)
    print('Building AENet model...done')

    print('Training...')
    for i in range(config['iterations']):
        batch_lbs, _ = get_random_batch(train_lbs, config['batch_size'])
        cur_loss = aenet.fit(batch_lbs)
        print('Iteration {:>8d}/{}: Loss: {}'.format(
            i + 1, config['iterations'], cur_loss))

    aenet.save(config['ckpt_dir'])
    print('Saving current AENet model...done')

    print('Training...done')

    tf.reset_default_graph()
    sess.close()


if __name__ == "__main__":
    config = read_config_file('./config/JSRT/AENet.cfg')
    create_dir(config['ckpt_dir'])
    train_aenet_model(config)
