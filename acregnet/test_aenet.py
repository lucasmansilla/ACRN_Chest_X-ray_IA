import tensorflow as tf
import numpy as np
import pickle
import sys

from data import DataHandler
from models import AENet
from utils import read_config_file, create_dir
from metrics import dc, hd, assd


def test_aenet_model(config):
    tf.reset_default_graph()
    sess = tf.Session()

    out_lb_dir = config['result_dir'] + '/labels'
    create_dir(out_lb_dir)

    # Load test data
    test_lbs, _ = DataHandler.load_labels(config['test_lbs_file'])
    print('Loading test data...done')

    config['batch_size'] = test_lbs.shape[0]

    # Load trained model
    aenet = AENet(sess, config, 'AENet', is_train=False)
    print('Building AE-Net model...done')
    aenet.restore(config['ckpt_dir'])
    print('Loading trained AE-Net model...done')

    print('Testing...')
    # Get reconstructed label maps
    rec_lbs = aenet.deploy(out_lb_dir, test_lbs)

    # Compute metrics
    metrics = {'dc': [], 'hd': [], 'assd': []}
    spacing = config['std_res'] / config['image_size'][0]
    for rec_lb, fix_lb in zip(rec_lbs, test_lbs):
        metrics['dc'].append(dc(rec_lb, fix_lb))
        metrics['hd'].append(hd(rec_lb, fix_lb, pixel_spacing=spacing))
        metrics['assd'].append(assd(rec_lb, fix_lb, pixel_spacing=spacing))

    # Save results to disk
    with open(config['result_dir'] + '/metrics.pkl', 'wb') as f:
        pickle.dump(metrics, f)

    # Report metric values
    print('Metrics:')
    for name, values in metrics.items():
        print('- {}: mean {:.3f}, std {:.3f}'.format(
            name, np.mean(values), np.std(values)))

    print('Testing done')


def main(dataset):
    file_path = './config/' + dataset + '/AENet.cfg'
    config = read_config_file(file_path)
    create_dir(config['ckpt_dir'])

    print('Test AE-Net | Dataset: {}'.format(dataset))
    print('\nLoading configuration file {}...done'.format(file_path))
    test_aenet_model(config)


if __name__ == "__main__":
    dataset_list = ['JSRT', 'Montgomery', 'Shenzhen']
    if len(sys.argv) == 2 and sys.argv[1] in dataset_list:
        dataset = sys.argv[1]
        main(dataset)
    else:
        print('Usage: {} [{}]'.format(
            sys.argv[0], ', '.join(map(str, dataset_list))))
        exit(1)
