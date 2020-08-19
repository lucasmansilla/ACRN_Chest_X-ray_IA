import tensorflow as tf
import numpy as np
import pickle
import sys

from data import DataHandler
from models import ACRegNet
from simple_warper import SimpleWarper
from utils import random_pairs, read_config_file, create_dir
from metrics import dc, hd, assd


def test_acregnet_model(config):
    tf.reset_default_graph()
    sess = tf.Session()

    out_im_dir = config['result_dir'] + '/images'
    create_dir(out_im_dir)

    # Load test data
    test_ims, _ = DataHandler.load_images(config['test_ims_file'])
    test_lbs, _ = DataHandler.load_labels(config['test_lbs_file'])
    print('Loading test data...done')

    config['batch_size'] = test_ims.shape[0] * 2
    config['image_size'] = [256, 256]

    # Load trained model
    acregnet = ACRegNet(sess, config, 'ACRegNet', is_train=False)
    print('Building AC-RegNet model...done')
    acregnet.restore(config['ckpt_dir'])
    print('Loading trained AC-RegNet model...done')

    data = random_pairs(test_ims, test_lbs, size=config['batch_size'])
    mov_ims, fix_ims, mov_lbs, fix_lbs = data

    print('Testing...')
    # Get deformation fields
    flow = acregnet.deploy(out_im_dir, mov_ims, fix_ims, True)[1]

    # Warp label maps
    sw = SimpleWarper(sess, config['batch_size'], config['image_size'],
                      config['n_labels'])
    warp_lbs = sw.warp_label(mov_lbs, flow)

    # Compute metrics
    metrics = {'dc': [], 'hd': [], 'assd': []}
    spacing = config['std_res'] / config['image_size'][0]
    for warp_lb, fix_lb in zip(warp_lbs, fix_lbs):
        metrics['dc'].append(dc(warp_lb, fix_lb))
        metrics['hd'].append(hd(warp_lb, fix_lb, pixel_spacing=spacing))
        metrics['assd'].append(assd(warp_lb, fix_lb, pixel_spacing=spacing))

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
    file_path = './config/' + dataset + '/ACRegNet.cfg'
    config = read_config_file(file_path)
    create_dir(config['ckpt_dir'])

    print('Test AC-RegNet | Dataset: {}'.format(dataset))
    print('\nLoading configuration file {}...done'.format(file_path))
    test_acregnet_model(config)


if __name__ == "__main__":
    dataset_list = ['JSRT', 'Montgomery', 'Shenzhen']
    if len(sys.argv) == 2 and sys.argv[1] in dataset_list:
        dataset = sys.argv[1]
        main(dataset)
    else:
        print('Usage: {} [{}]'.format(
            sys.argv[0], ', '.join(map(str, dataset_list))))
        exit(1)
