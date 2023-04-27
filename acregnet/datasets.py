from torch.utils.data import Dataset

from .utils.io import read_txt, read_image
from .utils.tensor import to_tensor, to_one_hot, swap_labels, relabel
from .utils.misc import get_pairs, get_image_info


class ImagePairsDataset(Dataset):

    def __init__(self, images_file_path, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        images_files = read_txt(images_file_path)
        labels_files = read_txt(labels_file_path)

        mov_images_files, fix_images_files = get_pairs(images_files)
        mov_labels_files, fix_labels_files = get_pairs(labels_files)

        self.images_files = {
            'mov': mov_images_files,
            'fix': fix_images_files
        }

        self.labels_files = {
            'mov': mov_labels_files,
            'fix': fix_labels_files
        }

        self.input_size, self.num_labels = get_image_info(labels_files[0], is_label=True)

    def __len__(self):
        return len(self.images_files['mov'])

    def __getitem__(self, index):

        mov_image_path = self.images_files['mov'][index]
        fix_image_path = self.images_files['fix'][index]
        mov_label_path = self.labels_files['mov'][index]
        fix_label_path = self.labels_files['fix'][index]

        mov_image = to_tensor(read_image(mov_image_path)) / 255.
        fix_image = to_tensor(read_image(fix_image_path)) / 255.
        mov_label = to_tensor(read_image(mov_label_path))
        fix_label = to_tensor(read_image(fix_label_path))

        mov_label = relabel(mov_label)
        fix_label = relabel(fix_label)

        if self.mode == 'train':
            mov_label = to_one_hot(mov_label)
            fix_label = to_one_hot(fix_label)

        return (mov_image, fix_image), (mov_label, fix_label)


class LabelsDataset(Dataset):

    def __init__(self, labels_file_path, mode='train'):
        assert mode in ['train', 'test']
        self.mode = mode

        self.labels_files = read_txt(labels_file_path)
        self.input_size, self.num_labels = get_image_info(self.labels_files[0], is_label=True)

    def __len__(self):
        return len(self.labels_files)

    def __getitem__(self, index):

        target_path = self.labels_files[index]
        target_label = to_tensor(read_image(target_path))
        target_label = relabel(target_label)

        if self.mode == 'train':
            input_label = swap_labels(target_label, p=0.1)

            input_label = to_one_hot(input_label)
            target_label = to_one_hot(target_label)

            return input_label, target_label

        target_label = to_one_hot(target_label)

        return target_label
