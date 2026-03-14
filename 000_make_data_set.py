import os
import glob
import random
import pickle
import configs
from tqdm import tqdm

def create_PLUSVein_annotation(args):
    def iter(root, Where):
        sub2classes = {}
        train_samples, test_samples = [], []
        data_path = os.path.join(root, Where)

        for folder in tqdm(os.listdir(data_path)):
            if not os.path.isdir(os.path.join(data_path, folder)):
                continue
            paths = glob.glob(os.path.join(data_path, folder, '*.png'))
            for idx in ['02', '03', '04', '07', '08', '09']:
                identity = f'{folder}_{idx}'
                filter_paths = [path for path in paths if identity in path]
                random.shuffle(filter_paths)

                train, test = args.split.split(':')
                ratio = int(test) / (int(train) + int(test))
                bps = int(len(filter_paths) * ratio)
                for_test = filter_paths[:bps]

                if not identity in sub2classes:
                    sub2classes[identity] = len(sub2classes)

                for path in filter_paths:
                    if path in for_test:
                        test_samples.append({'path': path, 'label': sub2classes[identity]})
                    else:
                        train_samples.append({'path': path, 'label': sub2classes[identity]})
        return train_samples, test_samples

    train_samples_LED, test_samples_LED = iter(args.data_root, os.path.join('PLUS-FV3-LED', 'PALMAR', '01'))
    train_samples_LASER, test_samples_LASER = iter(args.data_root, os.path.join('PLUS-FV3-Laser', 'PALMAR', '01'))
    pickle.dump({
        'LED': {
            'train_set': train_samples_LED,
            'test_set': test_samples_LED,
        },
        'LASER': {
            'train_set': train_samples_LASER,
            'test_set': test_samples_LASER,
        }
    }, open(args.annot_file, 'wb'))
    print(f'train_samples: {len(train_samples_LED)}')
    print(f'test_samples: {len(test_samples_LED)}')
    print(test_samples_LED[0])

if __name__ == '__main__':
    args = configs.get_all_params()
    configs.setup_seed(args.seed)

    args.datasets = 'PLUSVein-FV3'
    args = configs.get_dataset_params(args)
    create_PLUSVein_annotation(args)

