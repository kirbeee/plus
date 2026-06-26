# Pipeline: collect data -> split -> build sample -> save
import os
import glob
import random
from collections import defaultdict
import configs
from tqdm import tqdm
from datasets import BaseAnnotationBuilder


class PLUSVeinAnnotationGenerator(BaseAnnotationBuilder):
    """
    PLUSVein-FV3 identity definition:
        identity = f'{folder}_{idx}'

    idx is one of:
        ['02', '03', '04', '07', '08', '09']

    LED and LASER are generated separately by configs.py.
    """
    finger_indices = ['02', '03', '04', '07', '08', '09']
    def collect_identities_from_subset(self, where):
        identity_to_paths = defaultdict(list)

        data_path = os.path.join(self.args.data_root, where)

        for folder in tqdm(sorted(os.listdir(data_path)), desc=f"Collect {where}"):
            folder_path = os.path.join(data_path, folder)

            if not os.path.isdir(folder_path):
                continue

            paths = sorted(glob.glob(os.path.join(folder_path, '*.png')))

            for idx in self.finger_indices:
                identity = f'{folder}_{idx}'

                filtered_paths = [
                    path
                    for path in paths
                    if identity in os.path.basename(path)
                ]

                if len(filtered_paths) > 0:
                    identity_to_paths[identity].extend(filtered_paths)

        return dict(identity_to_paths)

    def build_annotation(self):
        identity_to_paths = self.collect_identities_from_subset(
            os.path.join('PALMAR', '01')
        )

        train_ids, test_ids, identity_to_paths = self.split_identities(
            identity_to_paths,
            min_images_per_identity=1
        )

        train_samples, train_id2label = self.make_samples_from_ids(
            identity_to_paths,
            train_ids)
        test_samples, test_id2label = self.make_samples_from_ids(
            identity_to_paths,
            test_ids)
        self.print_split_report(
            self.args.dataset,
            train_samples,
            test_samples,
            train_id2label,
            test_id2label
        )
        annotation = {
            'train_set': train_samples,
            'test_set': test_samples
        }
        return annotation

class FVUSMAnnotationGenerator(BaseAnnotationBuilder):
    def collect_identities(self):
        identity_to_paths = defaultdict(list)
        session_roots = [
            os.path.join(
                self.args.data_root,
                '1st_session',
                'extractedvein'),
            os.path.join(
                self.args.data_root,
                '2nd_session',
                'extractedvein')]

        for root in session_roots:
            for sub in tqdm(sorted(os.listdir(root))):
                sub_path = os.path.join(root, sub)
                if not os.path.isdir(sub_path):
                    continue
                paths = sorted(glob.glob(os.path.join(sub_path,"*.jpg")))

                if len(paths) > 0:
                    identity_to_paths[sub].extend(paths)

        return dict(identity_to_paths)

class UTFVPAnnotationGenerator(BaseAnnotationBuilder):
    """
    UTFVP identity definition:
        identity = f'{sub}_{finger}'

    Each subject-finger pair is treated as one identity class
    """
    def collect_identities(self):
        identity_to_paths = defaultdict(list)
        for sub in tqdm(sorted(os.listdir(self.args.data_root))):
            sub_path = os.path.join(self.args.data_root, sub)
            if not os.path.isdir(sub_path):
                continue

            for finger in range(1, 7):
                identity = f'{sub}_{finger}'
                paths = sorted(
                    glob.glob(
                        os.path.join(
                            sub_path,
                            f'{sub}_{finger}_*.png')))
                if len(paths) > 0:
                    identity_to_paths[identity].extend(paths)
        return dict(identity_to_paths)

if __name__ == '__main__':
    args = configs.get_all_params()
    configs.setup_seed(args.seed)

    args.dataset = 'FV-USM'
    args = configs.get_dataset_params(args)
    generator = FVUSMAnnotationGenerator(args)
    generator.generate()

    args.dataset = 'UTFVP'
    args = configs.get_dataset_params(args)
    generator = UTFVPAnnotationGenerator(args)
    generator.generate()

    args.dataset = 'PLUSVein-FV3-LED'
    args = configs.get_dataset_params(args)
    generator = PLUSVeinAnnotationGenerator(args)
    generator.generate()

    args.dataset = 'PLUSVein-FV3-LASER'
    args = configs.get_dataset_params(args)
    generator = PLUSVeinAnnotationGenerator(args)
    generator.generate()
