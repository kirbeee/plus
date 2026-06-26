import pickle
import random

class BaseAnnotationBuilder:
    """
    Base class for identity-level open-set annotation generation.

    1. collect identities
    2. split identities into train/test
    3. re-index labels within each split
    4. print report
    5. save annotation
    """
    def __init__(self, args):
        self.args = args

    def parse_split_ratio(self):
        train, test = self.args.split.split(":")
        train = int(train)
        test = int(test)
        if train <= 0 or test <= 0:
            raise ValueError(
                f"Invalid split: {self.args.split}. "
            )
        return test / (train + test)

    def collect_identities(self):
        """
        This method must be implemented by subclass.
        """
        raise NotImplementedError

    def split_identities(self, identity_to_paths, min_images_per_identity=1):
        """
        Split identities into train/test sets.

        Important:
        This function splits by identity, not by image.
        Therefore, train identities and test identities are disjoint.
        """
        filtered = {
            identity: sorted(paths)
            for identity, paths in identity_to_paths.items()
            if len(paths) >= min_images_per_identity
        }

        identities = sorted(filtered.keys())

        if len(identities) < 2:
            raise ValueError("Need at least two identities after filtering.")

        rng = random.Random(self.args.seed)
        rng.shuffle(identities)

        test_ratio = self.parse_split_ratio()
        num_test = int(round(len(identities) * test_ratio))
        num_test = max(1, min(num_test, len(identities) - 1))

        test_ids = sorted(identities[:num_test])
        train_ids = sorted(identities[num_test:])

        assert set(train_ids).isdisjoint(set(test_ids)), "Identity leakage detected!"

        return train_ids, test_ids, filtered

    def print_split_report(
            self,
            name,
            train_samples,
            test_samples,
            train_id2label,
            test_id2label
    ):
        train_ids = set(train_id2label.keys())
        test_ids = set(test_id2label.keys())
        overlap = train_ids & test_ids

        print(f"\n[{name}] identity-level open-set split")
        print(f"  train identities: {len(train_ids)}")
        print(f"  test identities : {len(test_ids)}")
        print(f"  overlap         : {len(overlap)}")
        print(f"  train samples   : {len(train_samples)}")
        print(f"  test samples    : {len(test_samples)}")

        if len(overlap) != 0:
            raise RuntimeError(
                f"Identity overlap found in {name}: {list(overlap)[:10]}"
            )

        if len(train_samples) > 0:
            print(f"  train example   : {train_samples[0]}")

        if len(test_samples) > 0:
            print(f"  test example    : {test_samples[0]}")

    def make_samples_from_ids(self, identity_to_paths, ids):
        """
        Assign contiguous labels within this split only.

        Train labels and test labels are intentionally independent.
        """
        samples = []
        id2label = {
            identity: label
            for label, identity in enumerate(sorted(ids))
        }

        for identity in sorted(ids):
            label = id2label[identity]

            for path in sorted(identity_to_paths[identity]):
                samples.append({
                    'path': path,
                    'label': label,
                    'identity': identity
                })

        return samples, id2label

    def build_annotation(self):
        """
        Build standard annotation format:
            {
                'train_set': train_samples,
                'test_set': test_samples
            }
        """
        identity_to_paths = self.collect_identities()

        train_ids, test_ids, identity_to_paths = self.split_identities(
            identity_to_paths,
            min_images_per_identity=1
        )

        train_samples, train_id2label = self.make_samples_from_ids(
            identity_to_paths,
            train_ids
        )

        test_samples, test_id2label = self.make_samples_from_ids(
            identity_to_paths,
            test_ids
        )

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

    def save_annotation(self, annotation):
        with open(self.args.annot_file, 'wb') as f:
            pickle.dump(annotation, f)
        print(f'Saved open-set annotation: {self.args.annot_file}')

    def generate(self):
        annotation = self.build_annotation()
        self.save_annotation(annotation)
