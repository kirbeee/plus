import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import configs
import datasets
from network.fsb_hash_net import FSB_Hash_Net, Hash_Generator
from network.mamba_net import Mamba_Hash_Net
from testkit.eer_metric import EERMetric
from testkit.unlinkability_metric import UnlinkabilityMetric

def load_models(args):
    feature_extractor = Mamba_Hash_Net(embedding_size=args.dim).to(args.device)
    hash_generator = Hash_Generator(embedding_size=args.dim, do_prob=0.0, device=args.device, out_embedding_size=args.hash_dim).to(
        args.device)
    fe_path = os.path.join(args.model_root, f'{args.dataset}_feature_extractor.pth')
    gen_path = os.path.join(args.model_root,  f'{args.dataset}_best_generator.pth')
    if os.path.exists(fe_path) and os.path.exists(gen_path):
        feature_extractor.load_state_dict(torch.load(fe_path, map_location=args.device))
        hash_generator.load_state_dict(torch.load(gen_path, map_location=args.device))
        print("load model")
    else:
        raise ModuleNotFoundError

    feature_extractor.eval()
    hash_generator.eval()
    return feature_extractor, hash_generator

def unlinkability(labels, hash_user, hash_renewed, out_dir=None):
    # --- 1. 不可連結性 (Unlinkability / Revocability) ---
    sim_matrix_cross = torch.mm(
        torch.nn.functional.normalize(hash_user, p=2, dim=1),
        torch.nn.functional.normalize(hash_renewed, p=2, dim=1).t()
    ).numpy()

    targets_np = labels.numpy()
    label_matrix = (targets_np[:, None] == targets_np[None, :]).astype(int)

    mated_scores = sim_matrix_cross[label_matrix == 1]
    non_mated_scores = sim_matrix_cross[label_matrix == 0]
    print(f"Mated pairs: {len(mated_scores)}, Non-Mated pairs: {len(non_mated_scores)}")

    # prepare dir
    metric = UnlinkabilityMetric(mated_scores, non_mated_scores)
    dsys = metric.evaluate()

    if out_dir is not None:
        # make plot
        fig_path = os.path.join(out_dir, 'image_unlinkability.pdf')
        metric.plot(figure_file=fig_path)

    return dsys

class ModelEvaluator:
    def __init__(self, args, metics):
        self.args = args
        self.metics = metics
    def extract_features(self, test_loader, feature_extractor, hash_generator):
        hash_user_list, labels_list = [], []
        with torch.no_grad():
            for imgs, labels in tqdm(test_loader, desc="Feature Extraction"):
                imgs, labels = imgs.to(args.device), labels.to(args.device)
                features = feature_extractor(imgs)
                h_user = hash_generator(features, labels, training=False)
                hash_user_list.append(h_user.cpu())
                labels_list.append(labels.cpu())
        return torch.cat(hash_user_list, dim=0), torch.cat(labels_list, dim=0)

    def run_evaluation(self, embeddings, labels, **kwargs):
        result = {}
        for metric in self.metics:
            result[metric.name] = metric.compute(embeddings, labels, **kwargs)
        return result

def main(args):
    configs.setup_seed(args.seed)
    test_dataset = datasets.ImagesDataset(args=args, phase='test')
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers)
    feature_extractor, hash_generator = load_models(args)

    metics_to_run = [
        EERMetric("EER & Best ACC"),
        UnlinkabilityMetric()
    ]
    evaluator = ModelEvaluator(args, metics_to_run)
    hash_user, label = evaluator.extract_features(test_loader, feature_extractor, hash_generator)
    results = evaluator.run_evaluation(hash_user, label)

    print(f"\n================ FSB-HashNet 評估結果 [{args.dataset}] ================")
    for metric_name, metric_result in results.items():
        print(f"[{metric_name}]")
        for key, val in metric_result.items():
            if isinstance(val, float):
                print(f"   - {key}: {val:.4f}%" if "EER" in key or "ACC" in key else f"   - {key}: {val:.4f}")
            else:
                print(f"   - {key}: {val}")
    print(f"==============================================================")

if __name__ == '__main__':
    args = configs.get_all_params()
    args.dim = 1024
    args.hash_dim = 512
    for dataset in ['FV-USM', 'PLUSVein-FV3-LED','PLUSVein-FV3-LASER', 'UTFVP']:
        args.dataset = dataset
        args = configs.get_dataset_params(args)
        main(args)