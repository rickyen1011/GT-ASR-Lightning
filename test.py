import pathlib
import argparse
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')
from utils.utils import initialize_config, load_config


def parse_args():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Script for model inference and evaluation."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=pathlib.Path,
        help="Model configuration file (json).",
    )
    parser.add_argument(
        "--testset-config",
        required=True,
        type=pathlib.Path,
        help="Dataset configuration file (json).",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=pathlib.Path,
        help="Path of the *.Pth file of the model.",
    )
    parser.add_argument(
        "--mode",
        default="inference",
        choices=["inference", "inference_dat"],
    )

    return parser.parse_args()


def main(args):
    """
    Main function for inference and evaluation.

    Args:
        args (argparse.Namespace): Command line arguments.
    """
    config = load_config(args.config)['config']
    print (config)
    testset_config = load_config(args.testset_config)

    testset_name = args.testset_config.stem
    output_dir = args.config.parent / "results" / testset_name
    output_dir.mkdir(parents=True, exist_ok=True)

    test_dataset = initialize_config(testset_config["test_dataset"])

    lightning_module = initialize_config(
        config["lightning_module"], pass_args=False
    ).load_from_checkpoint(
        checkpoint_path=args.checkpoint_path,
        config=config
    )
    lightning_module = lightning_module.to(device="cuda").eval()
    test_dataloader = lightning_module.get_test_dataloader(test_dataset, testset_config)

    score_records = run_inference(lightning_module, test_dataloader, args.mode, testset_config)
    save_results(score_records, output_dir, args.mode)


def run_inference(lightning_module, dataloader, mode, config):
    """
    Run inference or evaluation on a dataset using a given model.

    Args:
        lightning_module (torch.nn.Module): PyTorch model for inference or evaluation.
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        mode (str): 

    Returns:
        Dict: Dictionary of score records.
    """
    score_records = dict()
    i = 0
    if mode == "inference":
        test_ter_outputs, test_acc_outputs = lightning_module.inference(dataloader, config)
        all_ter_preds, all_ter_labels = \
            [output[0] for output in test_ter_outputs], [output[1] for output in test_ter_outputs]
        all_acc_preds, all_acc_labels = \
            [output[0] for output in test_acc_outputs], [output[1] for output in test_acc_outputs]
        ter = lightning_module.ter(all_ter_preds, all_ter_labels)
        acc = lightning_module.compute_acc(all_acc_preds, all_acc_labels)
        return {"Token Error Rate (TER)": ter, "Accuracy": acc}
    elif mode == 'inference_dat':
        test_dacc_outputs = lightning_module.inference_dat(dataloader)
        all_dacc_preds, all_dacc_labels = \
            [output[0] for output in test_dacc_outputs], [output[1] for output in test_dacc_outputs]
        dacc = lightning_module.compute_acc(all_dacc_preds, all_dacc_labels)
        return {"Domain Classification Accuracy": dacc}
    else:
        raise NotImplementedError



def save_results(score_records, output_dir, mode):
    """
    Save inference or evaluation results to a file.

    Args:
        score_records (defaultdict[list]): Dictionary of score records.
        output_dir (pathlib.Path): Directory to save the results.
        mode (str):
    """
    log_file_path = output_dir / "log.txt"
    with log_file_path.open("a") as outfile:
        for key, value in score_records.items():
            print(f"{key}: {value:.4f}")
            print(f"{key}: {value:.4f}", file=outfile)

    if mode == "inference":
        json_output_path = output_dir / "inference_scores.json"
    if mode == "inference_dat":
        json_output_path = output_dir / "dat_scores.json"
    with open(json_output_path, 'w') as f:
        json.dump(score_records, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)