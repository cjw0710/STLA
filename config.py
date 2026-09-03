from typing import List, Optional
import argparse
from dataclasses import dataclass
import os

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="The clean implementation of SILN.")
    parser.add_argument("--seed", type=int, default=21, help="Random seed for reproducibility")
    parser.add_argument('--dropout', type=float, default=0.3, help="Dropout ratio for model layers")
    data_group = parser.add_argument_group('Data Configs')
    data_group.add_argument('--dataset', type=str, default='christian', choices=list(DATASET_CONFIGS.keys()), help="Dataset to use for training")
    data_group.add_argument('--batch_size', type=int, default=32, help="Batch size for training")
    data_group.add_argument('--max_len', type=int, default=200, help="Maximum length of cascade")
    data_group.add_argument('--transfer_threshold', type=int, default=2, help="Maximum length of cascade")
    data_group.add_argument('--time_intervals', type=int, default=20, help="Maximum length of cascade")
    data_group.add_argument('--sample_hop', type=int, default=2, help="Maximum length of cascade")
    learning_group = parser.add_argument_group('Learning Configs')
    learning_group.add_argument('--max_epochs', type=int, default=50, help="Maximum number of training epochs")
    learning_group.add_argument('--print_steps', type=int, default=10, help="Number of steps between training metric logs")
    learning_group.add_argument('--learning_rate', type=float, default=1e-3, help="Initial learning rate")
    model_group = parser.add_argument_group('Model Hyperparameters')
    model_group.add_argument('--sample_k', type=int, default=100, help="Number of neighbors to sample")
    model_group.add_argument('--window_size', type=int, default=3, help="Size of current stage (q)")
    model_group.add_argument('--dim', type=int, default=64, help="Dimension of embeddings (d)")
    model_group.add_argument('--n_heads', type=int, default=6, help="Number of attention heads (B)")
    model_group.add_argument('--metric_k', type=List[int], default=[10, 50, 100], help="K values for evaluation metrics")
    model_group.add_argument('--gcn_layer', type=int, default=1, help="Number of attention heads (B)")
    model_group.add_argument('--model', type=str, default='dediff', choices=['dediff', 'baseline'], help="Choose model for inference/training")
    checkpoint_group = parser.add_argument_group('Checkpoint Configs')
    checkpoint_group.add_argument('--saved_model_path', type=str, default='checkpoint/', help="Directory to save model checkpoints")
    checkpoint_group.add_argument('--ckpt_file', type=str, default='checkpoint/model_.bin', help="Path to save the model checkpoint")
    checkpoint_group.add_argument('--best_score', type=float, default=0.0, help="Minimum score improvement to save checkpoint")

    inference_group = parser.add_argument_group('Inference Configs')
    inference_group.add_argument('--repeat_test', type=int, default=1, help="Repeat the test dataset N times for speed measurement")
    inference_group.add_argument('--warmup_batches', type=int, default=5, help="Warmup batches to skip in speed timing")
    inference_group.add_argument('--max_speed_batches', type=int, default=0, help="Max batches to time (0 for all)")

    return parser.parse_args()

def setup_info(args: argparse.Namespace) -> None:
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {args.dataset}. "
                       f"Available datasets: {list(DATASET_CONFIGS.keys())}")
    config = DATASET_CONFIGS[args.dataset]
    for key, value in config.__dict__.items():
        if value is not None:
            setattr(args, key, value)
    base_path = f'dataset/{args.dataset}'
    
    def get_path_if_exists(path: str) -> Optional[str]:
        return path if os.path.exists(path) else None
    
    args.cascade_train_path = f'{base_path}/cascade_train.json'
    args.cascade_valid_path = f'{base_path}/cascade_valid.json'
    args.cascade_test_path = f'{base_path}/cascade_test.json'
    args.graphPath = f'{base_path}/graph.npz'
    args.dis_path = get_path_if_exists(f'{base_path}/distance.npy')
    args.social_graph_path = f'{base_path}/social_graph.npz'
    args.interaction_graph_path = f'{base_path}/interaction_graph.pt'
    args.frequency_path = f'{base_path}/frequency.pt'
@dataclass
class DatasetConfig:
    user_num: int
    dim: int
    n_warmup_steps: int
    window_size: int
    batch_size: Optional[int] = None
    n_heads: Optional[int] = None
    transfer_threshold: Optional[int] = None
    sample_hop: Optional[int] = None
DATASET_CONFIGS = {
    'twitter': DatasetConfig(
        user_num=12627 + 2,
        dim=128,
        n_warmup_steps=1000,
        window_size=3,
        batch_size=64,
        n_heads = 10,
        transfer_threshold = 3,
        sample_hop = 2,
    ),
    'douban': DatasetConfig(
        user_num=12232 + 2,
        dim=64,
        n_warmup_steps=500,
        window_size=3,
        n_heads=8,
    ),
    'android': DatasetConfig(
        user_num=2927 + 2,
        dim=128,
        n_warmup_steps=200,
        window_size=2
    ),
    'christian': DatasetConfig(
        user_num=1651 + 2,
        dim=64,
        n_warmup_steps=200,
        window_size=2
    )
}
