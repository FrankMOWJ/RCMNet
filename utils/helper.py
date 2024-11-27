import torch
import argparse

def get_parser(**kwargs):
    parser = argparse.ArgumentParser(description='Surgical-DINO')
    
    # Seed
    parser.add_argument('--seed', type=int, default=3407, help='Random seed.')

    # Device
    parser.add_argument('--device', type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help='Device to use for training, either "cuda:0" or "cpu".')

    # Dataset directories
    parser.add_argument('--data_root', type=str, default='./dataset', help='directiory of the dataset.')
    parser.add_argument('--crop_size', type=int, default=224, help='Crop size of images.')
    parser.add_argument('--confidence_normalization', action='store_true', help='whether to normalze confidence map from [0, 255] to [0, 1].')

    # Model parameters
    parser.add_argument('--backbone_size', type=str, default='base', required=True, choices=['base', 'large', 'gaint'], help='Backbone size.')
    parser.add_argument('--confidence_head', type=str, default='segformer', help='Type of confidence head to use.')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training.')
    parser.add_argument('--epoch', type=int, default=100, help='Number of epochs for training.')
    parser.add_argument('--init_lr', type=float, default=0.001, help='Initial learning rate.')

    # Log and checkpoint
    parser.add_argument('--no_log', action='store_true', help='whethere to do logging.')
    parser.add_argument('--save_dir', type=str, default='./log', help='Log file and checkpoint save path.')

    # resume 
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from.')
    # suffix
    parser.add_argument('--suffix', type=str, default=None, help='suffix added to log and checkpoint name.')
    
    return parser.parse_args()

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False