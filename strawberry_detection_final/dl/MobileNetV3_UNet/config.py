# Paths\ 
TRAIN_IMAGES_DIR = 'dataset/images/train'
TRAIN_MASKS_DIR  = 'dataset/masks/train'
TEST_IMAGES_DIR  = 'dataset/images/test'
INFER_IMAGES_DIR  = 'dataset/debug_crops'
TEST_MASKS_DIR   = 'dataset/masks/test'
OUTPUT_DIR       = 'overlay_results'
INFER_OUTPUT_DIR = 'inference_result'

# Model hyperparameters
IMG_SIZE   = 64
BATCH_SIZE = 32
EPOCHS     = 100
LR         = 1e-3
VAL_SPLIT  = 0.15
THRESH     = 0.5
ALPHA      = 0.5
# Device
import torch
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Weights & Biases config
WANDB_PROJECT = 'strawberry_segmentation'
WANDB_ENTITY = None  # set your wandb entity if needed
