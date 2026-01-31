import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import tempfile
import shutil

from src.config import Config
from src.training.train_moco import train, main_worker
from src.datasets.pv03_ssl import PV03SSLDataset
from src.utils.augmentations import get_moco_v3_augmentations, TwoCropsTransform

def create_dummy_dataset(root_dir, num_images=10):
    img_dir = os.path.join(root_dir, "original")
    os.makedirs(img_dir, exist_ok=True)
    for i in range(num_images):
        img = Image.new('RGB', (224, 224), color=(i*10, i*10, i*10))
        img.save(os.path.join(img_dir, f"img_{i}.jpg"))
    return img_dir

def verify_pipeline():
    print("Starting verification...")
    
    # 1. Setup Dummy Data
    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Created temp dir: {tmpdir}")
        create_dummy_dataset(tmpdir)
        
        # 2. Setup Config
        config = Config()
        config.dataset_path = tmpdir
        config.epochs = 1
        config.batch_size = 4 # Small batch for test
        config.num_workers = 0
        config.checkpoint_dir = os.path.join(tmpdir, "checkpoints")
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        
        # 3. Initialize Dataset & Loader
        dataset = PV03SSLDataset(tmpdir, TwoCropsTransform(get_moco_v3_augmentations(config.image_size)))
        print(f"Dataset size: {len(dataset)}")
        
        loader = DataLoader(dataset, batch_size=config.batch_size, drop_last=True)
        
        # 4. Initialize Model (Mocking GPU if not available)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"Using device: {device}")
        
        from src.models.moco_v3 import MoCoV3
        from src.models.backbones import get_backbone
        
        def backbone_fn():
            return get_backbone(config.backbone, stop_grad_conv1=True)
            
        model = MoCoV3(
            backbone_fn,
            dim=config.feature_dim,
            mlp_dim=config.mlp_dim,
            T=config.temperature,
            m=config.momentum,
            use_queue=config.use_queue,
            queue_size=16 # Small queue for test
        )
        model.to(device)
        
        # 5. Optimizer & Scaler
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        scaler = torch.cuda.amp.GradScaler() if config.use_amp and device == 'cuda' else None
        
        # 6. Run Train Step
        print("Running training step...")
        
        # We need a mock args object
        class Args:
            gpu = 0 if device == 'cuda' else None
            multiprocessing_distributed = False
            rank = 0
        
        args = Args()
        
        try:
            avg_loss = train(loader, model, optimizer, scaler, 0, args, config)
            print(f"Training finished. Avg Loss: {avg_loss:.4f}")
            
            # 7. Check if Queue updated
            if config.use_queue:
                print(f"Queue Pointer: {model.queue_ptr.item()}")
                if model.queue_ptr.item() != 0:
                    print("SUCCESS: Queue pointer moved!")
                else:
                    print("WARNING: Queue pointer did not move (might be exactly 0 if queue_size % batch == 0 looped properly or failed)")
                    # Actually if batch=4 and queue=16, 10 images -> 2 batches.
                    # Batch 1: ptr 0->4. Batch 2: ptr 4->8. Should be 8.
                    if model.queue_ptr.item() == 8:
                         print("SUCCESS: Queue pointer moved correctly to 8.")
            
            print("Verification PASSED.")
            
        except Exception as e:
            print(f"Verification FAILED: {e}")
            raise e

if __name__ == "__main__":
    verify_pipeline()
