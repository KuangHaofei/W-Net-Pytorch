import os, shutil
from config import Config
import torch
from torchvision import transforms

config = Config()

# Clear progress images directory
def clear_progress_dir(): # Or make the dir if it does not exist
    if not os.path.isdir(config.segmentationProgressDir):
        os.mkdir(config.segmentationProgressDir)
    else: # Clear the directory
        for filename in os.listdir(config.segmentationProgressDir):
            filepath = os.path.join(config.segmentationProgressDir, filename)
            os.remove(filepath)

def enumerate_params(models):
	num_params = 0
	for model in models:
		for param in model.parameters():
			if param.requires_grad:
				num_params += param.numel()
	print(f"Total trainable model parameters: {num_params}")
