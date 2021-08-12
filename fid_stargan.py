from numpy import log
from numpy import trace
from numpy import iscomplexobj
from scipy.linalg import sqrtm
from numpy import cov
from numpy import expand_dims
from numpy import asarray
from numpy import mean
from numpy import exp
import torchvision.utils as vutils
import numpy as np
import torch
import os
import matplotlib.pyplot as plt
from torchvision import datasets
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from mpl_toolkits.axes_grid1 import ImageGrid
import sys
import pickle
from collections import defaultdict
from models import Cyclegan_Generator
from pathlib import Path

batch_size=10
use_gpu=True
my_gen=True

generator_class=Cyclegan_Generator
class myDataSet(Dataset):
    def __init__(self, path: Path):
        self.transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
              )

        self.file_paths = list(path.iterdir())

    def __getitem__(self, index):
        image = self.transform(Image.open(self.file_paths[index ]))
        return image.to(device)

    def __len__(self):
        return len(self.file_paths)
device = None

torch.no_grad()


if torch.cuda.is_available() and use_gpu:
    device = torch.device('cuda:0')
else:
    device = torch.device("cpu")


add_to_input = torch.zeros((batch_size, 5, 256, 256))
add_to_input[:, 1] = 1
add_to_input = add_to_input.to(device, dtype=torch.float)


# calculate frechet inception distance
def calculate_fid(act1, act2):
	# calculate mean and covariance statistics
	mu1, sigma1 = act1.mean(axis=0), cov(act1, rowvar=False)
	mu2, sigma2 = act2.mean(axis=0), cov(act2, rowvar=False)
	# calculate sum squared difference between means
	ssdiff = np.sum((mu1 - mu2)**2.0)
	# calculate sqrt of product between cov
	covmean = sqrtm(sigma1.dot(sigma2))
	# check and correct imaginary numbers from sqrt
	if iscomplexobj(covmean):
		covmean = covmean.real
	# calculate score
	fid = ssdiff + trace(sigma1 + sigma2 - 2.0 * covmean)
	return fid





def create_activations_real(loader, classifier, train_tfms, rounds,model=False):
    preds_all = False
    First = True
    for images in loader:
        if model:
            images = torch.cat((images, add_to_input), 1)
            images=model(images)
        #if First:
        #    plt.imshow((images[0]).cpu().permute(1, 2, 0).detach().numpy())
        #    plt.show()    
        images = train_tfms(images)
        preds = classifier(images)
        preds = asarray(preds.detach().cpu())
        rounds = rounds-1
        if not First:
            preds_all = np.concatenate((preds_all, preds), axis=0)
        else:
            First = False
            preds_all = preds
        if rounds == 0:
            return preds_all
    return preds_all





def Identity(x):
    return x

def main():
    if(len(sys.argv) > 1):
        start=Path("results")
        file_name = start / sys.argv[1] /"networks/25_generator.pth"
        print(file_name)
    else:
        raise Exception("You must enter model path ")

    if(len(sys.argv) > 2):
        rounds = int(sys.argv[2])
    else:
        rounds=20
    my_file = file_name

    if not my_file.is_file() or len(sys.argv) <= 1:
        raise Exception("file not exist")
    gen = None
    gen=generator_class(input_nc=8)
    gen.load_state_dict(torch.load(file_name))

    gen.to(device)

    gen.eval()
    dataset_transform = transforms.Compose([transforms.ToTensor(),])
    monet_dataset = myDataSet(Path("data/monet/"))
    real_dataset = myDataSet(Path("data/photo/"))
    monet_loader = DataLoader(monet_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
    softmax = torch.nn.Softmax(dim=1)
    train_tfms = transforms.Compose([transforms.Resize(299), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    all_score = 0

    classifier = torch.load("fid/classifier.txt", map_location=device).to(device)
    classifier.fc = torch.nn.Identity()
    classifier.eval()
    #uncomment for testing frechet distance on the pretrained model

    preds_images = create_activations_real(monet_loader, classifier, train_tfms, 30,False)
    for j in range(rounds):   
        preds_all = create_activations_real(real_loader, classifier, train_tfms, 30,gen)
        #print(preds_all.shape)
        score = calculate_fid(preds_all, preds_images)
        print(f'frechet inception distance retrained {j}: {score}')
        all_score = all_score+score
    print(f'Average frechet inception distance retrained: {all_score/rounds}')


    

    ##calcuating Frechet Inception Distance
if __name__ == '__main__':
    main()