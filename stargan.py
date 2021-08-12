import itertools
import random
from datetime import datetime
from pathlib import Path
from typing import List
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from torchvision.utils import save_image
from tqdm import tqdm
from models import Cyclegan_Generator,Cyclegan_Discriminator
from torch.optim.lr_scheduler import StepLR
from sys import argv
from collections import deque

cudnn.benchmark = True
random.seed(100)
torch.manual_seed(100)

if torch.cuda.is_available():
    print("Using GPU.")
    device = torch.device('cuda:0')
else:
    print("Using CPU.")
    device = torch.device("cpu")

gan_alpha = 1
cycle_alpha = 10
identity_alpha = 1
domain_alpha_gen = 1
domain_alpha_disc = 1


image_size = 256
batch_size = 4
epochs = 25
lr = 0.0002
betas = (0.5, 0.999)
max_pools_size = 1000
save_after_how_many=5
save_image_after_how_many_iter=500
lr_decay_start=10
lr_decay_step=1
lr_gama=0.8
resize_to=300
random_crop=256
min_plus_resize=10
max_plus_resize=100
use_cross_entropy=True

photo_domain = "photo"
painter_domains = ["monet30", "cezanne", "ukiyoe", "vangogh"]
all_domains = [photo_domain] + painter_domains
number_of_domains = len(all_domains
                        )
#for transfer learning only:
#bias: train only bias | one:train only the last layer | mul: train last 3 layers of generator and last 2 layers of discriminator 
#mulplus: train last 5 layers of generator and last 2 layers of discriminator #False- dont freeze
#mulsizes: train last 3 layers of generator and first 3 layers and last 2 layers of discriminator #False- dont freeze
freeze = False
# set to True to not freeze the bias layers
keep_bias=False

generator_class=Cyclegan_Generator

discriminator_class = Cyclegan_Discriminator



transformers={}
for i in range(min_plus_resize,max_plus_resize):
    transformers[i]= transforms.Compose([
        transforms.Resize(256+i, Image.BICUBIC),
        transforms.RandomCrop(256),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

datasets_folder = Path('data')
output_folder = Path('results/')
Fake_Y = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)



domain_generator_tensors = {}
for ind_domain, domain_name in enumerate(all_domains):
    tensor = torch.zeros((batch_size, len(all_domains), 256, 256))
    tensor[:, ind_domain] = 1
    domain_generator_tensors[domain_name] = tensor.to(device, dtype=torch.float)

shape_in_discriminator = []


def one_hot(batch_size,num, index):
    a = torch.zeros(batch_size,num)
    for i in range(batch_size):
        a[i][index] = 1
    return a

domain_discriminator_tensors = {}
for ind_domain, domain_name in enumerate(all_domains):
    if use_cross_entropy:
        tensor = torch.full((batch_size, *shape_in_discriminator), ind_domain)
    else:
        tensor=one_hot(batch_size,number_of_domains,ind_domain)
    domain_discriminator_tensors[domain_name] = tensor.to(device)

def init_net_func(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def save_networks(networks_dict, networks_output_folder, checkpoint_name):
    for net_name, net in networks_dict.items():
        torch.save(net.state_dict(), f"{networks_output_folder}/{checkpoint_name}_{net_name}.pth")


def load_networks(networks_dict, networks_folder,epoch):
    for net_name, net in networks_dict.items():
        net.load_state_dict(torch.load(f"{networks_folder}/{epoch}_{net_name}.pth"))


def net_results_to_images(*net_images):
    return (0.5 * (image.detach() + 1.0) for image in net_images)


def to_device(*tensors):
    return (tensor.to(device) for tensor in tensors)


class CycleGanDataset(Dataset):
    def __init__(self, path_to_all: Path, master_domain: str, other_domains: List[str]):
        self.master_domain = master_domain
        self.other_domains = other_domains
        self.images_by_domains = {}
        for domain in self.other_domains + [self.master_domain]:
            self.images_by_domains[domain] = [Image.open(x).convert('RGB') for x in (path_to_all/domain).iterdir()]
        self.master_size = len(self.images_by_domains[self.master_domain])
        self.number_of_other_domains = len(self.other_domains)
        self.ind = 0

    def __getitem__(self, index):
        image_in_batch = self.ind % batch_size
        general_batch_number = self.ind // batch_size
        per_class_batch_number = general_batch_number // self.number_of_other_domains
        current_class_index = general_batch_number % self.number_of_other_domains
        slave_image_index = per_class_batch_number * batch_size + image_in_batch

        transform = transformers[random.randrange(min_plus_resize, max_plus_resize)]

        domain_a = self.other_domains[current_class_index]
        slave_images = self.images_by_domains[domain_a]
        a_image = transform(slave_images[slave_image_index % len(slave_images)])

        domain_b = self.master_domain
        master_images = self.images_by_domains[domain_b]
        b_image = transform(master_images[self.ind % len(master_images)])

        self.ind += 1
        return a_image, b_image, domain_a, domain_b

    def __len__(self):
        return self.master_size

class Image_pool(object):
    def __init__(self, pool_size,device):
        self.pool = deque(maxlen=pool_size)
        self.device=device

    def size(self):
        return len(self.pool)


    def push_and_sample(self,images):
        images=images.detach().cpu()
        batch_size=len(images)
        for i in range(batch_size):
            self.pool.append(images[i])
        ans=random.sample(self.pool, batch_size)
        return torch.stack(ans).to(device)
def get_dataloader():
    dataset = CycleGanDataset(datasets_folder, photo_domain, painter_domains)
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, pin_memory=False,drop_last=True)
def freeze_model(model,freeze,keep_bias):

    if freeze!="bias" and freeze!="bottom":
        keep=["last_conv.weight","conv5.weight","last_conv.bias","conv5.bias"]
    else:
        keep=[]
    add=[]
    add2=[]
    if freeze=="mul" or freeze=="mulplus" or freeze=="mulsizes":
        add=["conv4.weight","conv4.bias","up1.weight","up2.weight","up1.bias","up2.bias"]
    if freeze=="mulplus":
        add2=["res_layers.8.conv1.bias","res_layers.8.conv1.weight","res_layers.7.conv1.bias","res_layers.7.conv1.weight"]
    if (freeze=="mulsizes" and  model.__class__.__name__=="Cyclegan_Generator") or freeze=="bottom":
        add = ["conv1.weight", "conv1.bias", "down1.weight", "down2.weight", "down1.bias", "down2.bias"]
    keep=keep+add+add2
    size = 0
    for name, param in model.named_parameters():
        if name in keep or (keep_bias and "bias" in name):
            param.requires_grad = True
            print(name, param.size())
        else:
            param.requires_grad = False


generator = generator_class(input_nc=3+number_of_domains).to(device)
discriminator = discriminator_class(output_nc=1+number_of_domains).to(device)

networks = {'generator': generator, 'discriminator': discriminator}

if len(argv)>2:
    load_networks(networks, "results/"+argv[2]+"/networks/", 25)
    print("loaded previous models")
    if(freeze!=False):
        freeze_model(networks['generator'], "bottom", keep_bias)
        freeze_model(networks['generator'], "mul", keep_bias)
        freeze_model(networks['discriminator'], "mul", keep_bias)
        #for key in networks:
            #freeze_model(networks[key],freeze,keep_bias)
    

else:
    [network.apply(init_net_func) for network in networks.values()]


l1_loss = torch.nn.L1Loss().to(device)
mse_loss = torch.nn.MSELoss().to(device)
cross_entropy = torch.nn.CrossEntropyLoss().to(device)
bce_loss=torch.nn.BCELoss().to(device)
gan_loss_func = mse_loss

if use_cross_entropy:
    domain_loss_func=cross_entropy
else:
    domain_loss_func=mse_loss

generator_optimizer = torch.optim.Adam(generator.parameters(), lr=lr, betas=betas)
discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=betas)


fake_pools = {}
for ind_domain, domain_name in enumerate(all_domains):
    fake_pools[domain_name] = Image_pool(max_pools_size, device)


def get_image_from_fake_pool(pool: list, new_fake_image):
    pool.append(new_fake_image.detach().cpu())
    chosen_ind = random.randint(0, len(pool)-1)
    if len(pool) < max_pools_size:
        chosen_image = pool[chosen_ind]
    else:
        chosen_image = pool.pop(chosen_ind)
    return chosen_image.to(device)


def train_generators_by_src(domain_src, domain_dest, real_image):
    generator_optimizer.zero_grad()
    if identity_alpha>0:
        real_image_domain_src = torch.cat((real_image, domain_generator_tensors[domain_src]), 1)
        identity_image = generator(real_image_domain_src)
        loss_identity = l1_loss(identity_image, real_image) * identity_alpha
    else:
        loss_identity=0
    real_image_domain_dest = torch.cat((real_image, domain_generator_tensors[domain_dest]), 1)
    fake_image = generator(real_image_domain_dest)
    output = discriminator(fake_image)
    validity, domain = output[:, 0:1], output[:, 1:]
    loss_gan = gan_loss_func(validity, Real_Y) * gan_alpha
    #domain_target=domain_discriminator_tensors[domain_dest]
    loss_domain = domain_loss_func(domain, domain_discriminator_tensors[domain_dest])

    fake_image_domain_src = torch.cat((fake_image, domain_generator_tensors[domain_src]), 1)
    recovered_image = generator(fake_image_domain_src)
    loss_cycle = l1_loss(recovered_image, real_image) * cycle_alpha

    generator_loss = loss_gan + loss_cycle + loss_identity + loss_domain * domain_alpha_gen
    generator_loss.backward()
    generator_optimizer.step()
    return fake_image, recovered_image, generator_loss.item()


def train_discriminator(dest_domain, fake_image, real_image):
    discriminator_optimizer.zero_grad()

    output = discriminator(real_image)
    validity, domain = output[:, 0:1], output[:, 1:]
    loss_real = gan_loss_func(validity, Real_Y)
    loss_domain = domain_loss_func(domain, domain_discriminator_tensors[dest_domain])

    output = discriminator(fake_image)
    validity, domain = output[:, 0:1], output[:, 1:]
    loss_fake = gan_loss_func(validity, Fake_Y)

    real_and_fake_loss = (loss_real + loss_fake) / 2 + loss_domain * domain_alpha_disc
    real_and_fake_loss.backward()
    discriminator_optimizer.step()
    return real_and_fake_loss.item()


def single_train_cycle(real_image_a, real_image_b, domain_a, domain_b):
    real_image_a, real_image_b =to_device(real_image_a, real_image_b)

    fake_image_b, recovered_image_a, gen_loss1 = train_generators_by_src(domain_a, domain_b, real_image_a)

    fake_image_a, recovered_image_b, gen_loss2 = train_generators_by_src(domain_b, domain_a, real_image_b)

    fake_image_a_for_d =fake_pools[domain_a].push_and_sample(fake_image_a)
    disc_loss1=train_discriminator(domain_a, fake_image_a_for_d, real_image_a)

    fake_image_b_for_d =fake_pools[domain_b].push_and_sample(fake_image_b)
    disc_loss2=train_discriminator(domain_b, fake_image_b_for_d, real_image_b)

    fake_image_a, fake_image_b = net_results_to_images(fake_image_a, fake_image_b)
    recovered_image_a, recovered_image_b = net_results_to_images(recovered_image_a, recovered_image_b)
    gen_loss=(gen_loss1+gen_loss2)/2
    disc_loss=(disc_loss1+disc_loss2)/2
    return {
        'a_real': real_image_a, 'a_fake': fake_image_b, 'a_rec': recovered_image_a,
        'b_real': real_image_b, 'b_fake': fake_image_a, 'b_rec': recovered_image_b,
    },gen_loss,disc_loss


def train_loop():
    
    if len(argv)>1:
        network_name=argv[1]
    else:
        network_name=datetime.now().strftime("%d-%m-%Y__%H-%M-%S")
    train_folder_path = output_folder / network_name
    images_output_folder = train_folder_path / 'images'
    networks_output_folder = train_folder_path / 'networks'
    images_output_folder.mkdir(parents=True, exist_ok=True)
    networks_output_folder.mkdir(parents=True, exist_ok=True)

    print_freq = 10
    dataloader = get_dataloader()
    scheduler1 = StepLR(generator_optimizer, step_size=lr_decay_step, gamma=lr_gama)
    scheduler2 = StepLR(discriminator_optimizer, step_size=lr_decay_step, gamma=lr_gama)

    for epoch in range(1,epochs+1):
        progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
        total_gen_loss=0
        total_disc_loss=0
        iters=0
        for i, (image_a, image_b, domain_a_batch, domain_b_batch) in progress_bar:
            results_dict,gen_loss,disc_loss = single_train_cycle(image_a, image_b, domain_a_batch[0], domain_b_batch[0])
            total_gen_loss+=gen_loss 
            total_disc_loss+=disc_loss
            iters+=1
            if iters % save_image_after_how_many_iter==0:
                for name, image in results_dict.items():
                    save_image(image, f"{images_output_folder}/{epoch}_{iters}_{name}.png", normalize=True)
        for name, image in results_dict.items():
            save_image(image, f"{images_output_folder}/{epoch}_final_{name}.png", normalize=True)
        print(f"epoch {epoch} gen loss: {total_gen_loss/iters} disc_loss: {total_disc_loss/iters}")
        if epoch % save_after_how_many==0:
            save_networks(networks, networks_output_folder, f'{epoch}')
        if epoch >= lr_decay_start:
            scheduler1.step()
            scheduler2.step()

    save_networks(networks, networks_output_folder, 'last')


train_loop()
