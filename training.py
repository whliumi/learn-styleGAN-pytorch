
from training.networks_stylegan import G_synthesis, D_basic, G_mapping
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import dataset
import argparse
import os
import random
from torch import optim
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import torchvision
import torchvision.transforms as transforms

def gettensor(x, device):
    return x.to(device)

learning_rate = 0.0001
def train(mapping_net, generator, discriminator, loss_fc, train_dataloader, epochs, save_dir, device):
    costs = []
    target = torch.zeros(4, 1)
    for ep in range(epochs):
        if ep % 2 == 0:
            mapping_net.train()
            generator.train()
            discriminator.eval()
            mapping_optimizer = optim.Adam(mapping_net.parameters(), lr=learning_rate)
            generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
            discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
            print_loss_total = 0
            print_ep = 100

            print('Ep %d' % ep)
            for i, x in enumerate(train_dataloader):

                mapping_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                x = gettensor(x, device)
                latent_z = torch.randn(size=(x.shape[0], 512))
                latent_w = mapping_net(latent_z)
                fake_image, image_out = generator(latent_w)
                loss = loss_fc(discriminator(fake_image), target)
                loss.backward()
                mapping_optimizer.step()
                generator_optimizer.step()
                discriminator_optimizer.step()

                print_loss_total += loss.item()
                if (i + 1) % print_ep == 0:
                    print_loss_avg = print_loss_total / print_ep
                    print_loss_total = 0
                    print(' %.4f' % print_loss_avg)
                    costs.append(loss.item())
        else:
            mapping_net.eval()
            generator.eval()
            discriminator.train()
            mapping_optimizer = optim.Adam(mapping_net.parameters(), lr=learning_rate)
            generator_optimizer = optim.Adam(generator.parameters(), lr=learning_rate)
            discriminator_optimizer = optim.Adam(discriminator.parameters(), lr=learning_rate)
            print_loss_total = 0
            print_ep = 100

            print('Ep %d' % ep)
            for i, x in enumerate(train_dataloader):

                mapping_optimizer.zero_grad()
                generator_optimizer.zero_grad()
                discriminator_optimizer.zero_grad()
                x = gettensor(x, device)
                latent_z = torch.randn(size=(x.shape[0], 512))
                latent_w = mapping_net(latent_z)
                fake_image, image_out = generator(latent_w)
                fake_loss = discriminator(fake_image)
                real_loss = discriminator(x)
                loss = loss_fc(real_loss, target) - loss_fc(fake_loss, target)
                loss.backward()
                mapping_optimizer.step()
                generator_optimizer.step()
                discriminator_optimizer.step()

                print_loss_total += loss.item()
                if (i + 1) % print_ep == 0:
                    print_loss_avg = print_loss_total / print_ep
                    print_loss_total = 0
                    print(' %.4f' % print_loss_avg)
                    costs.append(loss.item())
    torch.save(mapping_net.state_dict(),
               os.path.join(save_dir, 'mapping_net.pt'))
    torch.save(generator.state_dict(),
               os.path.join(save_dir, 'generator.pt'))
    torch.save(discriminator.state_dict(),
               os.path.join(save_dir, 'discriminator.pt'))
    plt.plot(np.squeeze(costs))
    plt.ylabel('costs')
    plt.xlabel('training batch (hundred)')
    plt.title("learning_rate" + str(learning_rate))
    plt.show()

def generate_image(mapping_net, generator, batch_size, save_dir):
    mapping_net.load_state_dict(torch.load(os.path.join(save_dir, 'mapping_net.pt')))
    generator.load_state_dict(torch.load(os.path.join(save_dir, 'mapping_net.pt')), strict=False)
    latent_z = torch.randn(size=(batch_size, 512))
    latent_w = mapping_net(latent_z)
    generate_img, image_out = generator(latent_w)
    for i in range(batch_size):
        img = generate_img[i].detach().numpy()
        img = np.clip(img, 0, 255)
        print(img.shape)
        print(img)
        img = img.transpose([1,2,0])
        fname = '%d.jpg' % i
        save_file = os.path.join(save_dir, fname)
        plt.imsave(save_file, img)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='-1')
    parser.add_argument('--save_dir', type=str, default='train_checkpoint')
    parser.add_argument('--num_epochs', type=int, default=6)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--rand_seed', type=int, default=42)
    args = parser.parse_args()

    opt = parser.parse_args()

    if int(opt.gpu) < 0:
        device = torch.device('cpu')
        torch.manual_seed(opt.rand_seed)
    else:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu
        device = torch.device("cuda")
        torch.manual_seed(opt.rand_seed)
        torch.cuda.manual_seed(opt.rand_seed)

    random.seed(opt.rand_seed)

    image = dataset.create_from_images('FFHQ_data', 'FFHQ1', shuffle=1, need_save=0)
    train_image = dataset.ImageDataset(image)
    train_dataloader = DataLoader(train_image, batch_size=opt.batch_size, drop_last=True)

    mapping_net = G_mapping()
    generator = G_synthesis(opt.batch_size)
    discriminator = D_basic(opt.batch_size)
    loss = torch.nn.L1Loss(reduction='mean')
    train(mapping_net, generator, discriminator, loss, train_dataloader, opt.num_epochs, opt.save_dir, device)



    generate_image(mapping_net, generator, opt.batch_size, opt.save_dir)




