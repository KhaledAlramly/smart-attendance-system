from torch.utils.data import DataLoader
from facenet_pytorch import MTCNN,training
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
import sys




def crop(data_dir,batch_size):
    # define mtcnn model
    # define dataset 
    # define loader
    # define training loop
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))

    mtcnn = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device)

    dataset = ImageFolder(data_dir, transform=transforms.Resize((512, 512)))

    dataset.samples = [
    (p, p.replace(data_dir, data_dir + '_cropped'))
        for p, _ in dataset.samples]


    loader = DataLoader(
    dataset,
    num_workers=0,
    batch_size=batch_size,
    collate_fn=training.collate_pil)

    for i, (x, y) in enumerate(loader):
        mtcnn(x, save_path=y)
        print('\rBatch {} of {}'.format(i + 1, len(loader)), end='')
    
    # Remove mtcnn to reduce GPU memory usage
    del mtcnn


if __name__ =='__main__':
    # train_dir = './data/train'
    # val_dir = './data/val'
    data_dir = './data/all'
    batch_size = int(sys.argv[1])


    # for folder in [train_dir,val_dir]:
    #     crop(folder,batch_size)
    crop(data_dir,batch_size)























