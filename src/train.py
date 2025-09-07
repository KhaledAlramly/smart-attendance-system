import numpy as np
import torch
import sys

from torchvision import transforms
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization, training
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader,SubsetRandomSampler
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter


#take train,val dataset
#outputs saved trained model   
#------------------------

# define datasets
# define dataloader
# define model
# define training loop

def train(data_dir,
          batch_size,
          workers,
          num_classes,
          epochs,
          model_path,
          model_name
          ):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('Running on device: {}'.format(device))
    trans = transforms.Compose([
        np.float32,
        transforms.ToTensor(),
        fixed_image_standardization
                                ])
    

    # trainset = ImageFolder(train_dir,transform=trans)
    # valset= ImageFolder(val_dir,transform=trans)
    dataset = ImageFolder(data_dir,transform=trans)
    img_inds = np.arange(len(dataset))
    np.random.shuffle(img_inds)
    train_inds = img_inds[:int(0.8 * len(img_inds))]
    val_inds = img_inds[int(0.8 * len(img_inds)):]


    trainloader = DataLoader(
    dataset,
    num_workers=workers,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(train_inds)
    )
    valloader =DataLoader(
        dataset,
        num_workers=workers,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(val_inds)
    )
    
    resnet = InceptionResnetV1(
    classify=True,
    pretrained='vggface2',
    num_classes=num_classes
        ).to(device)

    optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    scheduler = MultiStepLR(optimizer, [5, 10])
    loss_fn = torch.nn.CrossEntropyLoss()
    metrics = {
        'fps': training.BatchTimer(),
        'acc': training.accuracy
                                }
    writer = SummaryWriter()
    writer.iteration, writer.interval = 0, 10

    print('\n\nInitial')
    print('-' * 10)
    resnet.eval()
    training.pass_epoch(
        resnet, loss_fn, valloader,
        batch_metrics=metrics, show_running=True, device=device,
        writer=writer
    )

    for epoch in range(epochs):
        print('\nEpoch {}/{}'.format(epoch + 1, epochs))
        print('-' * 10)

        resnet.train()
        training.pass_epoch(
            resnet, loss_fn, trainloader, optimizer, scheduler,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

        resnet.eval()
        training.pass_epoch(
            resnet, loss_fn, valloader,
            batch_metrics=metrics, show_running=True, device=device,
            writer=writer
        )

    torch.save({
    'model_state_dict': resnet.state_dict(),
    'class_to_idx': dataset.class_to_idx,
    'idx_to_class': {v: k for k, v in dataset.class_to_idx.items()},
    'num_classes': len(dataset.class_to_idx)
}, f'{model_path}/{model_name}.pth')

    print("Model saved successfully!")
    print(f"Classes: {list(dataset.class_to_idx.keys())}")
    

    writer.close()


if __name__ == "__main__":
    # train_dir = "./data/train_cropped"
    # val_dir = "./data/val_cropped"
    data_dir = "./data/all_cropped"
    batch_size = int(sys.argv[1])
    workers = int(sys.argv[2])
    num_classes = 5
    epochs = 10
    model_name = 'model'
    model_path = "./models"


    train(data_dir,
          batch_size,
          workers,
          num_classes,
          epochs,
          model_path,
          model_name)
    





