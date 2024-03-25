from 本地数据集的制作 import Mask_dataset

from torchvision import transforms
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader

transform = transforms.Compose([
    transforms.Resize((56,56)),
    transforms.Grayscale(num_output_channels=1),
    ToTensor(),
    ])
train_itr = Mask_dataset("data/SF-MASK/train/trian.csv","data/SF-MASK/train",transform)
test_itr = Mask_dataset("data/SF-MASK/test/test.csv","data/SF-MASK/test",transform)
train_dataloader=DataLoader(train_itr,batch_size=255,shuffle=True)
test_dataloader=DataLoader(test_itr,batch_size=255,shuffle=True)

