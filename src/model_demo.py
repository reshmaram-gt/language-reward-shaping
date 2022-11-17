import torchvision.models as models
from torch.utils.data import DataLoader

from dataset import VideoLanguageDataset

dataset = VideoLanguageDataset()
dataloader = DataLoader(dataset=dataset, batch_size=5)

model = models.vgg16()

for batch in dataloader:
    output = model(batch["video"][:, 0, :, :, :])
    print(output.shape)