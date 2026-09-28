from imutils.paths import list_images
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image, ImageFile

Image.MAX_IMAGE_PIXELS          = None
ImageFile.LOAD_TRUNCATED_IMAGES = True

def train_transform():

    return transforms.Compose([
                                transforms.Resize(size = (512, 512)),
                                transforms.RandomCrop(256),
                                transforms.ToTensor()
                            ])


class StyleTransferDataset(Dataset):

    def __init__(self, path, transform):

        self.image_paths = sorted(list_images(path))
        self.transform   = transform


    def __len__(self): return len(self.image_paths)


    def __getitem__(self, idx):

        path  = self.image_paths[idx]
        image = Image.open(path).convert('RGB')
        image = self.transform(image)

        return image

        