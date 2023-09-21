from torch.utils.data import Dataset
import cv2

class SegmentationDataset(Dataset):

    def __init__(self, image_paths, mask_paths, transform = None):

        self.image_paths = image_paths
        self.mask_paths  = mask_paths
        self.transform  = transform


    def __len__(self):

        return len(self.image_paths)


    def __getitem__(self, idx):

        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask  = cv2.imread(self.mask_paths[idx], 0)


        if self.transform:

            image = self.transform(image)
            mask  = self.transform(mask)

        return (image, mask)
            
    