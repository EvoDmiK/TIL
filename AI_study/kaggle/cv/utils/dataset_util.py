from sklearn.model_selection import train_test_split
from imutils.paths import list_images
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import os, cv2
def build_dataset(dataset_path, size = None):
    image_paths    = sorted(list_images(dataset_path))
    images, labels = [], []

    
    for image_path in image_paths:
        image  = cv2.imread(image_path)
        resize = (size + 15, size + 15) if size != None else (image.shape[0] + 15, image.shape[1] + 15)
        
        try:
          image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
          image = cv2.resize(image, resize)
          
          labels.append(image_path.split(os.path.sep)[-2])
          images.append(image)
          
        except Exception as e:
            print(e)
    
    label_names    = set(labels)
    lb2idx         = {lb : idx for idx, lb in enumerate(label_names)}
    idx2lb         = {idx  : lb for lb, idx in lb2idx.items()}
    
    labels         = [lb2idx[lb] for lb in labels]
    images, labels = np.array(images), np.array(labels)
        
    return (images, labels), (lb2idx, idx2lb)
        
        
def display_distribution(labels, add_num = 100, dtype = 'train'):
    
  lb_count  = Counter(labels)
  max_count = max(lb_count.values())
  
  plt.bar(lb_count.keys(), lb_count.values())
  plt.ylim(0, max_count + add_num)
  plt.title(f'{dtype} dataset')
    
  for grade, num in lb_count.items():
      plt.text(
                  grade, num, num,
                  fontsize = 11, color = 'black',
                  horizontalalignment = 'center',
                  verticalalignment = 'bottom'
              )
  plt.show()


def display_image(data, idx2lb, tensor = True):
    
    image, idx = data
    image     = np.transpose(image, (1, 2, 0)) if tensor else image
    
    plt.imshow(image)
    plt.title(idx2lb[idx])
    plt.axis(False)
    
    