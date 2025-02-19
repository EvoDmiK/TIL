from argparse import ArgumentParser
from collections import Counter
import random as rand
import shutil
import time
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from imutils.paths import list_images
import numpy as np
import cv2

## args.horizontal_flip (-hf) : 이미지를 좌우 반전 시켜줌 (python augmenttation.py -hf 일 때 좌우 반전됨.)
## args.vertical_flip (-vf) : 이미지를 상하 반전 시켜줌 (python augmenttation.py -vf 일 때 상하 반전됨.)
## args.angle (-a) : 이미지를 회전시킬 각도의 범위 (e.g. -a 90 => 0~90도 사이의 각도에서 회전시킴)
## args.width_shift (-ws) :  이미지를 좌우로 움직이는 비율
## args.height_shift(-hs) : 이미지를 상하로 움직이는 비율
## args.num_augment (-n) : 증가시킬 이미지 데이터 수
## args.dataset (-d) : 데이터 셋 경로

ap = ArgumentParser()
ap.add_argument('-n', '--num_augment', required = False, type = int, default = 0)
ap.add_argument('-ws', '--width_shift', required = False, type = float, default = 0.0)
ap.add_argument('-hs', '--height_shift', required = False, type = float, default = 0.0)
ap.add_argument('-a', '--angle', required = False, type = int, default = 0)
ap.add_argument('-hf', '--horizontal_flip', action='store_true')
ap.add_argument('-vf', '--vertical_flip', action='store_true')
ap.add_argument('-d', '--dataset', required = True)
args = ap.parse_args()

## seed 값을 현재 시간에 따라 달라지게 해주는 부분
rand.seed(time.ctime())

##  설정 값보다 데이터 수가 많으면 sample/residue 폴더로 옮겨주는 함수
def move_residue(path_dict, num_dict, label):

    if num_dict[label] < 0:
        random_samples = rand.sample(path_dict[label], abs(num_dict[label]))
        os.makedirs(f'sample/residue/{label}', exist_ok = True)

        for sample in random_samples:
            file_name = sample.split(os.path.sep)[-1]

            shutil.move(sample, f'sample/residue/{label}/{file_name}')

## 데이터 셋에 있는 이미지 경로와 라벨별 이미지 경로들을 나눠주는 함수
def sorting_dataset(dataset):
    image_paths = sorted(list_images(dataset))
    labels = [image_path.split(os.path.sep)[-2] for image_path in image_paths]

    image_path_dict = {label : sorted(list_images(f'{args.dataset}/{label}')) for label in set(labels)}

    return image_paths, image_path_dict, labels


## 데이터 증강시켜주는 함수
def augment(image_path, label, option):

    ## 파일 경로를 seperator로 나눠 파일 이름 가져오는 부분.
    file_name = image_path.split(os.path.sep)[-1]

    ## 파일 이름과 확장자를 나누는 부분.
    file_name, ext = os.path.splitext(file_name)

    ## 이미지 읽고, 전처리 해주는 부분.
    image = cv2.imread(sample_image)
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)

    ## augmentation option에 맞춰 한 장씩 늘려주는 부분.
    augmented = option.flow(image, batch_size = 1)
    augmented = augmented.next()
    augmented = augmented[0].astype('uint8')
    
    cv2.imwrite(f'{args.dataset}/{label}/{file_name}_aug.{ext}', augmented)
    

## 라벨별로 늘려줘야 하는 이미지 수, 현재 이미지 수, 이미지 경로들을 나눠주는 함수 
def count_processing(dataset):

    image_paths, image_path_dict, labels = sorting_dataset(dataset)
    label_count = Counter(labels)

    mean_number = int(np.mean(list(label_count.values())))
    print(f'\n현재 데이터 셋 {args.dataset} 총 갯수 : {len(image_paths)} \n라벨별 갯수 : {label_count} \n평균 갯수 : {mean_number}, 설정 갯수 : {args.num_augment} \n')

    ## 데이터를 늘릴 수의 설정 값 (args.num_augment)이 0이면 데이터 셋의 평균 값만큼 데이터를 늘림.
    ## 0이 아니면 설정 값만큼 데이터를 늘림.
    num_iteration = {lb : (args.num_augment - num_data) for lb, num_data in label_count.items()} if args.num_augment != 0 
                     else  {lb : (mean_number- num_data) for lb, num_data in label_count.items()}    
    print(f'라벨 별로 늘려야 하는 이미지 수 입니다. \n{num_iteration} \n')
    [move_residue(image_path_dict, num_iteration, label) for label in set(labels)]

    image_paths, image_path_dict, _ = sorting_dataset(dataset)
    return num_iteration, label_count, image_path_dict


## 데이터 변형시켜서 증강시키는 함수
def data_augmentation(label, image_path, num_iteration, label_count):

    ## 인자값으로 받은 데이터 증강 옵션 설정
    aug_options = ImageDataGenerator(
        width_shift_range = args.width_shift,
        height_shift_range = args.height_shift,
        horizontal_flip = args.horizontal_flip,
        vertical_flip = args.vertical_flip,
        rotation_range = args.angle
    )

    ## num_iteration이 1이상인 경우
    if num_iteration > 0 :

        ## 이미지 1장당 늘려야 하는 횟수
        iter_for_each = num_iteration / label_count
        condition = [0 < iter_for_each, 1 > iter_for_each]

        print(f'현재 데이터 셋의 갯수는 {label_count}이고, 늘려야하는 데이터 수는 {num_iteration} 입니다.')
        print(f'한 장의 이미지가 늘려야 하는 데이터 수는 {iter_for_each} 입니다.')

        
        ## 이미지 1장당 늘려야 하는 횟수가 0 ~ 1사이의 값인 경우
        ## e.g.) 목표 이미지 수가 50 장인데, 현재는 47장만 있는 경우,
        ##       47장에서 3장만 랜덤하게 뽑아 증강시킴.
        if all(condition):
            print(f'{label} : 한 장 당 늘려야 하는 이미지의 갯수가 1개 미만 이므로, 랜덤으로 추출해 늘립니다.')
            sample_dataset = rand.sample(image_path, num_iteration)
            print(f'샘플 데이터 셋 갯수 입니다. {len(sample_dataset)} \n')

            for image_path in sample_dataset:
                augment(sample_image, label, aug_options)


        ## 이미지 1장당 늘려야 하는 횟수가 1이상인 경우
        ## e.g.) 목표 이미지 수가 50장인데 15장만 있는 경우,
        ##       전체 이미지를 3장으로 늘리고, 나머지 5장은 랜덤하게 뽑아 더 늘려줌.
        else:
            residue = num_iteration % label_count
            random_idx = rand.choice(range(len(image_path)))

            print(f'{label} : 한 장 당 늘려야 하는 이미지의 갯수가 1개 이상 이므로, 전체 데이터 셋을 늘립니다.')
            print(f'{random_idx} 번째 이미지는 다른 이미지보다 {residue}장 더 늘립니다. \n')

            for odx, image_path in enumerate(image_path, 1):

                ## 이미지 늘려주는 횟수 지정해 주는 부분.
                ## 랜덤하게 뽑힌 idx에 해당되는 이미지는 나머지 만큼 더 늘려줌.
                fix_iter_for_each = iter_for_each if odx != random_idx else (iter_for_each + residue)

                for idx in range(int(fix_iter_for_each)):
                    augment(image_path, label, aug_options)
                    
    ## 목표한 이미지 수보다 현재 이미지 수가 더 많은 경우
    ## e.g.) 목표 이미지 수가 50장인데 현재 53장 있는 경우,
    ##       데이터를 증강시키지 않고 넘김.
    else:
        print(f'{label} : 현재 데이터 수가 더 많습니다.  데이터를 늘리지 않습니다.\n')


num_iteration, label_count, image_path_dict = count_processing(args.dataset)
for label in num_iteration.keys():
    data_augmentation(label, image_path_dict[label], num_iteration[label], label_count[label])

_, _, labels = sorting_dataset(args.dataset)
print(f'augmentation이 완료 되었습니다. \naugmentation 이후 갯수입니다. {Counter(labels)}')
