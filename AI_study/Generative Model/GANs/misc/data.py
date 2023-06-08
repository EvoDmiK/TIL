import tensorflow as tf


def preprocess_image(image):
    
    ## 이미지의 픽셀 값을 -1 ~ 1 사이의 값으로 정규화
    image = tf.cast(image, tf.float32) / 127.5 - 1
    return (image)


def random_jitter(image):
    
    ## 256 x 256 사이즈를 286 x 286 사이즈로 늘린 후 256 x 256 사이즈의 이미지로 랜덤하게 자름
    image   = tf.image.resize(image, [286, 286],
                              method = tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    cropped = tf.image.random_crop(image, size = [256, 256, 3])
    
    ## 랜덤하게 잘린 이미지를 랜덤하게 좌우로 뒤집힘
    image   = tf.image.random_flip_left_right(cropped)
    return image


## 샘플 데이터를 읽어오는 함수
def read_train_sample(data):
    
    image = preprocess_image(data['image'])
    
    ## 학습용 데이터 셋인 경우 random jitter 과정도 수행
    image = random_jitter(image)
    image = tf.image.resize(image, [256, 256])
    
    return (image)


def read_test_sample(data):
    
    image = preprocess_image(data['image'])
    image = tf.image.resize(image, [256, 256])
    
    return (image)