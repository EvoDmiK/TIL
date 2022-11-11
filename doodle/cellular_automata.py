from matplotlib import pyplot as plt
import numpy as np
import cv2

ROW, COLOUMN = 512, 512

## n x m 사이즈의 영행렬 생성
base_plate = np.zeros((COLOUMN, ROW))

# ## 영행렬에 랜덤하게 1로 변경해줌
for odx, base in enumerate(base_plate, 1):
    
    random_number = np.random.randint(ROW)
    random_idx = np.random.choice(ROW, random_number)
    
    for idx in random_idx: base[idx] = 1

# base_plate = np.zeros((316, 316), dtype = np.int8)

# print(base_plate.shape)
# two_powers = np.array([3**idx for idx in range(200)])

# for idx, two in enumerate(two_powers):
#     plate = [pow for pow in map(int, np.binary_repr(two))]
#     plate = np.append(np.zeros(316-len(plate)), plate)

#     base_plate[idx] = plate


# print(base_plate, type(base_plate))


def celluar_automata(array, pool_size = 3):
    
    for odx in range(array.shape[1] - pool_size + 1):
        
        for idx in range(array.shape[0] - pool_size + 1):
            neighbor = array[idx: idx + pool_size,odx: odx + pool_size]

            count = np.count_nonzero(neighbor) - 1 if neighbor[1][1] == 1 else np.count_nonzero(neighbor) 
            
            if count == 3: center = 1
            elif count >= 4: center = 0
            elif count <= 1: center = 0
            else: center = neighbor[1][1]
                
            neighbor[1][1] = center
    return array

base_plate = celluar_automata(base_plate)

iterations = 5000
for idx in range(iterations):
    print(f'{idx+1}/{iterations}')
    
    base_plate = celluar_automata(base_plate)

    plate = cv2.resize(base_plate.astype('float32'), (350, 350))
    cv2.imshow('automata', plate)

    key = cv2.waitKey(1) & 0xff

    if key == ord('q'):
        exit()
    