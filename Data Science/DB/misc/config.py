import json
import os

from easydict import EasyDict as edict

SEP         = os.path.sep
ROOT_PATH   = SEP.join(os.getcwd().split(SEP)[:-5])
CONFIG_PATH = f'{ROOT_PATH}/utils/configs/config.json'
PORT_PATH   = f'{ROOT_PATH}/utils/configs/ports.json'

read_json   = lambda json_path       : json.loads(open(json_path, 'r').read())
save_json   = lambda json_path, dict_: json.dump(dict_, open(json_path, 'w'), indent = 2)

def repair_json(json_path: str, type_: str = 'config') -> dict:
    
    try:
        
        print(f'[INFO] {type_}.json 파일을 로딩합니다.')
        json_ = read_json(json_path)
        
    except:
        
        print(f'[WARN] {type_}.json 파일이 손상되어 데이터를 복구 합니다.')
        texts = open(f'{ROOT_PATH}/BACKUPS/configs/{type_}.txt', 'r').readlines()
        
        texts = [text.split() for text in texts]
        json_ = {name : text for name, text in texts}
        
        save_path = [f'{ROOT_PATH}/utils/configs', f'{ROOT_PATH}/BACKUPS/configs']
        
        for save_path in save_paths:
            os.makedirs(save_path, exist_ok = True)
            save_json(f'{save_path}/{type_}.json', json_)
            
    finally: return json_


def get_json(json_path: str, type_: str = 'config') -> edict:
    
    if os.path.isfile(json_path):
        
        json_ = repair_json(json_path, type_ = type_)
        
    else:
        print(f'[WARN] {type_}.json 파일이 존재하지 않아 백업 데이터를 로딩합니다.')
        json_ = repair_json(f'{ROOT_PATH}/BACKUPS/configs/{type_}.json', type_ = type_)
        
    
    return edict(json_)


CONFIGS = get_json(CONFIG_PATH)
PORTS   = get_json(PORT_PATH, type_ = 'ports')
