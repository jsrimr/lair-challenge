import pickle
import torch


from sklearn.metrics import f1_score


def accuracy_function(real, pred):
    real = real.cpu()
    pred = torch.argmax(pred, dim=1).cpu()
    score = f1_score(real, pred, average='macro')
    return score

def initialize():
    # csv_feature_dict = {
    #     '내부 온도 1 평균': [3.4, 47.3],
    #     '내부 온도 1 최고': [3.4, 47.6],
    #     '내부 온도 1 최저': [3.3, 47.0],
    #     '내부 습도 1 평균': [23.7, 100.0],
    #     '내부 습도 1 최고': [25.9, 100.0],
    #     '내부 습도 1 최저': [0.0, 100.0],
    #     '내부 이슬점 평균': [0.1, 34.5],
    #     '내부 이슬점 최고': [0.2, 34.7],
    #     '내부 이슬점 최저': [0.0, 34.4]
    # }

    # crop = {'1': '딸기', '2': '토마토', '3': '파프리카',
    #         '4': '오이', '5': '고추', '6': '시설포도'}
    # disease = {
    #     '1': {
    #         'a1': '딸기잿빛곰팡이병', 'a2': '딸기흰가루병', 'b1': '냉해피해',
    #         'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
    #     },
    #     '2': {
    #         'a5': '토마토흰가루병', 'a6': '토마토잿빛곰팡이병', 'b2': '열과', 'b3': '칼슘결핍',
    #         'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
    #     },
    #     '3': {
    #         'a9': '파프리카흰가루병', 'a10': '파프리카잘록병', 'b3': '칼슘결핍',
    #         'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
    #     },
    #     '4': {
    #         'a3': '오이노균병', 'a4': '오이흰가루병', 'b1': '냉해피해',
    #         'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
    #     },
    #     '5': {
    #         'a7': '고추탄저병', 'a8': '고추흰가루병', 'b3': '칼슘결핍',
    #         'b6': '다량원소결핍 (N)', 'b7': '다량원소결핍 (P)', 'b8': '다량원소결핍 (K)'
    #     },
    #     '6': {'a11': '시설포도탄저병', 'a12': '시설포도노균병', 'b4': '일소피해', 'b5': '축과병'}
    # }
    # risk = {'1': '초기', '2': '중기', '3': '말기'}

    # label_description = {}
    # for key, value in disease.items():
    #     label_description[f'{key}_00_0'] = f'{crop[key]}_정상'
    #     for disease_code in value:
    #         for risk_code in risk:
    #             label = f'{key}_{disease_code}_{risk_code}'
    #             label_description[label] = f'{crop[key]}_{disease[key][disease_code]}_{risk[risk_code]}'

    # label_encoder = {key: idx for idx, key in enumerate(label_description)}
    # label_decoder = {val: key for key, val in label_encoder.items()}

    # with open('csv_feature_dict.pkl', 'rb') as pkl:
    #     csv_feature_dict = pickle.load(pkl)
    csv_feature_dict = {
        '내부 온도 1 평균': [3.4, 47.3],
        '내부 온도 1 최고': [3.4, 47.6],
        '내부 온도 1 최저': [3.3, 47.0],
        '내부 습도 1 평균': [23.7, 100.0],
        '내부 습도 1 최고': [25.9, 100.0],
        '내부 습도 1 최저': [0.0, 100.0],
        '내부 이슬점 평균': [0.1, 34.5],
        '내부 이슬점 최고': [0.2, 34.7],
        '내부 이슬점 최저': [0.0, 34.4]
    }

    label_description = ["1_00_0", "2_00_0", "2_a5_2", "3_00_0", "3_a9_1", "3_a9_2", "3_a9_3", "3_b3_1", "3_b6_1", "3_b7_1", "3_b8_1", "4_00_0", "5_00_0", "5_a7_2", "5_b6_1", "5_b7_1", "5_b8_1", "6_00_0", "6_a11_1", "6_a11_2", "6_a12_1", "6_a12_2", "6_b4_1", "6_b4_3", "6_b5_1",]
    label_encoder = {key: idx for idx, key in enumerate(label_description)}
    label_decoder = {val: key for key, val in label_encoder.items()}
    
    return csv_feature_dict, label_encoder, label_decoder


from glob import glob
import albumentations as A
import pandas as pd
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold, train_test_split
ROOT_DIR = 'data'

def get_train_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),
        # A.OneOf([
        #                   A.HorizontalFlip(p=1),
        #                   A.RandomRotate90(p=1),
        #                   A.VerticalFlip(p=1) ], p=0.75),
        ToTensorV2(),
    ])


def get_valid_transforms(height, width):
    return A.Compose([
        A.Resize(height=height, width=width),
        A.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])



def split_data(split_rate=0.2, seed=42, mode='train'):
    """
    Use for model trained image and time series.
    """
    if mode == 'train':
        train = sorted(glob(f'{ROOT_DIR}/train/*'))

        labelsss = pd.read_csv(f'{ROOT_DIR}/train.csv')['label']
        train, val = train_test_split(
            train, test_size=split_rate, random_state=seed, stratify=labelsss)

        return train, val
    elif mode == 'test':
        test = sorted(glob(f'{ROOT_DIR}/test/*'))

        return test