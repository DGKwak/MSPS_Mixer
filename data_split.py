import os
import random
import shutil
from tqdm import tqdm

def read_file_name(file_path):
    return os.listdir(file_path)

def find_path(action):
    return os.path.join('./data/STFT', action)

def make_people_list(file_name_list):
    people_list = []
    for file_name in file_name_list:
        if file_name[2:5] in people_list:
            continue

        people_list.append(file_name[2:5])

    return people_list

def file_split(file_name_list):
    train, test = [], []

    people_list = make_people_list(file_name_list)
    random.shuffle(people_list)

    train_size = int(len(people_list)*0.8)

    train_people = people_list[:train_size]

    for file_name in file_name_list:
        if file_name[2:5] in train_people:
            train.append(file_name)
        else:
            test.append(file_name)
    
    return train, test

if __name__ == '__main__':
    rand_state = 42
    random.seed(rand_state)

    save_folder = './data/STFT_people_split'
    action_list = ['Drinking', 'Falling', 'Picking', 'Sitting', 'Standing', 'Walking']

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

        for split in ['train', 'test']:
            os.makedirs(os.path.join(save_folder, split))

            for action in action_list:
                os.makedirs(os.path.join(save_folder, split, action))
    
    for action in action_list:
        file_path = find_path(action)
        file_name_list = read_file_name(file_path)

        train, test = file_split(file_name_list)

        for train_file in tqdm(train, desc=f'{action} train moving'):
            src = os.path.join(file_path, train_file)
            dst = os.path.join(save_folder, 'train', action, train_file)
            shutil.copy(src, dst)

        for test_file in tqdm(test, desc=f'{action} test moving'):
            src = os.path.join(file_path, test_file)
            dst = os.path.join(save_folder, 'test', action , test_file)
            shutil.copy(src, dst)