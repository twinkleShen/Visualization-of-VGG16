#
# 这是一个数据集生成文件，用于从原始数据中分类出训练、测试、验证集
# 已有数据集则不需要调用

import os
import shutil

# The path to the directory where the original dataset was uncompressed
original_dataset_dir = 'S:\\A_study\\data\\KaggleCatsDogs\\kaggle_original'
# The directory where we will store our smaller dataset
base_dataset_dir = 'S:\\A_study\\data\\KaggleCatsDogs\\kaggle_smallSet'


def path_join_mkdir(base_dataset_dir, one_name):
    new_dir = os.path.join(base_dataset_dir, one_name)
    os.mkdir(new_dir)
    return new_dir


def get_data_set():
    os.mkdir(base_dataset_dir)  # 生成此路径

    # Directories for our training,validation and test splits
    train_dir = path_join_mkdir(base_dataset_dir, 'train')
    validation_dir = path_join_mkdir(base_dataset_dir, 'validation')
    test_dir = path_join_mkdir(base_dataset_dir, 'test')

    # Directory with our training cat pictures
    train_cats_dir = path_join_mkdir(train_dir, 'cats')

    # Directory with our training dog pictures
    train_dogs_dir = path_join_mkdir(train_dir, 'dogs')

    # Directory with our validation cat pictures
    validation_cats_dir = path_join_mkdir(validation_dir, 'cats')

    # Directory with our validation dog pictures
    validation_dogs_dir = path_join_mkdir(validation_dir, 'dogs')

    # Directory with our validation cat pictures
    test_cats_dir = path_join_mkdir(test_dir, 'cats')

    # Directory with our validation dog pictures
    test_dogs_dir = path_join_mkdir(test_dir, 'dogs')

    # Copy first 1000 cat images to train_cats_dir
    # Copy next 500 cat images to validation_cats_dir
    # Copy next 500 cat images to test_cats_dir
    fnames = ['cat.{}.jpg'.format(i) for i in range(2000)]
    i = 0
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        if i < 1000:
            dst = os.path.join(train_cats_dir, fname)
        elif i<1500:
            dst = os.path.join(validation_cats_dir, fname)
        else:
            dst = os.path.join(test_cats_dir, fname)
        i += 1
        shutil.copyfile(src, dst)

    # Copy first 1000 dog images to train_dogs_dir
    # Copy next 500 dog images to validation_dogs_dir
    # Copy next 500 dog images to test_dogs_dir
    fnames = ['dog.{}.jpg'.format(i) for i in range(2000)]
    i = 0
    for fname in fnames:
        src = os.path.join(original_dataset_dir, fname)
        if i < 1000:
            dst = os.path.join(train_dogs_dir, fname)
        elif i < 1500:
            dst = os.path.join(validation_dogs_dir, fname)
        else:
            dst = os.path.join(test_dogs_dir, fname)
        i += 1
        shutil.copyfile(src, dst)

    # 显示数据信息
    print('total training cat images:', len(os.listdir(train_cats_dir)),
          '\n total training dog images:', len(os.listdir(train_dogs_dir)),
          '\n total validation cat images:', len(os.listdir(validation_cats_dir)),
          '\n total validation dog images:', len(os.listdir(validation_dogs_dir)),
          '\n total test cat images:', len(os.listdir(test_cats_dir)),
          '\n total test dog images:', len(os.listdir(test_dogs_dir)))


if __name__ == '__main__':
    get_data_set()

