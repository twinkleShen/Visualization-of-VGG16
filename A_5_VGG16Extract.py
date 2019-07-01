#
# 利用VGG16做特征提取 ，并存储，方便后续调用
# 有dropout，无数据增强，可用于CPU；有数据增强方法不适合用于CPU
# loss: 0.0943 - acc: 0.9710 - val_loss: 0.2347 - val_acc: 0.9020

import os
import numpy as np
from A_1_PreData import base_dataset_dir
import time
from keras.utils.vis_utils import plot_model
from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['SimHei']  # SimHei是黑体的意思


train_dir = os.path.join(base_dataset_dir, 'train')
validation_dir = os.path.join(base_dataset_dir, 'validation')
test_dir = os.path.join(base_dataset_dir, 'test')


def extract_features(directory, sample_count):
    from keras.applications import VGG16
    from keras.preprocessing.image import ImageDataGenerator

    # 卷积基
    conv_base = VGG16(weights='imagenet',
                      include_top=False,
                      input_shape=(150, 150, 3))

    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 20
    features = np.zeros(shape=(sample_count, 4, 4, 512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150, 150),
        batch_size=batch_size,
        class_mode='binary')
    i = 0
    for inputs_batch, labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size: (i + 1) * batch_size] = features_batch
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            # Note that since generators yield data indefinitely in a loop,
            # we must `break` after every image has been seen once.
            break
    return features, labels


def build_model():
    from keras import models
    from keras import layers
    from keras import optimizers

    model = models.Sequential()
    model.add(layers.Dense(256, activation='relu', input_dim=4 * 4 * 512))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=2e-5),
                  loss='binary_crossentropy', metrics=['acc'])
    return model


def train_model(model):
    train_features, train_labels = extract_features(train_dir, 2000)
    validation_features, validation_labels = extract_features(validation_dir, 1000)
    test_features, test_labels = extract_features(test_dir, 1000)

    train_features = np.reshape(train_features, (2000, 4 * 4 * 512))
    validation_features = np.reshape(validation_features, (1000, 4 * 4 * 512))
    test_features = np.reshape(test_features, (1000, 4 * 4 * 512))

    history = model.fit(train_features, train_labels,
                        epochs=30, batch_size=20, validation_data=(validation_features, validation_labels))
    return history


def show_model_info(model):
    model.summary()


def show_acc_loss(history):
    import matplotlib.pyplot as plt

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'y-.', label=u'训练精度')
    plt.plot(epochs, val_acc, 'b--', label=u'验证精度')
    plt.xlabel(u'训练轮次')
    plt.ylabel(u'识别精度')
    plt.title(u'训练集和验证集的精度曲线')
    plt.legend()  # 显示图片标签
    plt.savefig(u'small训练集和验证集的精度曲线')

    plt.figure()  # 新建一幅图
    plt.plot(epochs, loss, linestyle='-.', label=u'训练损失')
    plt.plot(epochs, val_loss, linestyle='-', label=u'验证损失')
    plt.xlabel(u'训练轮次')
    plt.ylabel(u'损失值')
    plt.title(u'训练集和验证集的损失曲线')
    plt.legend()
    plt.savefig(u'small训练集和验证集的损失曲线')

    plt.show()


def main():
    start = time.time()
    print("_ _ _ _ timing_ _ _  >_< _")

    model = build_model()
    show_model_info(model)

    history = train_model(model)

    end = time.time()
    print(end - start)
    # 保存模型
    # model.save('cats_and_dogs_small_VGG16.h5')

    # 绘制损失函数、精度函数
    show_acc_loss(history)
    # plot_model(model, to_file='my_vgg16_model.png')

if __name__ == '__main__':
    main()

    # 964.2477221488953 1001.8703298568726 615.7922213077545 608.6821143627167

