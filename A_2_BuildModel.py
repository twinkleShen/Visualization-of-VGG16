#
# 构建简单模型，并存储，方便后续调用
#


from A_1_PreData import base_dataset_dir
import os

# 加了这样一段话
import tensorflow as tf
from keras import backend as K
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
sess = tf.Session(config=config)
K.set_session(sess)

train_dir = os.path.join(base_dataset_dir, 'train')
validation_dir = os.path.join(base_dataset_dir, 'validation')


def build_simple_model():
    from keras import layers
    from keras import models

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu',
                            input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Flatten())
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))

    # 配置模型用于训练
    from keras import optimizers
    # 编译
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-4),
                  metrics=['acc'])

    return model


def train_model(model):
    # 预处理 读取图像
    from keras.preprocessing.image import ImageDataGenerator

    # 对图像数据进行扩充
    # All images will be rescaled by 1./255
    # rescale: 值将在执行其他处理前乘到整个图像上，我们的图像在RGB通道都是0~255的整数，
    # 这样的操作可能使图像的值过高或过低，所以我们将这个值定为0~1之间的数
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=40,  # 转动的角度
        width_shift_range=0.2,  # 水平偏移的幅度
        height_shift_range=0.2,  # 竖直偏移的幅度
        shear_range=0.2,  # 剪切变换的程度
        zoom_range=0.2,  # 缩放的幅度
        horizontal_flip=True, )
    # 根据路径生成图片
    # "categorical", "binary", "sparse"或None之一.默认为"categorical. 该参数决定了返回的标签数组的形式,
    # categorical 返回2D的one-hot编码标签
    # binary 返回1D的二值标签
    # sparse 返回1D的整数标签
    # None则不返回任何标签,
    # 生成器将仅仅生成batch数据, 这种情况在使用model.predict_generator()和model.evaluate_generator()等函数时会用到.
    train_generator = train_datagen.flow_from_directory(
        # This is the target directory
        train_dir,
        # All images will be resized to 150x150
        target_size=(150, 150),
        batch_size=32,  # batch数据的大小,默认32
        # Since we use binary_crossentropy loss, we need binary labels
        class_mode='binary')

    # 注意验证数据不应该扩充
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

    # 此函数返回值为history，可以跳转查阅
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,  # 步骤的总数（样本批次总数） 通常情况下，应该等于数据集的样本数量除以批量的大小。
        epochs=100,  # 在数据集上迭代的总数
        validation_data=validation_generator,
        validation_steps=50)

    # for data_batch, labels_batch in train_generator:
    #     print('data batch shape:', data_batch.shape)
    #     print('labels batch shape:', labels_batch.shape)
    #     break
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

    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()  # 显示图片标签

    plt.figure()  # 新建一幅图
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()


def main():
    model = build_simple_model()
    # show_model_info(model)

    # 返回值为history
    history = train_model(model)

    # 保存模型
    # model.save('cats_and_dogs_small_2.h5')

    # 绘制损失函数、精度函数
    show_acc_loss(history)


if __name__ == '__main__':
    main()

