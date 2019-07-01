#
#
# 网络中各个卷积层和池化层输出的特征图（层的输出通常被称为该层的激活，即激活函数的输出）

import os
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from A_1_PreData import original_dataset_dir

model = load_model('cats_and_dogs_small_2.h5')
img_path = os.path.join(original_dataset_dir, 'cat.12300.jpg')


# We preprocess the image into a 4D tensor
def image2tensor4D(img_path):
    from keras.preprocessing import image

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # 转换（150， 150， 3）
    img_tensor = np.expand_dims(img_tensor, axis=0)  # 增加第一个维度（1，150， 150， 3）
    # Remember that the model was trained on inputs that were preprocessed in the following way:
    img_tensor /= 255.

    return img_tensor


def show_original_img(img_tensor):
    plt.imshow(img_tensor[0])
    plt.show()


def get_layers_output(input_img_tensor):
    from keras import models

    # 构建一个显示激活的模型 这个模型有一个输入和8个输出，即每层激活对应一个输出
    layer_outputs = [layer.output for layer in model.layers[:8]]  # Extracts the outputs of the top 8 layers
    # Creates a model that will return these outputs, given the model input:
    activation_model = models.Model(inputs=model.input, outputs=layer_outputs)
    # This will return a list of 8 Numpy arrays one array per layer activation
    activations = activation_model.predict(input_img_tensor)

    return activations


# Post-process the feature to make it visually palatable
def img_to_show(channel_image):

    channel_image -= channel_image.mean()
    channel_image /= (channel_image.std() + 1e-5)
    channel_image *= 64
    channel_image += 128
    # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    return channel_image


# 将某一层的某一通道可视化
def show_one_layer(layer_number, channel_number, activations):
    one_layer_activation = activations[layer_number]
    plt.matshow(one_layer_activation[0, :, :, channel_number], cmap='viridis')
    plt.show()


# 将每个中间激活的所有通道可视化
def show_all_layers(layer_names, activations):
    images_per_row = 16  # 每行显示数目
    # Now let's display our feature maps
    for layer_name, layer_activation in zip(layer_names, activations):
        # This is the number of features in the feature map
        n_features = layer_activation.shape[-1]
        # The feature map has shape (1, size, size, n_features)
        size = layer_activation.shape[1]

        # We will tile the activation channels in this matrix
        n_cols = n_features // images_per_row  # 向下取整
        display_grid = np.zeros((size * n_cols, images_per_row * size))

        # We'll tile each filter into this big horizontal grid
        for col in range(n_cols):
            for row in range(images_per_row):
                channel_image = layer_activation[0, :, :, col * images_per_row + row]
                display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = img_to_show(channel_image)

        display_grid = np.clip(display_grid, 0, 255).astype('uint8')

        # Display the grid
        scale = 1. / size
        plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
        plt.title(layer_name)
        plt.grid(False)
        plt.imshow(display_grid, aspect='auto', cmap='viridis')  # gray
    plt.show()


def main():
    # model.summary()  # As a reminder.
    img_tensor = image2tensor4D(img_path)  # 加载需要显示的图片
    # show_original_img(img_tensor)

    activations = get_layers_output(img_tensor)

    # These are the names of the layers, so can have them as part of our plot
    layer_names = []
    for layer in model.layers[:8]:
        layer_names.append(layer.name)

    show_all_layers(layer_names, activations)


if __name__ == '__main__':
    main()
