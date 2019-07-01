#
# 显示VGG6多个层的多个通道的输出
import numpy as np

import matplotlib.pyplot as plt

def img_to_show(channel_image):
    channel_image -= channel_image.mean()
    channel_image /= (channel_image.std() + 1e-5)
    channel_image *= 64
    channel_image += 128
    # channel_image = np.clip(channel_image, 0, 255).astype('uint8')
    return channel_image



def image2tensor4D(img_path):
    from keras.preprocessing import image

    img = image.load_img(img_path, target_size=(150, 150))
    img_tensor = image.img_to_array(img)  # 转换（150， 150， 3）
    img_tensor = np.expand_dims(img_tensor, axis=0)  # 增加第一个维度（1，150， 150， 3）
    # Remember that the model was trained on inputs that were preprocessed in the following way:
    img_tensor /= 255.

    return img_tensor


from keras.applications import VGG16
from keras.models import Model
# 卷积基
conv_base = VGG16(weights='imagenet',
                  include_top=False,
                  input_shape=(150, 150, 3))
# img_path = "S:\\A_study\\data\\batte.jpg"
img_path = "S:\\A_study\\data\\KaggleCatsDogs\\kaggle_original\\dog.444.jpg"  # dog.909.jpg
inputs_batch = image2tensor4D(img_path)
# plt.imshow(inputs_batch[0, :, :, 0], cmap='gray')
plt.imshow(inputs_batch[0])
plt.show()

# We will tile the activation channels in this matrix
n_cols = 2
images_per_row = 16


for layer_name in ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1', ]:
    model = Model(inputs=conv_base.input, outputs=conv_base.get_layer(layer_name).output)
    features_batch = model.predict(inputs_batch)
    print(features_batch.shape)
    n_features = features_batch.shape[-1]
    # The feature map has shape (1, size, size, n_features)
    size = features_batch.shape[1]

    display_grid = np.zeros((size * n_cols, images_per_row * size))

    # We'll tile each filter into this big horizontal grid
    for col in range(n_cols):
        for row in range(images_per_row):
            channel_image = features_batch[0, :, :, col * images_per_row + row]
            display_grid[col * size: (col + 1) * size, row * size: (row + 1) * size] = img_to_show(channel_image)

    display_grid = np.clip(display_grid, 0, 255).astype('uint8')

    # Display the grid
    scale = 1. / size
    plt.figure(figsize=(scale * display_grid.shape[1], scale * display_grid.shape[0]))
    plt.grid(False)
    plt.imshow(display_grid, aspect='auto', cmap='gray')  # viridis
plt.show()

