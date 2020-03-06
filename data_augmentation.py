import numpy as np
from keras.preprocessing.image import ImageDataGenerator


def save_da_images(images, batch_size=10, save_to_dir=None):
    """
    :param images: Numpy array of rank 4 (image_num, height, width, channels)
    :param batch_size: int
    :param save_to_dir: str.
    :return: None
    """
    datagen = ImageDataGenerator(
        rotation_range=1,
        width_shift_range=0.10,
        height_shift_range=0.05,
        shear_range=1, zoom_range=0.1)

    gen = datagen.flow(images, batch_size=batch_size, save_to_dir=save_to_dir, save_prefix='da')

    if save_to_dir:
        for i in range(batch_size):
            gen_img = next(gen)


if __name__ == "__main__":
    import numpy as np
    from keras.preprocessing.image import load_img, img_to_array

    image = load_img('images/dataset/train/1/1_0.png')
    x = img_to_array(image)[np.newaxis]  # (Height, Width, Channels)  -> (1, Height, Width, Channels)

    import os
    save_to_dir = 'images/dataset/train_da/1/'
    if not os.path.exists(save_to_dir):
        os.mkdir(save_to_dir)

    save_da_images(images=x, batch_size=5, save_to_dir=save_to_dir)