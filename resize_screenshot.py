import numpy as np
import cv2

def resize_screenshot(img, save_name=None):
    """
    (1080, 1920) -> (160, 360)
    :param img: np.ndarray (1080, 1920, 3)
    :param save_name: str
    :return: np.ndarray (160, 360, 3)
    """
    # 縮小
    ratio = 0.25
    img = cv2.resize(img, dsize=None, fx=ratio, fy=ratio)

    # 手牌付近切り出し
    img = img[70:230, 100:460, :]

    if save_name is not None:
        cv2.imwrite(save_name, img)

    return img

if __name__ == "__main__":
    # 画像の読み込み
    img = cv2.imread('images/sample/screenshot.png')
    # リサイズ
    img = resize_screenshot(img, save_name='images/sample/screenshot_res.png')