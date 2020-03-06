import numpy as np
import matplotlib.pyplot as plt
import cv2
from common import tile_detection_util as tdu
from CNN.SimpleConvNet import SimpleConvNet2


def tile_detection_CNN(img, mask, cnn, save_name=None, tmp_num_per_tile=3, tile_kind=37):
    # 手牌領域抽出
    tiles_area = tdu.get_tiles_area(img, mask)

    # 手牌諸元推定用座標
    tiles_LRTB = tdu.get_tiles_LRTB(tiles_area)
    tiles_dy = tdu.get_tiles_dy(tiles_area, tiles_LRTB)

    # 牌枚数推定
    tiles_num = tdu.get_tiles_num(tiles_LRTB, tiles_dy)
    print("--------tiles_num:", tiles_num)

    # 左端牌の幅推定
    left_tile_width_float = tdu.get_left_tile_width_float(tiles_LRTB, tiles_dy)

    # 各牌左下のx座標
    tiles_x = tdu.get_tiles_x(tiles_LRTB, left_tile_width_float, tiles_num)
    # 各牌左下のy座標
    tiles_y = tdu.get_tiles_y(tiles_area, tiles_x, tiles_dy)

    # CNN対象範囲の切り出し
    areas = tdu.get_target_areas_CNN(img, tiles_x, tiles_y)

    # CNN手牌分類
    tiles = cnn.predict(areas).argmax(axis=1) + 1

    # 描画
    if save_name is not None:
        img_recog = tdu.write_tiles_recognition(img, tiles_LRTB, tiles_x, tiles_y, tiles, save_name=save_name)
        tdu.cvimshow(img_recog)

    return tiles

if __name__ == "__main__":
    # 固定マスク画像
    mask = cv2.imread('images/mask/mask.png', cv2.IMREAD_GRAYSCALE)

    # 手牌画像
    img2 = cv2.imread('images/sample/sample.png')
    tdu.cvimshow(img2)

    # CNN
    cnn = SimpleConvNet2(input_dim=(3, 36, 24),
                         conv_param={'filter_num': 30, 'filter_size': 5, 'pad': 2, 'stride': 1},
                         hidden_size=100, output_size=37, weight_init_std=0.01)
    cnn.load_params("CNN/params.pkl")

    # 手牌検出
    tiles = tile_detection_CNN(img2, mask, cnn, save_name='images/sample/result_cnn.png')