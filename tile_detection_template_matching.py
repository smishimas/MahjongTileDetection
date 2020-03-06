import numpy as np
import matplotlib.pyplot as plt
import cv2
from common import tile_detection_util as tdu


def tile_detection(img, mask, templates, save_name=None, tmp_num_per_tile=3, tile_kind=37):
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

    # テンプレートマッチング対象範囲の切り出し
    areas = tdu.get_target_areas(img, tiles_x, tiles_y)

    # テンプレートマッチング手牌分類
    tiles = []
    for area_i in range(len(areas)):
        tile = tdu.recognize_tile(areas[area_i], templates, tmp_num_per_tile, tile_kind)
        tiles.append(tile)

    # 描画
    if save_name is not None:
        img_recog = tdu.write_tiles_recognition(img, tiles_LRTB, tiles_x, tiles_y, tiles, save_name=save_name)
        tdu.cvimshow(img_recog)

    return tiles

if __name__ == "__main__":
    # テンプレート画像読み込み
    dir_path = 'images/template/'
    temp_num_per_tile = 1
    templates = tdu.get_template_list(dir_path, temp_num_per_tile)

    # 固定マスク画像
    mask = cv2.imread('images/mask/mask.png', cv2.IMREAD_GRAYSCALE)

    # 手牌画像
    img = cv2.imread('images/sample/sample.png')
    tdu.cvimshow(img)

    # 手牌検出
    save_name = 'images/sample/result_template_matching.png'
    tiles = tile_detection(img, mask, templates, save_name=save_name, tmp_num_per_tile=1, tile_kind=2)