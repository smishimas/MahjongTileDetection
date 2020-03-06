# coding: utf-8
import os, glob
import numpy as np
import matplotlib.pyplot as plt
import cv2

tiles_dict = {1:'1m', 2:'2m', 3:'3m', 4:'4m', 5:'5m', 6:'6m', 7:'7m', 8:'8m', 9:'9m', 10:'r5m',
                 11:'1p', 12:'2p', 13:'3p', 14:'4p', 15:'5p', 16:'6p', 17:'7p', 18:'8p', 19:'9p', 20:'r5p',
                 21:'1s', 22:'2s', 23:'3s', 24:'4s', 25:'5s', 26:'6s', 27:'7s', 28:'8s', 29:'9s', 30:'r5s',
                 31:'E', 32:'S', 33:'W', 34:'N', 35:'Wh', 36:'G', 37:'R'}

def cvimshow(img):
    if img.ndim==3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
    else:
        plt.imshow(img, cmap='gray')

def get_template_list(dir_path='images/template/', temp_num_per_tile=1):
    """テンプレート画像読み込み
    :param dir_path: str
    :param temp_num_per_tile: int
    :return: list
    """
    templates = []

    # 各牌のディレクトリパス
    temp_dir_paths = glob.glob(dir_path +'*')

    for temp_dir_path in temp_dir_paths:
        # 画像パス取得
        temp_paths = glob.glob(temp_dir_path +'/*.png')
        for path in temp_paths[:temp_num_per_tile]:
            template = cv2.imread(path)
            templates.append(template)

    return templates

def remove_objects(img, lower_size=None, upper_size=None):
    """小領域を除去
    https://axa.biopapyrus.jp/ia/opencv/remove-objects.html
    """
    # find all objects
    nlabels, labels, stats, centroids = cv2.connectedComponentsWithStats(img)

    sizes = stats[1:, -1]
    _img = np.zeros((labels.shape))

    # process all objects, label=0 is background, objects are started from 1
    for i in range(1, nlabels):

        # remove small objects
        if (lower_size is not None) and (upper_size is not None):
            if lower_size < sizes[i - 1] and sizes[i - 1] < upper_size:
                _img[labels == i] = 255

        elif (lower_size is not None) and (upper_size is None):
            if lower_size < sizes[i - 1]:
                _img[labels == i] = 255

        elif (lower_size is None) and (upper_size is not None):
            if sizes[i - 1] < upper_size:
                _img[labels == i] = 255

    return _img


def morphology(img):
    '''モルフォロジー変換
    '''
    # カーネルの指定
    kernel = np.ones((5, 5), np.uint8)
    # クロージング
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    # オープニング
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)

    return img


def bgr2binary(img, mask=None):
    """
    牌以外の領域を除外する
    """
    binary_img = np.ones((img.shape[0], img.shape[1]))

    # 固定マスク
    if mask is not None:
        img[mask == 0] = 0
        binary_img[mask == 0] = 0

    # 雀卓の緑
    loc_desk = (img[:, :, 0] > 90) * (img[:, :, 0] < 220) * \
               (img[:, :, 1] > 80) * (img[:, :, 1] < 220) * \
               (img[:, :, 2] > 40) * (img[:, :, 2] < 120)
    # 肌
    loc_skin = (img[:, :, 0] > 60) * (img[:, :, 0] < 180) * \
               (img[:, :, 1] > 60) * (img[:, :, 1] < 180) * \
               (img[:, :, 2] > 190) * (img[:, :, 2] < 256)
    # 牌の背中（黄）
    loc_yellow = (img[:, :, 0] > 100) * (img[:, :, 0] < 180) * \
                 (img[:, :, 1] > 170) * (img[:, :, 1] < 256) * \
                 (img[:, :, 2] > 230) * (img[:, :, 2] < 256)
    # 牌の背中（青）
    loc_blue = (img[:, :, 0] > 110) * (img[:, :, 0] < 200) * \
               (img[:, :, 1] > 70) * (img[:, :, 1] < 120) * \
               (img[:, :, 2] > 40) * (img[:, :, 2] < 70)

    # 牌の側面
    loc_tile = (img[:, :, 0] > 240) * (img[:, :, 1] > 240) * (img[:, :, 2] > 240)
    # 小領域除去
    loc_tile = remove_objects(loc_tile.astype(np.uint8), lower_size=200, upper_size=None)
    # 牌の側面より右側
    tile_right = np.where(loc_tile)[1].max() + 1  # オフセット
    loc_right = np.zeros_like(binary_img, dtype=np.bool)
    loc_right[:, tile_right:] = True

    # 統合
    binary_img[loc_desk + loc_skin + loc_yellow + loc_blue + loc_right] = 0

    return binary_img


def get_tiles_area(img, mask=None, lower_size=600):
    """手牌画像、固定マスクから手牌領域を抽出
    """
    # 2値化
    binary_img = bgr2binary(np.copy(img), mask)

    # モルフォロジー変換
    binary_img = morphology(binary_img)

    # 小領域除去
    binary_img = remove_objects(binary_img.astype(np.int8), lower_size, upper_size=None)

    return binary_img


def get_tiles_LRTB(tiles_area):
    """手牌両端の下角座標
    """
    left_tile_bottom = np.where(tiles_area!=0)[0][-1]
    left_tile_left = np.min(np.where(tiles_area[left_tile_bottom, :]!=0)[0]) + 2# オフセット
    right_tile_right = np.max(np.where(tiles_area!=0)[1])# オフセット
    right_tile_top = np.max(np.where(tiles_area[:, right_tile_right])[0])
    tiles_LRTB = np.array([left_tile_left, right_tile_right, right_tile_top, left_tile_bottom])
    return tiles_LRTB


def get_tiles_dy(tiles_area, tiles_LRTB):
    """手牌全体の傾き
    """
    x2 = int(tiles_LRTB[1] - 20)
    y2 = np.max(np.where(tiles_area[:, x2]))
    tiles_dy = (y2 - tiles_LRTB[3]) / (x2 - tiles_LRTB[0])
    return tiles_dy


def get_tiles_num_float(tiles_LRTB, tiles_dy, a=2.5894, b=-0.05048, c=-0.05913, d=0.05472, e=-15.9759):
    """牌枚数推定:出力float
    """
    tiles_num = a + b * tiles_LRTB[0] + c * tiles_LRTB[3] + d * tiles_LRTB[1] + e * tiles_dy
    return tiles_num


def get_tiles_num(tiles_LRTB, tiles_dy, a=2.5894, b=-0.05048, c=-0.05913, d=0.05472, e=-15.9759):
    """牌枚数推定:出力int
    """
    tiles_num = a + b * tiles_LRTB[0] + c * tiles_LRTB[3] + d * tiles_LRTB[1] + e * tiles_dy
    return int(round(tiles_num))


def get_left_tile_width_float(tiles_LRTB, tiles_dy, a=-0.003410, b=0.098469, c=-1.3088700, d=9.465729):
    """左端牌の幅推定:出力float
    """
    width = a * tiles_LRTB[0] + b * tiles_LRTB[3] + c * tiles_dy + d
    return width


def get_tiles_x(tiles_LRTB, left_tile_width, tiles_num):
    """各牌の左下角のx座標
        牌幅は右(奥)にいくほど等差数列的に小さくなると仮定
        最後尾に右端牌の右下角のx座標も追加
    :return: list
                len(list): tiles_num + 1
    """
    W = tiles_LRTB[1] - tiles_LRTB[0]
    w0 = left_tile_width
    N = tiles_num
    a = 2 * (W - w0 * N) / (N * (N - 1))# 公差

    tiles_x = []
    x = tiles_LRTB[0]
    tiles_x.append(x)
    for i in range(tiles_num):
        width = w0 + a * i
        x += width
        tiles_x.append(int(round(x)))

    return tiles_x


def get_tiles_y(tiles_area, tiles_x, tiles_dy):
    """各牌の左下角のy座標
    :return: list
                len(list):　len(tiles_x)
    """
    tiles_y = []
    for i in range(len(tiles_x) - 1):
        y = np.max(np.where(tiles_area[:, int(tiles_x[i])]!=0))
        tiles_y.append(int(y))
    # 右端牌の右下角
    y = tiles_y[-1] + tiles_dy * (tiles_x[-1] - tiles_x[-2])
    tiles_y.append(int(round(y)))
    return tiles_y


def get_target_areas(img, tiles_x, tiles_y):
    """パターンマッチング対象の範囲を切り出し
    """
    w, h = 32, 44
    tiles_num = len(tiles_x)-1
    areas = np.empty((tiles_num, h, w, 3))
    for i in range(tiles_num):
        x = tiles_x[i] - 4
        y = tiles_y[i] + 8
        area = np.copy(img[y-h:y, x:x+w, :])
        if x+w>tiles_x[i+1]+4:# 右隣の牌が映らないように黒塗りする
            x_del = tiles_x[i+1]+2-x
            area[:, x_del:, :] = 0
        areas[i] = area
    return areas.astype(np.uint8)


def get_target_areas_CNN(img, tiles_x, tiles_y):
    """CNN対象の範囲を切り出し
    """
    w, h = 24, 36
    tiles_num = len(tiles_x)-1
    areas = np.empty((tiles_num, h, w, 3))
    for i in range(tiles_num):
        x = tiles_x[i] - 1 - int(5*i/tiles_num)
        y = tiles_y[i] + 1 + int(6*i/tiles_num)
        area = np.copy(img[y-h:y, x:x+w, :])
        areas[i] = area
    areas = areas.transpose(0, 3, 1, 2)
    areas /= 255
    return areas


def red_or_black(img, tile, thresh_val=30, thresh_sum=50):
    """赤牌 or 黒牌 識別
    :param img: np.ndarray 切り出した牌画像
    :param tile: int パターンマッチングの分類結果(数牌の五)
    :param thresh_val:
    :param thresh_sum:
    :return: int 牌番号
    """
    # 雀卓の緑色を黒にする
    loc_desk = (img[:, :, 0] > 90) * (img[:, :, 0] < 220) * \
               (img[:, :, 1] > 80) * (img[:, :, 1] < 220) * \
               (img[:, :, 2] > 40) * (img[:, :, 2] < 120)
    img[loc_desk] = 0
    # Rが(G+B)/2をthresh以上超えているピクセルの数(奥牌は図柄面積が小さくなるので比率より数とした)
    RoverGB = (img[:, :, 2] - (img[:, :, 0].astype(np.float) + img[:, :, 1].astype(np.float)) / 2) > thresh_val

    if RoverGB.sum() > thresh_sum:  # 赤と識別
        if tile % 10 == 5:  # もともと黒と予測していたら赤に置き換える
            tile += 5
    else:  # 黒と識別
        if tile % 10 == 0:  # もともと赤と予測していたら黒に置き換える
            tile -= 5

    return tile


def recognize_tile(img, templates, tmp_num_per_tile, tile_kind=37):
    """テンプレートマッチング
    :param img: 切り出した牌画像
    :param templates: テンプレートのリスト
    :return: int 牌番号
    """
    ress = []
    ress_idxs = []
    tmp_shapes = []
    # 各牌
    for tile_i in range(1, tile_kind+1):
        res_ = []
        res_idx_ = []
        tmp_shape_ = []
        # 各テンプレート
        for tmp_i in range(tmp_num_per_tile):
            template = templates[tmp_num_per_tile*(tile_i-1)+tmp_i]
            # テンプレートマッチング
            res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
            res_.append(res.max())# 各テンプレートの類似度最大値
            res_idx_.append(np.unravel_index(np.argmax(res), res.shape))
            tmp_shape_.append(template.shape)
        ress.append(max(res_))# 各牌の類似度最大値
        ress_idxs.append(res_idx_[res_.index(max(res_))])
        tmp_shapes.append(tmp_shape_[res_.index(max(res_))])

    tile = ress.index(max(ress)) + 1

    # 赤黒牌識別
    if tile%5==0 and tile<=30:#予測が5m,r5m,5p,r5p,5s,r5s
        idx = ress_idxs[ress.index(max(ress))]
        tmp_shape = tmp_shapes[ress.index(max(ress))]
        img_ = img[idx[0]:idx[0]+int(tmp_shape[0]/2), idx[1]:idx[1]+tmp_shape[1]]#牌の上半分だけで評価する(萬子対応)
        tile = red_or_black(img_, tile)

    return tile


def write_tiles_recognition(img, tiles_LRTB, tiles_x, tiles_y, tiles, save_name=None, tiles_dict=tiles_dict):
    """推定結果の書き出し
    """
    color = (0, 255, 0)
    for i in range(len(tiles_x) - 1):
        # 縦線
        pt1 = (tiles_x[i], tiles_y[i])
        pt2 = (tiles_x[i], tiles_y[i] - 20)
        cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
        # 横線
        pt1 = (tiles_x[i], tiles_y[i])
        pt2 = (tiles_x[i] + 15, tiles_y[i])
        cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
        # 牌名背景
        img[tiles_y[i]+1:tiles_y[i]+12, tiles_x[i]:tiles_x[i] +21, :] = 0
    for i in range(len(tiles_x) - 1):
        # 牌名
        pt1 = (tiles_x[i]+1, tiles_y[i] + 8)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, tiles_dict[tiles[i]], pt1, font, fontScale=0.3, color=color, thickness=1, lineType=cv2.LINE_AA)

    # 右端の縦線
    pt1 = (tiles_x[-1], tiles_y[-1])
    pt2 = (tiles_x[-1], tiles_y[-1] - 20)
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)

    # LRTB
    color = (0, 0, 255)
    # 縦線
    pt1 = (tiles_LRTB[0], tiles_LRTB[3])
    pt2 = (tiles_LRTB[0], tiles_LRTB[3] + 4)
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
    # 横線
    pt1 = (tiles_LRTB[0], tiles_LRTB[3])
    pt2 = (tiles_LRTB[0] - 10, tiles_LRTB[3])
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
    # 縦線
    pt1 = (tiles_LRTB[1], tiles_LRTB[2])
    pt2 = (tiles_LRTB[1], tiles_LRTB[2] + 4)
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)
    # 横線
    pt1 = (tiles_LRTB[1], tiles_LRTB[2])
    pt2 = (tiles_LRTB[1] + 10, tiles_LRTB[2])
    cv2.line(img, pt1, pt2, color, thickness=1, lineType=cv2.LINE_8, shift=0)

    if save_name is not None:
        cv2.imwrite(save_name, img)

    return img