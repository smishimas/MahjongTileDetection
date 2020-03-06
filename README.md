# Mahjong Tile Detection
https://qiita.com/smishima/items/4f79eeb98537bd133e2a

## Description
- Detect a position of a 13-or-14-tile hand from M-league broadcast 
- Crop areas of each tile in the hand and classify them with template matching or CNN


## Requirement
- python                             3.6.4
- Keras                              2.2.0
- matplotlib                         2.1.2
- numpy                              1.18.1
- opencv-python                      4.0.0.21
- tensorflow                         1.14.0


## Installation
    $ git clone https://github.com/smishimas/MahjongTileDetection.git
You need to make template images of each tile in 'images/template/'.

## Demonstration
    $ python3 tile_detection_cnn.py
A result image will be saved in 'images/sample/'.

## Author
[@_smishima_](https://twitter.com/_smishima_)
