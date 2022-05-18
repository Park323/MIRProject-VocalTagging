# MIRProject-VocalTagging
Deep Learning for Music and Audio - Final Project
---
KVT dataset을 사용하여 semantic tagging을 scratch에서부터 구현해보고,
수업에서 배운 내용을 적용하여 성능을 비교해본다.

## Directory
└ __data__\
&nbsp;&nbsp;&nbsp;&nbsp;└ ???\
└ dataloader.py : dataset을 처리하고 dataloader를 정의한다.\
└ model.py : semantic tagging 모델을 정의한다.\
└ train.py : dataloader를 불러와 model로 학습을 진행한다.\
└ test.py : 학습된 model을 불러오고, inference 내용을 출력한다.\

## Model
- base_model

![image](https://user-images.githubusercontent.com/42057488/169036325-27564cdf-7e90-4dca-a42a-7050734b4e00.png)
