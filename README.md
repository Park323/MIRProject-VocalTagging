# kpop vocal tagging
Final Project of Deep Learning for Music and Audio
---
semantic tagging with KVT Dataset

## Directory
└ __data__\
&nbsp;&nbsp;&nbsp;&nbsp;└ ???\
└ utils.py : Loss, Metric 등을 불러오기 위해 정의된다.\
└ dataset.py : dataset을 정의한다.\
└ model.py : semantic tagging 모델을 정의한다.\
└ train.py : dataloader를 불러와 model로 학습을 진행한다.\
└ test.py : 학습된 model을 불러오고, inference 내용을 출력한다.\

## Model
- base_model

![image](https://user-images.githubusercontent.com/42057488/169036325-27564cdf-7e90-4dca-a42a-7050734b4e00.png)


### train example
train with learning rate 0.0001 for 5 epochs
```
python train.py -learning_rate 0.0001 -epoch 5
```

### demo result
![image](https://user-images.githubusercontent.com/42057488/174925754-2f41a1e9-2662-47b4-89a5-f43b256a7505.png)

![image](https://user-images.githubusercontent.com/42057488/174925866-0c526faa-966d-4024-b9d3-b3e159fd559b.png)
