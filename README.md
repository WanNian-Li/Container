# Container

训练模型：train_genal.py

创建数据集：data_create_old.py（不是data_create.py）

绘制训练结果图：plot.py

### model文件夹

* Module.py是可自适应集装箱形状的模型文件，里面包含了CNN和LSTM两个模型
* CNN76.py、CNN104.py、LSTM76.py、LSTM104.py是过去的模型文件，每一个.py只能适应一种集装箱形状，现在不再使用这些模型

### fig文件夹

* 存放了训练结果图，由plot.py生成

### result文件夹

* 存放了训练结果
* 文件命名格式为{stack}-{tier}-{<model_name>}
