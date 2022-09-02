import os
import matplotlib.pyplot as plt
import tensorflow as tf
from PIL import Image
model_save_path = './save_file/mnist.ckpt'
import numpy as np



# 加载数字图片
def process_data():
    mnist = tf.keras.datasets.mnist
    # 下载图片
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # 归一化处理
    x_train = x_train/255.0
    x_test = x_test / 255.0
    return x_train, y_train, x_test, y_test


# 展示部分图片
def show_image(x_train):
    # 新建一个画布
    plt.figure()
    for i in range(100):
        plt.subplot(10, 10, i+1)
        plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.show()


# 设计神经网络
def nn_model(x_train, y_train, x_test, y_test):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),  # 拉直层
        tf.keras.layers.Dense(256, activation="sigmoid"),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(64, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="softmax")
        ])  # 定义顺序网络框架
    # 为模型配置优化方法
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy",  # 交叉熵代价函数
                  metrics="sparse_categorical_accuracy")


    """""""""""""""""""""""""""""""""
    1.2第一次会保存模型
    再运行时，会加载该模型，fit时会以加载的模型为基准，继续往下训练
    """""""""""""""""""""""""""""""""
    cp_callback = save_model(model) #保存模型
    # 训练模型
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,  # 迭代次数
              validation_data=(x_test, y_test),
              validation_freq=1,
              callbacks=[cp_callback]
              )
    # 模型评估
    loss, acc = model.evaluate(x_test, y_test)
    print("loss: ", loss)
    print("acc: ", acc)
    # 获取计算过程
    model.summary()
    return model


# 模型存储和加载（具有断点续训功能，从上一次训练的节点继续训练）
def save_model(model):
    checkpoint_save_path = "./save_file/mnist.ckpt"
    if os.path.exists(checkpoint_save_path + '.index'):
        print('模型加载中......')
        model.load_weights(checkpoint_save_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_save_path,
                                                     save_weights_only=True,
                                                     save_best_only=True)
    return cp_callback


# 绘图预测
def testing():
    # 模型设计
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(),  # 拉直层
        tf.keras.layers.Dense(256, activation="sigmoid"),
        tf.keras.layers.Dense(128, activation="sigmoid"),
        tf.keras.layers.Dense(64, activation="sigmoid"),
        tf.keras.layers.Dense(10, activation="softmax")
    ])
    # 加载参数
    model.load_weights(model_save_path)
    preNum = int(input("请输入要预测的图片数量："))

    for i in range(preNum):
        image_name = input("输入图片名称：")
        img = Image.open("./images/"+image_name)  # 加载图片
        img = img.resize((28, 28),Image.ANTIALIAS)  # 28*28大小的图片
        img_arr = np.array(img.convert('L'))  # 转换为灰度图片

        for i in range(28):#转为黑底白字
            for j in range(28):
                if img_arr[i][j] < 200:
                    img_arr[i][j] = 255
                else:
                    img_arr[i][j] = 0

        img_arr = img_arr / 255.0#归一化
        #预测时，输入的图片维度要是[图片数量,图片高，图片宽]
        x_predict = img_arr[tf.newaxis, ...]#tf.newaxis表示增加一个维度，即[1,28,28]
        result = model.predict(x_predict)
        pred = tf.argmax(result, axis=1)
        tf.print("该图片中的数字预测为：",pred)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = process_data()
    # show_image(x_train)
    model = nn_model(x_train, y_train, x_test, y_test)
    testing()