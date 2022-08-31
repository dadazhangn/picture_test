import matplotlib.pyplot as plt
import tensorflow as tf


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
    # 训练模型
    model.fit(x_train, y_train,
              batch_size=32,
              epochs=10,  # 迭代次数
              validation_data=(x_test, y_test),
              validation_freq=1
              )
    # 模型评估
    loss, acc = model.evaluate(x_test, y_test)
    print("loss: ", loss)
    print("acc: ", acc)


if __name__ == "__main__":

    x_train, y_train, x_test, y_test = process_data()
    # show_image(x_train)
    nn_model(x_train, y_train, x_test, y_test)