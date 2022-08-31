import tensorflow as tf


#加载数字图片
def process_data():
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()#下载图片
    print(x_train.shape)


if __name__ == "__main__":
    process_data()