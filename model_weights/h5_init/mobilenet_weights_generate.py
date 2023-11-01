from tensorflow import keras


if __name__ == '__main__':
    model = keras.applications.MobileNet(
        input_shape=(512, 512, 3),
        weights=None,
    )
    model.load_weights('./mobilenet_1_0_224_tf.h5')
    # model_weights.summary()

    model2 = keras.Model(model.input, model.get_layer('conv_pw_11_relu').output)
    # model2.save('../base_mobilenet_weights.h5')
    model2.summary()