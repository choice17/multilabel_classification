from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout
from tensorflow.keras.layers import Dense, Input, Flatten, BatchNormalization
from tensorflow.keras.layers import ReLU, Add, GlobalMaxPooling2D, Activation
from tensorflow.keras.models import Model
INPUT_DIM = (64,64,3)
class MyModel(object):
    def getBase(self, inputs):
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
        x = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='linear')(inputs)
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x1 = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='linear')(x1)
        x = Add()([x1, x])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        x1 = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='relu')(x1)
        x = Conv2D(128, (3, 3), strides=(1, 1), padding='same', activation='linear')(x)
        x = Add()([x1, x])
        x = BatchNormalization()(x)
        x = ReLU()(x)
        x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)
        return x

    def getTop(self, inputs):
        x = Conv2D(256, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputs)
        x = Conv2D(128, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        x = Conv2D(256, (1, 1), strides=(1, 1), padding='same', activation='relu')(x)
        return x

    def set(self, input_dim=INPUT_DIM, num_class=6, activation=activation):
        inputs = Input(shape=input_dim)
        x = self.getBase(inputs)
        x = self.getTop(x)
        x = Conv2D(num_class, (1, 1), strides=(1, 1), padding='same', activation='linear')(x)
        x = GlobalMaxPooling2D()(x)
        logits = Flatten()(x)
        prediction = Activation(activation)(logits)
        return Model(inputs, prediction)
