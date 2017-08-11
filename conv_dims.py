from keras.models import Sequential
from keras.layers import Conv2D

model = Sequential()
# K*F*F*D_in + K = 16*2*2*1 + 16 = 80
# model.add(Conv2D(filters=16, kernel_size=2, strides=2, padding='valid', 
#    activation='relu', input_shape=(200, 200, 1)))



# K*F*F*D_in + K = 32*3*3*3 + 32 = 896
# activation maps: depth=32, width=ceil(h_in/s)=ceil(128/2)=64, width=64

model.add(Conv2D(filters=32, kernel_size=3, strides=2, padding='same', 
    activation='relu', input_shape=(128, 128, 3)))

model.summary()
