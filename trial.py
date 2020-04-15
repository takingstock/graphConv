import tensorflow as tf
import tensorflow.keras as K
import numpy as np

class MyDenseLayer(tf.keras.layers.Layer):
  def __init__(self, num_outputs):
    super(MyDenseLayer, self).__init__()
    self.num_outputs = num_outputs

    self.kernel = self.add_weight("kernel",
                                  shape=[3,
                                         self.num_outputs])

  def call(self, input):
    return tf.nn.relu( tf.matmul(input, self.kernel) )

inp = K.Input(shape=(3))
layer = MyDenseLayer(10)(inp)
dense = K.layers.Dense(1, activation="sigmoid")(layer)

adam = K.optimizers.Adam(lr=0.0001)
model = K.models.Model( inp , dense , name='masaladosa')
model.compile( optimizer=adam , loss='binary_crossentropy', metrics=['accuracy'] )

model.fit( x=[ [ [1,2,3], [10,20,30] ] ], y=[[0,1]] , epochs=10 , verbose=1 )
print( model.predict( [ [ [ 1, 3, 4], [11, 21, 31] ] ] ) )
