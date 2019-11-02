from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.losses import Huber
import numpy as np

class DQN():
    def __init__(self, K, conv_layer_sizes, hidden_layer_sizes, scope, image_size):
        self.K = K
        self.X = Input(shape=[image_size,image_size,4])
        Z = self.X / 255.0
        Z = BatchNormalization()(Z)
        Z = Conv2D(32, 8,input_shape=(image_size, image_size, 4), activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(4, 4))
        Z = BatchNormalization()(Z)
        Z = Conv2D(64, 4, activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(2, 2))
        Z = BatchNormalization()(Z)
        Z = Conv2D(64, 3, activation='relu')(Z)
        Z = Flatten()(Z)
        Z = Dense(512, activation='relu')(Z)
        self.predict_op = Dense(self.K)(Z)
        
        self.model = Model(inputs=self.X, outputs=self.predict_op)
        self.loss_object  = Huber(from_logits = True)
        
        
        
        self.train_op = tf.train.AdamOptimizer(1e-5)
            
    def copy_from(self, other):
        self.model.set_weights(other.get_weights()) 
    
    def save(self):
        self.model.save_weights('model_weights.h5')
    
    def load(self):
        self.model.load_weights('model_weights.h5')
        
    def predict(self, states):
        return self.model.predict(states)
    
    @tf.function
    def train_step(self,states, actions, targets, inputs, labels):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            selected_action_value = tf.reduce_sum(predictions * tf.one_hot(actions,self.K), reduction_indices=[1])
            cost = tf.reduce_mean(self.loss_object(targets, selected_action_value))
        gradients = tape.gradient(cost, self.model.trainable_variables)
        self.train_op.apply_gradients(zip(gradients, self.model.trainable_variables))
      
    def sample_action(self,x,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            return np.argmax(self.predict([x])[0])