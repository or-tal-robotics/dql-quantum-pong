from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D, Input, BatchNormalization, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.losses import Huber
import numpy as np

class DQN():
    def __init__(self, K, image_size):
        self.K = K
        self.X = Input(shape=[image_size,image_size,4])
        Z = self.X / 255.0
        Z = Conv2D(32, 8,input_shape=(image_size, image_size, 4), activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(4, 4))(Z)
        Z = BatchNormalization()(Z)
        Z = Conv2D(64, 4, activation='relu')(Z)
        Z = MaxPooling2D(pool_size=(2, 2))(Z)
        Z = BatchNormalization()(Z)
        Z = Conv2D(64, 3, activation='relu')(Z)
        Z = Flatten()(Z)
        Z = Dense(512, activation='relu')(Z)
        self.predict_op = Dense(self.K)(Z)
        
        self.model = Model(inputs=self.X, outputs=self.predict_op)
        self.loss_object  = Huber()
        
        initial_learning_rate = 1e-6
        # lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        #     initial_learning_rate,
        #     decay_steps=100000,
        #     decay_rate=0.96,
        #     staircase=True)
        
        self.train_op = tf.keras.optimizers.Adam(initial_learning_rate)
            
    def copy_from(self, other):
        self.model.set_weights(other.model.get_weights()) 
    
    def save(self):
        self.model.save_weights('model_weights.h5')
    
    def load(self):
        self.model.load_weights('model_weights.h5')
        
    def predict(self, states):
        return self.model.predict(states)
    
    def get_weights(self):
        return np.reshape(self.model.get_weights()[-2], (-1))
    
    @tf.function
    def train_step(self,states, actions, targets):
        with tf.GradientTape() as tape:
            predictions = self.model(states, training=True)
            selected_action_value = tf.reduce_sum(predictions * tf.one_hot(actions,self.K), axis=1)
            cost = tf.reduce_mean(self.loss_object(targets, selected_action_value))
        gradients = tape.gradient(cost, self.model.trainable_variables)
        self.train_op.apply_gradients(zip(gradients, self.model.trainable_variables))
        return cost
      
    def sample_action(self,x,eps):
        if np.random.random() < eps:
            return np.random.choice(self.K)
        else:
            inp = tf.expand_dims(x, 0)
            return np.argmax(self.predict(inp)[0])