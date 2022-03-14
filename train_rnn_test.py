import sys
import os
import numpy as np
import matplotlib
# matplotlib.use('agg')
from matplotlib import pyplot as plt
# from IPython.display import clear_output
import tensorflow as tf
# physical_devices = tf.config.experimental.list_physical_devices('GPU')
# assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
# for _device in physical_devices:
#     config = tf.config.experimental.set_memory_growth(_device, True)

from predict import predict

# max_epochs = int(sys.argv[1])

si = predict(fname='../public_html/full_histories_illustris.h5')
si.training_mask()

sn = 50
predictors = si.load_arr('SFH/log_8')
illustris_dust, wl = si.load_spectra('Dust')
# illustris_noise = np.random.normal(loc=0, scale=illustris_dust / sn)
# illustris_final_dust_noise = illustris_dust + illustris_noise
# si.generate_standardisation(key='Dust Noise SN50', spec=illustris_final_dust_noise)
si.generate_standardisation(key='Dust', spec=illustris_dust)
features = si.prepare_features(illustris_dust, key='Dust', RNN=True)


train = np.random.rand(len(features)) > 0.2 

## split out predictors into sequences
N_bins = predictors.shape[1]
N_predictors = len(predictors)
predictors_reshape = np.full(((N_bins-1)*N_predictors, (N_bins-1)), -1)
predictors_y = np.full((N_predictors*(N_bins-1)), -1.)
for j in np.arange(N_predictors):
    for i,p in enumerate(predictors[j][:-1]):
        predictors_reshape[(j*(N_bins-1))+i][:(i+1)] = predictors[j][:(i+1)]
        predictors_y[(j*(N_bins-1))+i] = predictors[j,i]


train_reshape = np.full(N_predictors*(N_bins-1), False)
for i in np.where(train)[0]:
    train_reshape[i*(N_bins-1):(i*(N_bins-1))+(N_bins-1)] = True


features_reshape = np.repeat(features,7, axis=0)



## build the model
tf.keras.backend.clear_session()

inputs1 = tf.keras.layers.Input(shape=(features.shape[1],))
fe1 = tf.keras.layers.Dense(32, activation='sigmoid')(inputs1)

inputs2 = tf.keras.layers.Input(shape=(N_bins-1,1))
se1 = tf.keras.layers.Masking(mask_value=-1, input_shape=(N_bins-1,1))(inputs2)
se2 = tf.keras.layers.LSTM(32, input_shape=(N_bins-1,))(se1)

decoder1 = tf.keras.layers.add([fe1, se2])
decoder2 = tf.keras.layers.Dense(256, activation='relu')(decoder1)
outputs = tf.keras.layers.Dense(1, activation='relu')(decoder2)

model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)

# model.compile(optimizer='adam', loss=si._SMAPE_tf)
model.compile(optimizer='adam', loss='MeanSquaredLogarithmicError')


model.fit([features_reshape[train_reshape],predictors_reshape[train_reshape]], 
           predictors_y[train_reshape], epochs=10, verbose=1)

model.evaluate([features_reshape[~train_reshape], predictors_reshape[~train_reshape]], 
    predictors_y[~train_reshape], verbose=0)



prediction = model.predict([features_reshape[~train_reshape], predictors_reshape[~train_reshape]])

plt.scatter(predictors_y[~train_reshape], prediction, s=1)
plt.show()


fig,ax = plt.subplots(1,1)
ax.scatter(predictors_y[~train_reshape], prediction[:,0] / predictors_y[~train_reshape])
ax.set_yscale('log')
plt.show()

i = 0
plt.plot(np.arange(8), predictors[i], label='True')
plt.plot(np.arange(7)+1, prediction[:,0][i*(N_bins-1):(i*(N_bins-1))+(N_bins-1)], label='prediction')
plt.legend()
plt.show()




# def create_rnn_model(self, features, predictors, batch_size=10, train=None, plot=True, 
#                      max_epochs=1000, loss=None, verbose=True, fit=True, learning_rate=0.0007):
# 
#     if train is None:
#         if self.train is None: raise ValueError('No training mask initialised')
#         train = self.train
# 
#     if loss is None:
#         loss = self._SMAPE_tf
# 
# 
#     input_dim = features.shape[1:]
#     out_dim = predictors.shape[1]
# 
#     model = Sequential()
#     model.add(GRU(16, input_shape=input_dim, return_sequences=True))
#     model.add(GRU(16, return_sequences=False))
# 
#     model.add(Dense(out_dim,
#                     kernel_initializer='normal',
#                     kernel_constraint=NonNeg())
#               )
# 
#     lr = learning_rate #default 0.0007
#     beta_1 = 0.9
#     beta_2 = 0.999
# 
#     optimizer = Adam(lr=lr, beta_1=beta_1, beta_2=beta_2, decay=0.0)
# 
#     model.compile(loss=loss,
#                   optimizer=optimizer,
#                   metrics=['mae','mse','accuracy'])
# 
#     # validation split is unshuffled, so need to pre-shuffle before training
#     mask = np.random.permutation(np.sum(train))
# 
#     history = model.fit(features[train][mask], predictors[train][mask],
#                         epochs=max_epochs,
#                         batch_size=batch_size, validation_split=0.2,
#                         verbose=verbose)
#     
#     score, mae, mse, acc = model.evaluate(features[~train], predictors[~train], verbose=0)
#     return model, {'loss': score, 'mse': mse, 'mae': mae, 'acc': acc, 'history': history}
#     
# 
# # model,scores = si.create_rnn_model(features, predictors, batch_size=100, train=si.train, 
# #                                    learning_rate = 0.07, max_epochs = max_epochs)
# 
# 
# f = 'data/rnn_trained_illustris_dust.h5'
# if os.path.isfile(f): os.remove(f)
# model.save(f)
