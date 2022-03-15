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
bins = si.load_arr('bins/log_8/bins')
illustris_dust, wl = si.load_spectra('Dust')
# illustris_noise = np.random.normal(loc=0, scale=illustris_dust / sn)
# illustris_final_dust_noise = illustris_dust + illustris_noise
# si.generate_standardisation(key='Dust Noise SN50', spec=illustris_final_dust_noise)
si.generate_standardisation(key='Dust', spec=illustris_dust)
features = si.prepare_features(illustris_dust, key='Dust', RNN=True)


train = np.random.rand(len(features)) > 0.2 

## split out predictors into sequences
N_bins = len(bins)
N_predictors = len(predictors)
predictors_reshape = np.full((N_bins*N_predictors, N_bins), -1.)
predictors_y = np.full((N_predictors*N_bins), -1.)
for j in np.arange(N_predictors):
    for i,p in enumerate(predictors[j][:-1]):
        predictors_reshape[(j*N_bins)+(i+1)][:(i+1)] = predictors[:,::-1][j][:(i+1)]
    for i,p in enumerate(predictors[j]):
        predictors_y[(j*N_bins)+i] = predictors[:,::-1][j,i]


train_reshape = np.full(N_predictors*N_bins, False)
for i in np.where(train)[0]:
    train_reshape[i*N_bins:(i*N_bins)+N_bins] = True


features_reshape = np.repeat(features,8,axis=0)



def _SMAPE_tf(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    return 2 * tf.reduce_sum(tf.abs(y_pred - y_true), axis=-1) / \
            tf.reduce_sum(y_pred + y_true, axis=-1)




## build the model
tf.keras.backend.clear_session()

inputs1 = tf.keras.layers.Input(shape=(features.shape[1],))
fe1 = tf.keras.layers.Dense(16, activation='sigmoid')(inputs1)
# fe2 = tf.keras.layers.Dense(16, activation='sigmoid')(inputs1)
# fe1 = tf.keras.layers.Dense(, activation='sigmoid')(inputs1)

inputs2 = tf.keras.layers.Input(shape=(N_bins,1))
se1 = tf.keras.layers.Masking(mask_value=-1, input_shape=(N_bins,1))(inputs2)
se2 = tf.keras.layers.LSTM(16, input_shape=(N_bins,))(se1)

decoder1 = tf.keras.layers.add([fe1, se2])
decoder2 = tf.keras.layers.Dense(32, activation='relu')(decoder1)
outputs = tf.keras.layers.Dense(1, activation='relu')(decoder2)

model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)

# model.compile(optimizer='adam', loss=_SMAPE_tf)
# model.compile(optimizer='adam', loss='MeanSquaredLogarithmicError')
model.compile(optimizer='adam', loss='mse')

early_stopping_min_delta = 1e-4
early_stopping_patience = 6

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                           min_delta=early_stopping_min_delta,
                           patience=early_stopping_patience,
                           verbose=2, mode='min')

reduce_lr_patience = 4
reduce_lr_min = 0.0

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                          factor=0.5,
                          patience=reduce_lr_patience,
                          min_lr=reduce_lr_min,
                          mode='min', verbose=2)

model.fit([features_reshape[train_reshape],predictors_reshape[train_reshape]], 
           predictors_y[train_reshape], epochs=50, verbose=1, callbacks=[early_stopping,reduce_lr], 
           batch_size=1000)

model.evaluate([features_reshape[~train_reshape], predictors_reshape[~train_reshape]], 
    predictors_y[~train_reshape], verbose=0)

prediction = model.predict([features_reshape[~train_reshape], predictors_reshape[~train_reshape]])



## comparison model
tf.keras.backend.clear_session()

inputs1 = tf.keras.layers.Input(shape=(features.shape[1],))
fe1 = tf.keras.layers.Dense(64, activation='sigmoid')(inputs1)
fe2 = tf.keras.layers.Dense(32, activation='sigmoid')(fe1)
fe3 = tf.keras.layers.Dense(32, activation='sigmoid')(fe2)
outputs = tf.keras.layers.Dense(N_bins, activation='relu')(fe3)
model = tf.keras.models.Model(inputs=inputs1, outputs=outputs)
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss=_SMAPE_tf)
model.fit(features[train], predictors[train], epochs=200, verbose=1, 
          callbacks=[early_stopping,reduce_lr])
prediction_comp = model.predict(features[~train])




## 1:1 scatter plot
fig, ax = plt.subplots(1,1)
ax.scatter(predictors_y[~train_reshape], prediction, s=1, label='RNN')
ax.scatter(predictors_y[~train_reshape], prediction_comp.reshape(-1), s=1, label='ANN')
ax.legend()
# plt.show()
plt.savefig('1_to_1.png'); plt.close()

## relative error
fig,ax = plt.subplots(1,1)
ax.scatter(predictors_y[~train_reshape], 
           prediction[:,0] / predictors_y[~train_reshape], alpha=0.5, label='RNN', s=1)
ax.scatter(predictors_y[~train_reshape], 
           prediction_comp.reshape(-1) / predictors_y[~train_reshape], alpha=0.5, label='ANN', s=1)
ax.legend()
ax.set_yscale('log')
# plt.show()
plt.savefig('relative_error.png'); plt.close()

# ## bin-by-bin relative error
fig,ax = plt.subplots(1,1)
for i,b in enumerate(bins):
    ax.scatter((np.random.rand(np.sum(~train))/2)+i+0.5, 
               (prediction[:,0].reshape((-1,8)) / predictors[~train])[:,i], s=1, c='C%i'%i)
    ax.scatter((np.random.rand(np.sum(~train))/2)+i+1, 
               (prediction_comp / predictors[~train])[:,i], s=1, c='C%i'%i)
    ax.vlines(i+1, 10**-3, 10**3, linestyle='dashed',color='black')
    ax.vlines(i+0.5, 10**-3, 10**3, linestyle='solid',color='black')
ax.set_xticklabels(np.hstack([0.,["%0.2f"%b for b in bins][::-1]]))
ax.set_yscale('log')
ax.legend()
# plt.show()
plt.savefig('bin_relative_error.png'); plt.close()


## example history
i = 100
fig,ax = plt.subplots(1,1)
ax.step(bins, predictors[i], label='True', where='mid')
ax.step(bins, prediction.reshape((-1,8))[i][::-1], label='prediction RNN', where='mid')
ax.step(bins, prediction_comp[i][::-1], label='prediction ANN', where='mid')
ax.legend()
ax.set_xlim(7.5,9.85)
# plt.show()
plt.savefig('example_history.png'); plt.close()




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
