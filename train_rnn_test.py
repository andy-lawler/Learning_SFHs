# import sys
# import os
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

import schwimmbad
from functools import partial

from predict import predict

# max_epochs = int(sys.argv[1])

si = predict(fname='../public_html/full_histories_illustris.h5')
si.training_mask()


# ## finer binning
# upperBin = np.log10(si.cosmo.age(si.redshift).value * 1e9)
# binLimits = np.hstack([[0.0], np.linspace(7.5, upperBin, num=32)])
# binWidths = 10**binLimits[1:] - 10**binLimits[:len(binLimits)-1]
# bins = binLimits[:-1] + ((binLimits[1:] - binLimits[:-1]) / 2)
# binLimits = 10**binLimits / 1e9
# custom = {'binLimits': binLimits, 'bins': bins, 'binWidths': binWidths}
# 
# bins, binLimits, binWidths = si.init_bins(name='log_32', custom=custom, verbose=True)
# 
# import schwimmbad
# from functools import partial
# shids = si.load_arr('Subhalos/ID')
# pool = schwimmbad.choose_pool()# processes=args.n_cores)
# lg = partial(si.bin_histories, binLimits=binLimits, binWidths=binWidths)
# sfh = np.array(list(pool.map(lg,shids)))
# pool.close()
# # sfh = si.bin_histories(shids, binLimits=binLimits, binWidths=binWidths)
# si.save_arr(sfh,'log_32','SFH')

_bin_no = 8
predictors = si.load_arr('SFH/log_%i'%_bin_no)
bins = si.load_arr('bins/log_%i/bins'%_bin_no)


illustris_dust, wl = si.load_spectra('Dust')
# sn = 50
# illustris_noise = np.random.normal(loc=0, scale=illustris_dust / sn)
# illustris_final_dust_noise = illustris_dust + illustris_noise
# si.generate_standardisation(key='Dust Noise SN50', spec=illustris_final_dust_noise)
si.generate_standardisation(key='Dust', spec=illustris_dust)
features = si.prepare_features(illustris_dust, key='Dust', RNN=True)


# train = np.random.rand(len(features)) > 0.2 
# np.savetxt('train.txt',train)
train = np.array(np.loadtxt('train.txt'), dtype=bool)


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


features_reshape = np.repeat(features,N_bins,axis=0)



def _SMAPE_tf(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    return 2 * tf.reduce_sum(tf.abs(y_pred - y_true), axis=-1) / \
            tf.reduce_sum(y_pred + y_true, axis=-1)


def _SMAPE_tf_single(y_true, y_pred):
    """
    Symmetric Mean Absolute Percentage Error
    https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    """
    # y_true += 0.01
    # y_pred += 0.01
    return tf.multiply(2.,tf.divide( tf.abs(tf.subtract(y_pred, y_true)), tf.add(y_true, y_pred)))





## build the model
tf.keras.backend.clear_session()

inputs1 = tf.keras.layers.Input(shape=(features.shape[1],))
fe1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal')(inputs1)
fe2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal')(fe1)
# fe3 = tf.keras.layers.Dense(32, activation='sigmoid')(fe2)

inputs2 = tf.keras.layers.Input(shape=(N_bins,1))
se1 = tf.keras.layers.Masking(mask_value=-1, input_shape=(N_bins,1))(inputs2)
se2 = tf.keras.layers.LSTM(32, input_shape=(N_bins,))(se1)

decoder1 = tf.keras.layers.add([fe2, se2])
decoder2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal')(decoder1)
outputs = tf.keras.layers.Dense(1, kernel_constraint=tf.keras.constraints.NonNeg(),
                                kernel_initializer='normal' )(decoder2)

model = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)

optimizer = tf.keras.optimizers.Adam()#clipnorm=1, clipvalue=1)
model.compile(optimizer=optimizer, loss=_SMAPE_tf_single)
# model.compile(optimizer='adam', loss='mse')
# model.compile(optimizer='adam', loss='mean_absolute_percentage_error')

early_stopping_min_delta = 1e-4
early_stopping_patience = 8

early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss',
                           min_delta=early_stopping_min_delta,
                           patience=early_stopping_patience,
                           verbose=2, mode='min')

reduce_lr_patience = 6
reduce_lr_min = 0.0

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss',
                          factor=0.5,
                          patience=reduce_lr_patience,
                          min_lr=reduce_lr_min,
                          mode='min', verbose=2)

model.fit([features_reshape[train_reshape],predictors_reshape[train_reshape]], 
           predictors_y[train_reshape], epochs=500, verbose=1, callbacks=[early_stopping,reduce_lr], 
           batch_size=1000)

model.save('RNN_test_bin%i.model'%_bin_no)
model = tf.keras.models.load_model('RNN_test_bin%i.model'%_bin_no, 
             custom_objects={"_SMAPE_tf_single": _SMAPE_tf_single})


# print("Evaluate:",model.evaluate([features_reshape[~train_reshape], 
#                                   predictors_reshape[~train_reshape]], 
#                                  predictors_y[~train_reshape], verbose=0))

# prediction = model.predict([features_reshape[~train_reshape], predictors_reshape[~train_reshape]])

def predict_history(_features, _bin_no):
    predictors_temp = (np.ones(_bin_no) * -1).T
    for i in np.arange(_bin_no):
        _pred_obin = model.predict([np.expand_dims(_features,0),
                                    np.expand_dims(predictors_temp,0)])

        predictors_temp[i] = _pred_obin

    return predictors_temp


# pool = schwimmbad.choose_pool(processes=8)
# lg = partial(predict_history, _bin_no=_bin_no)
# prediction = np.array(list(pool.map(lg,features[~train])))
# pool.close()

prediction = np.array([predict_history(_f, _bin_no) for _f in features[~train]])
np.savetxt('rnn_prediction_bin%i.txt'%_bin_no, prediction)
prediction = np.loadtxt('rnn_prediction_bin%i.txt'%_bin_no)

def evaluate_model(predictors, predicted):
    return np.mean(_SMAPE_tf(predictors, predicted))


print("Evaluate:", evaluate_model(predictors[~train], prediction[:,::-1]))


## comparison model
tf.keras.backend.clear_session()

inputs1 = tf.keras.layers.Input(shape=(features.shape[1],))
fe1 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal')(inputs1)
fe2 = tf.keras.layers.Dense(32, activation='relu', kernel_initializer='normal')(fe1)
# fe3 = tf.keras.layers.Dense(32, activation='sigmoid')(fe2)
outputs = tf.keras.layers.Dense(N_bins, kernel_constraint=tf.keras.constraints.NonNeg(), 
                                kernel_initializer='normal')(fe2)
model = tf.keras.models.Model(inputs=inputs1, outputs=outputs)
# model.compile(optimizer='adam', loss='mse')
model.compile(optimizer='adam', loss=_SMAPE_tf)
model.fit(features[train], predictors[train], epochs=500, verbose=1, 
          callbacks=[early_stopping,reduce_lr])

prediction_comp = model.predict(features[~train])

model.save('ANN_test_bin%i.model'%_bin_no)
model = tf.keras.models.load_model('ANN_test_bin%i.model'%_bin_no,
         custom_objects={"_SMAPE_tf": _SMAPE_tf})

print("Evaluate:",model.evaluate(features[~train], predictors[~train], verbose=0))


## example histories
fig,axes = plt.subplots(2,4,figsize=(14,8))
# for i,ax in zip([0,10,100,200,500,1000,800,5],axes.flatten()):
for i,ax in zip([0,10,20,30,40,50,60,70],axes.flatten()):
    print(evaluate_model(predictors[~train][i], prediction[i][::-1]))
    ax.step(bins, predictors[~train][i], label='True', where='mid')
    ax.step(bins, prediction[i][::-1], label='prediction RNN', where='mid')
    ax.step(bins, prediction_comp[i], label='prediction ANN', where='mid')
    ax.set_ylim(0,)
    ax.set_xlim(7.5,9.8)
 
for ax in axes.flatten()[4:]: 
    ax.set_xlabel('$\mathrm{log_{10}}(t_{\mathrm{L}} \,/\, \mathrm{Gyr})$')
for ax in axes.flatten()[[0,4]]: 
    ax.set_ylabel('$\mathrm{SFR} \,/\, M_{\odot} \mathrm{yr^{-1}}$')


axes.flatten()[0].legend()
plt.show()
# plt.savefig('example_histories.png'); plt.close()



## 1:1 scatter plot
fig, ax = plt.subplots(1,1)
ax.scatter(predictors_y[~train_reshape], prediction.flatten(), s=1, label='RNN')
ax.scatter(predictors_y[~train_reshape], prediction_comp[:,::-1].reshape(-1), s=1, label='ANN')
ax.legend()
ax.plot([0,200],[0,200],linestyle='dashed', color='black')
ax.set_xlim(0,160); ax.set_ylim(0,160)
ax.set_xlabel('SFR (True)')
ax.set_ylabel('SFR (Predicted)')
# plt.show()
plt.savefig('1_to_1.png'); plt.close()

## relative error
fig,ax = plt.subplots(1,1)
ax.scatter(predictors_y[~train_reshape], 
           prediction[:,0] / predictors_y[~train_reshape], alpha=0.5, label='RNN', s=1)
ax.scatter(predictors_y[~train_reshape], 
        prediction_comp[:,::-1].reshape(-1) / predictors_y[~train_reshape], alpha=0.5, label='ANN', s=1)
ax.legend()
ax.set_yscale('log')
# plt.show()
plt.savefig('relative_error.png'); plt.close()

# ## bin-by-bin relative error
fig,ax = plt.subplots(1,1)
for i,b in enumerate(bins):
    ax.scatter((np.random.rand(np.sum(~train))/2)+i+0.5, 
            (prediction[:,::-1] / predictors[~train])[:,i], s=1, c='C%i'%i)
    # ax.scatter((np.random.rand(np.sum(~train))/2)+i+1, 
    #            (prediction_comp / predictors[~train])[:,i], s=1, c='C%i'%i)
    ax.vlines(i+1, 10**-3, 10**3, linestyle='dashed',color='black')
    ax.vlines(i+0.5, 10**-3, 10**3, linestyle='solid',color='black')
ax.set_xticklabels(np.hstack([0.,["%0.2f"%b for b in bins][::-1]]))
ax.set_yscale('log')
ax.legend()
# plt.show()
plt.savefig('bin_relative_error.png'); plt.close()


## example history
fig,axes = plt.subplots(2,4,figsize=(20,8))
for i,ax in zip([0,10,100,200,500,1000,800,5],axes.flatten()):
    ax.step(bins, predictors[~train][i], label='True', where='mid')
    ax.step(bins, prediction.reshape((-1,N_bins))[i][::-1], label='prediction RNN', where='mid')
    ax.step(bins, prediction_comp[i], label='prediction ANN', where='mid')
    ax.legend()
    ax.set_xlim(7.5,10.0)
# plt.show()
plt.savefig('example_history.png'); plt.close()


