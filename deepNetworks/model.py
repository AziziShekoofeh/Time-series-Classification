from keras.callbacks import CSVLogger, Callback, ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

module_root = '..'
import sys
import os
from utils import settings as s
import json
import csv
import numpy as np
# import h5py

sys.path.append(module_root)
from sklearn.metrics import roc_auc_score


class AUCHistory(Callback):
    def __init__(self, validation_data):
        self.validation_d = validation_data[0]
        self.validation_l = validation_data[1]

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.validation_d)
        # print(np.concatenate((y_pred, self.validation_l), axis=1))
        logs['val_auc'] = roc_auc_score(self.validation_l, y_pred)
        print("- AUC: {0:0.2f}".format(logs['val_auc']))


class RNNModel:
    def __init__(self, data, model, params, log_dir, division=False, bmode=False):
        self.ds = data
        self.model = model
        self.log_dir = log_dir
        self.training_params_dict = params
        self.type = type
        self.bmode = bmode
        if not division:
            self.train_seq, self.train_label, self.validation_seq, self.validation_label, self.test_data,\
                self.test_label = self.ds.load_data(bmode=self.bmode)
        if division:
            self.train_seq, self.train_label, self.validation_seq, self.validation_label, self.test_data,\
                self.test_label = self.ds.load_data_split(bmode=self.bmode)

    def train(self, uid, batch_size, es, nb_epoch, verbose):
        print('-' * 30)
        print('Fitting model...')
        print('-' * 30)
        callbacks_list = []
        logs_dir = os.path.join(s.intermediate_folder, 'logs', self.log_dir)
        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)

        model_json = self.model.to_json()
        model_log_dir = os.path.join(logs_dir, 'model_logs')
        if not os.path.isdir(model_log_dir):
            os.mkdir(model_log_dir)

        with open(os.path.join(model_log_dir, uid + '.json'), 'w') as outfile:
            json.dump(model_json, outfile)

        train_log_dir = os.path.join(logs_dir, 'train_logs')
        if not os.path.isdir(train_log_dir):
            os.mkdir(train_log_dir)

        with open(os.path.join(train_log_dir, uid + '.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.training_params_dict.items():
                writer.writerow([key, value])

        validation_data = (self.validation_seq, self.validation_label[:])

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9,
                                      patience=15, min_lr=1e-9,
                                      epsilon=0.001, verbose=1)

        model_checkpoint_dir = os.path.join(s.intermediate_folder, 'model_checkpoints')
        if not os.path.exists(model_checkpoint_dir):
            os.mkdir(model_checkpoint_dir)

        model_checkpoint = ModelCheckpoint(os.path.join(model_checkpoint_dir, uid + '.hdf5'),
                                           monitor='val_acc', save_best_only=True)
        callbacks_list.append(model_checkpoint)
        history = AUCHistory(validation_data)
        callbacks_list.append(history)
        callbacks_list.append(reduce_lr)
        if es:
            es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2, verbose=1)
            callbacks_list.append(es)

        csv_log_dir = os.path.join(logs_dir, 'csv_logs')
        if not os.path.isdir(csv_log_dir):
            os.mkdir(csv_log_dir)
        csv_logger = CSVLogger(os.path.join(csv_log_dir, uid + '.log'))
        callbacks_list.append(csv_logger)

        callbacks_list.append(reduce_lr)

        self.model.fit(self.train_seq, self.train_label, batch_size=batch_size, epochs=nb_epoch, verbose=verbose,
                       shuffle=True, callbacks=callbacks_list, validation_data=validation_data)

    def predict_test(self, uid):
        test_seq, test_labels = self.ds.load_test(bmode=self.bmode)
        model_checkpoint_dir = os.path.join(s.intermediate_folder, 'model_checkpoints/opt')
        model_checkpoint_file = os.path.join(model_checkpoint_dir, uid + '.hdf5')
        self.model.load_weights(model_checkpoint_file)
        test_predictions = self.model.predict(test_seq, verbose=1)
        test_auc = roc_auc_score(test_labels, test_predictions)
        print(["Test AUC: ", test_auc])
        return test_predictions

    def opt_model_train(self, uid, batch_size, es, nb_epoch, verbose):

        # Re-define local train (train + validation) and test sequence
        # train_seq, train_label, test_seq, test_label = self.ds.load_train_test(bmode=self.bmode)
        train_seq = np.concatenate((self.train_seq, self.validation_seq))
        train_label = np.concatenate((self.train_label, self.validation_label))
        test_seq = self.test_data
        test_label = self.test_label
        print('-' * 30)
        print('Fitting optimum model ...')
        print('-' * 30)
        callbacks_list = []
        logs_dir = os.path.join(s.intermediate_folder, 'logs', self.log_dir)
        if not os.path.isdir(logs_dir):
            os.mkdir(logs_dir)

        model_json = self.model.to_json()
        model_log_dir = os.path.join(logs_dir, 'model_logs')
        if not os.path.isdir(model_log_dir):
            os.mkdir(model_log_dir)

        with open(os.path.join(model_log_dir, uid + '.json'), 'w') as outfile:
            json.dump(model_json, outfile)

        train_log_dir = os.path.join(logs_dir, 'train_logs')
        if not os.path.isdir(train_log_dir):
            os.mkdir(train_log_dir)

        with open(os.path.join(train_log_dir, uid + '.csv'), 'w') as csv_file:
            writer = csv.writer(csv_file)
            for key, value in self.training_params_dict.items():
                writer.writerow([key, value])

        test_data = (test_seq, test_label[:])

        reduce_lr = ReduceLROnPlateau(monitor='val_acc', factor=0.9,
                                      patience=15, min_lr=1e-9,
                                      epsilon=0.001, verbose=1)

        model_checkpoint_dir = os.path.join(s.intermediate_folder, 'model_checkpoints/opt')
        if not os.path.exists(model_checkpoint_dir):
            os.mkdir(model_checkpoint_dir)

        model_checkpoint = ModelCheckpoint(os.path.join(model_checkpoint_dir, uid + '.hdf5'),
                                           monitor='val_acc', save_best_only=True)
        callbacks_list.append(model_checkpoint)
        history = AUCHistory(test_data)
        callbacks_list.append(history)
        callbacks_list.append(reduce_lr)

        if es:
            es = EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=2, verbose=1)
            callbacks_list.append(es)

        csv_log_dir = os.path.join(logs_dir, 'csv_logs')
        if not os.path.isdir(csv_log_dir):
            os.mkdir(csv_log_dir)
        csv_logger = CSVLogger(os.path.join(csv_log_dir, uid + '.log'))
        callbacks_list.append(csv_logger)

        callbacks_list.append(reduce_lr)

        self.model.fit(train_seq, train_label, batch_size=batch_size, epochs=nb_epoch, verbose=verbose,
                       shuffle=True, callbacks=callbacks_list, validation_data=test_data)
