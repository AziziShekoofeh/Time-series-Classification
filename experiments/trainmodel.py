import sys
import time
# import os
from pprint import pprint as p
module_root = '..'
sys.path.append(module_root)
p(sys.path)
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from utils.data_loader import DataLoader
from deepNetworks.model import RNNModel
from deepNetworks.netArch import DeepNetArch1, DeepNetArch2, DeepNetArch3, DeepNetArch1L1, DeepNetArch2L1, DeepNetArch3L1
from keras import backend as k
# import tensorflow as tf

# config = tf.ConfigProto(allow_soft_placement=True)
# config.gpu_options.allow_growth = True
# session = tf.Session(config=config)
# k.set_session(session)

if __name__ == '__main__':
    log_dir = 'DeepNetArch3'
    early_stopping = False
    sl = 100
    validation_split = 0.2
    n_epoch = 100
    batch_sizes = [64, 128]
    initial_lrs = [1e-2, 1e-4]
    l2_regulizers = [0.0001, 0.0002]
    dropouts = [0, 0.4]
    rec_dropouts = [0]
    optimizers = ['sgd', 'rmsprop', 'adam']

    grid_size = len(batch_sizes) * len(initial_lrs) * len(l2_regulizers) * len(dropouts) * len(rec_dropouts) * len(
        optimizers)
    i = 1
    model_number = 40
    for batch_size in batch_sizes:
        for initial_lr in initial_lrs:
            for l2_regulizer in l2_regulizers:
                for dropout in dropouts:
                    for rec_dropout in rec_dropouts:
                        for optimizer in optimizers:

                            if i < model_number:
                                i += 1
                                continue

                            print('-' * 50)
                            print('-' * 50)
                            print('-' * 50)
                            print('batchsize:{}, initial_lr:{}, l2_regulizer:{}, dropout:{}, rec_dropout:{},'
                                  ' optimizer:{} '.format(
                                    batch_size, initial_lr, l2_regulizer, dropout, rec_dropout, optimizer))
                            print("experiment {} of total {}".format(i, grid_size))
                            ds = DataLoader(sl=sl, validation_split=validation_split)

                            if log_dir == 'DeepNetArch1':
                                arch = DeepNetArch1(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer, dropout=dropout,
                                                    rec_dropout=rec_dropout, optimizer=optimizer, summary=1)
                            if log_dir == 'DeepNetArch2':
                                arch = DeepNetArch2(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer, dropout=dropout,
                                                    rec_dropout=rec_dropout, optimizer=optimizer, summary=1)
                            if log_dir == 'DeepNetArch3':
                                arch = DeepNetArch3(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer, dropout=dropout,
                                                    rec_dropout=rec_dropout, optimizer=optimizer, summary=1)
                            if log_dir == 'DeepNetArch1L1':
                                arch = DeepNetArch1L1(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer,
                                                      dropout=dropout, rec_dropout=rec_dropout, optimizer=optimizer,
                                                      summary=1)
                            if log_dir == 'DeepNetArch2L1':
                                arch = DeepNetArch2L1(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer,
                                                      dropout=dropout, rec_dropout=rec_dropout, optimizer=optimizer,
                                                      summary=1)
                            if log_dir == 'DeepNetArch3L1':
                                arch = DeepNetArch3L1(sl=sl, initial_lr=initial_lr, l2_reg=l2_regulizer,
                                                      dropout=dropout, rec_dropout=rec_dropout, optimizer=optimizer,
                                                      summary=1)

                            model, model_id = arch.arch_generator()

                            params = dict()
                            params['batch_size'] = batch_size
                            params['initial_lr'] = initial_lr
                            params['l2_regulizer'] = l2_regulizer
                            params['dropout'] = dropout
                            params['rec_dropout'] = rec_dropout
                            params['n_epoch'] = n_epoch
                            params['sl'] = sl
                            params['optimizer'] = optimizer

                            rnn_model = RNNModel(ds, model, params, log_dir=log_dir, division=True)
                            uid = time.strftime("%Y_%m_%d_%H_%M_%S_") + model_id
                            print('-' * 50)
                            print('UID: {}'.format(uid))
                            print('-' * 50)

                            rnn_model.train(uid=uid, batch_size=batch_size, es=early_stopping, nb_epoch=n_epoch,
                                            verbose=2)
                            k.clear_session()
                            i += 1
