import sys
import time
import os

module_root = '..'
sys.path.append(module_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from deepNetworks.model import RNNModel
from utils.data_loader import DataLoader
from deepNetworks.netArch import DeepNetArch1, DeepNetArch2, DeepNetArch3, DeepNetArch1L1, DeepNetArch2L1, \
    DeepNetArch3L1
from keras import backend as k
from utils.history import History

if __name__ == '__main__':
    logs_dir = 'DeepNetArch1-Div'
    model_type = 'DeepNetArch1'
    sl = 100
    ds_rate = 2
    early_stopping = True
    downsample = False
    bmode = True
    subdir = '/bmode/'
    model_history = History(logs_dir)
    opt_params, opt_model_uid = model_history.find_opt_model(auc=False, loss=False, acc=True)

    if downsample:
        ds = DataLoader(sl=sl, downsample=True, downsample_rate=ds_rate)
        sl = int(sl/ds_rate)
    else:
        ds = DataLoader(sl=sl)

    if model_type == 'DeepNetArch1':
        arch = DeepNetArch1(sl=sl, initial_lr=float(opt_params['initial_lr']), l2_reg=float(opt_params['l2_regulizer']),
                            dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                            optimizer=opt_params['optimizer'], summary=1)
    if model_type == 'DeepNetArch2':
        arch = DeepNetArch2(sl=sl, initial_lr=float(opt_params['initial_lr']), l2_reg=float(opt_params['l2_regulizer']),
                            dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                            optimizer=opt_params['optimizer'], summary=1)
    if model_type == 'DeepNetArch3':
        arch = DeepNetArch3(sl=sl, initial_lr=float(opt_params['initial_lr']), l2_reg=float(opt_params['l2_regulizer']),
                            dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                            optimizer=opt_params['optimizer'], summary=1)
    if model_type == 'DeepNetArch1L1':
        arch = DeepNetArch1L1(sl=sl, initial_lr=float(opt_params['initial_lr']),
                              l2_reg=float(opt_params['l2_regulizer']),
                              dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                              optimizer=opt_params['optimizer'], summary=1)
    if model_type == 'DeepNetArch2L1':
        arch = DeepNetArch2L1(sl=sl, initial_lr=float(opt_params['initial_lr']),
                              l2_reg=float(opt_params['l2_regulizer']),
                              dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                              optimizer=opt_params['optimizer'], summary=1)
    if model_type == 'DeepNetArch3L1':
        arch = DeepNetArch3L1(sl=sl, initial_lr=float(opt_params['initial_lr']),
                              l2_reg=float(opt_params['l2_regulizer']),
                              dropout=float(opt_params['dropout']), rec_dropout=float(opt_params['rec_dropout']),
                              optimizer=opt_params['optimizer'], summary=1)

    model, model_id = arch.arch_generator()

    rnn_model = RNNModel(ds, model, opt_params, log_dir=logs_dir + subdir + str(sl), division=True, bmode=bmode)
    uid = time.strftime("%Y_%m_%d_%H_%M_%S_") + model_id
    print('-' * 50)
    print('UID: {}'.format(uid))
    print('-' * 50)

    rnn_model.opt_model_train(uid=uid, batch_size=int(opt_params['batch_size']), es=early_stopping,
                              nb_epoch=int(opt_params['n_epoch']), verbose=2)

    test_predictions = rnn_model.predict_test(uid=uid)
    k.clear_session()
