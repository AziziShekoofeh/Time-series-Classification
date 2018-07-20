import sys
import os

module_root = '..'
sys.path.append(module_root)

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from utils.data_loader import DataLoader
from keras import backend as k
from keras.models import load_model
from utils import settings as s
from sklearn.metrics import roc_auc_score
import scipy.io as spio


if __name__ == '__main__':
    logs_dir = 'DeepNetArch1-Div'
    sl = 100
    ds_rate = 2
    early_stopping = True
    downsample = False
    bmode = True
    whole_map = False

    uid = '2017_10_20_09_48_15_arch3'
    uid = '2017_10_19_09_55_17_arch2'
    uid = '2017_10_19_20_54_29_arch1'

    if downsample:
        ds = DataLoader(sl=sl, downsample=True, downsample_rate=ds_rate)
        sl = int(sl/ds_rate)
    else:
        ds = DataLoader(sl=sl)

    if whole_map:
        test_seq = ds.load_whole_test(bmode)
    else:
        test_seq, test_label = ds.load_test(bmode)

    model_checkpoint_dir = os.path.join(s.intermediate_folder, 'model_checkpoints/opt')
    model_checkpoint_file = os.path.join(model_checkpoint_dir, uid + '.hdf5')
    model = load_model(model_checkpoint_file)

    test_predictions = model.predict(test_seq, verbose=1)
    results = {'test_predictions': test_predictions}

    logs_dir = os.path.join(s.intermediate_folder, 'logs', logs_dir)
    test_log_dir = os.path.join(logs_dir, 'test_logs/')

    if not whole_map:
        test_auc = roc_auc_score(test_label, test_predictions)
        spio.savemat(test_log_dir + uid + '.mat', results)
        print(["Test AUC: ", test_auc])
    else:
        spio.savemat(test_log_dir + uid + '_whole.mat', results)

    # print('-' * 50)
    # print('UID: {}'.format(uid))
    # print('-' * 50)

    k.clear_session()
