import glob
import json
import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, SVG
from keras.models import model_from_json
from keras.utils.vis_utils import model_to_dot

module_root = '../..'
sys.path.append(module_root)
from utils import settings as s


class History:
    def __init__(self, logs_dir):
        self.logs_folder = os.path.join(s.intermediate_folder, 'logs', logs_dir)
        self.csv_logs_folder = os.path.join(self.logs_folder, 'csv_logs')
        self.model_logs_folder = os.path.join(self.logs_folder, 'model_logs')
        self.train_logs_folder = os.path.join(self.logs_folder, 'train_logs')

    def plot_results(self, train, validation, params, model_visualization, loss, acc, auc, min_acc):
        csv_name_sorted = sorted(glob.glob(self.csv_logs_folder + '/*.log'))
        csv_train_sorted = sorted(glob.glob(self.train_logs_folder + '/*.csv'))
        uid_sample = os.path.split(csv_name_sorted[1])[1].split('.')[0]
        final = pd.DataFrame(
            index=pd.read_csv(list(filter(lambda x: uid_sample in x, csv_train_sorted))[0], header=None)[0].tolist())
        final.index.name = None
        plt.figure(figsize=(8, 6))
        for csv_file in csv_name_sorted:
            if os.path.getsize(csv_file) > 0:
                data = pd.read_csv(csv_file)
                uid = os.path.split(csv_file)[1].split('.')[0]

                if np.amax(data['val_acc']) > min_acc:
                    if validation:
                        if loss:
                            plt.plot(data['epoch'], data['val_loss'], label=uid + ' Val Loss')
                        if acc:
                            plt.plot(data['epoch'], data['val_acc'], label=uid + ' Val Acc')
                        if auc:
                            plt.plot(data['epoch'], data['val_auc'], label=uid + ' Val AUC')
                    if train:
                        if loss:
                            plt.plot(data['epoch'], data['loss'], label=uid + ' Train Loss')
                        if acc:
                            plt.plot(data['epoch'], data['acc'], label=uid + ' Train Acc')

                    if params:
                        print(uid)
                        train_csv = pd.read_csv(list(filter(lambda x: uid in x, csv_train_sorted))[0], header=None)
                        train_csv.columns = ['parameters', uid[8:25]]
                        train_csv.set_index('parameters', inplace=True)
                        final = final.join(train_csv)
                        print("*" * 100)
                        print("*" * 100)
                    if model_visualization:
                        model_log = glob.glob(self.model_logs_folder + '/' + uid + '*.json')[0]
                        with open(model_log) as model_file:
                            json_string = json.load(model_file)
                        model = model_from_json(json_string)
                        print(uid)
                        #                     print(model.summary())
                        print("*" * 100)
                        dot = model_to_dot(model).create(prog='dot', format='svg')
                        return SVG(dot)
        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.xlabel('Epoch')
        plt.legend()
        # plt.close()
        plt.savefig(self.logs_folder + '/out.pdf', transparent=True)
        if params:
            display(final.drop(['data_ID', 'data_id'], axis=0))
        return plt

    def find_opt_model(self, loss, acc, auc):
        csv_name_sorted = sorted(glob.glob(self.csv_logs_folder + '/*.log'))
        csv_train_sorted = sorted(glob.glob(self.train_logs_folder + '/*.csv'))
        uid_sample = os.path.split(csv_name_sorted[1])[1].split('.')[0]
        final = pd.DataFrame(
            index=pd.read_csv(list(filter(lambda x: uid_sample in x, csv_train_sorted))[0], header=None)[0].tolist())
        final.index.name = None
        validation_results = []
        train_uids = []
        for csv_file in csv_name_sorted:
            if os.path.getsize(csv_file) > 0:
                data = pd.read_csv(csv_file)
                uid = os.path.split(csv_file)[1].split('.')[0]

                train_uids.append(uid)
                if auc:
                    validation_results.append(np.amax(data['val_auc']))
                if loss:
                    validation_results.append(np.amin(data['val_loss']))
                if acc:
                    validation_results.append(np.amax(data['val_acc']))
        if loss:
            opt_model_uid = train_uids[validation_results.index(min(validation_results))]
        else:
            opt_model_uid = train_uids[validation_results.index(max(validation_results))]
        print("Optimum Model ID:  ", opt_model_uid)
        print("Optimum Training Value: ", max(validation_results))
        train_csv = pd.read_csv(list(filter(lambda x: opt_model_uid in x, csv_train_sorted))[0], header=None)
        train_csv.columns = ['parameters', opt_model_uid[8:25]]
        train_csv.set_index('parameters', inplace=True)
        final = final.join(train_csv)
        opt_model = final
        print("Optimum Params:")
        print(final)
        print("*" * 100)
        print("*" * 100)

        opt_model_axes = opt_model.axes
        opt_model_rows = opt_model_axes[0]
        opt_model_cols = opt_model_axes[1].values

        opt_params = dict()

        for rows in opt_model_rows:
            for cols in opt_model_cols:
                opt_params[rows] = opt_model.get_value(rows, cols)

        return opt_params, opt_model_uid

    def plot_learning_curve(self, model_id, acc, auc, loss):
        csv_name_sorted = sorted(glob.glob(self.csv_logs_folder + '/*.log'))
        for csv_file in csv_name_sorted:
            uid = os.path.split(csv_file)[1].split('.')[0]
            if uid == model_id:
                data = pd.read_csv(csv_file)
                if auc:
                    train_results = data['auc']
                    validation_results = data['val_auc']
                if acc:
                    train_results = data['acc']
                    validation_results = data['val_acc']
                if loss:
                    train_results = data['loss']
                    validation_results = data['val_loss']
                break

        plt.figure()
        plt.title("Learning curve of " + model_id)

        plt.xlabel("Epoch Number")
        plt.ylabel("Accuracy")

        train_sizes = range(1, train_results.shape[0] + 1, 1)
        train_scores_mean = np.mean(train_results)
        train_scores_std = np.std(train_results)
        validation_scores_mean = np.mean(validation_results)
        validation_scores_std = np.std(validation_results)
        plt.grid()

        plt.fill_between(train_sizes, train_results - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
        plt.fill_between(train_sizes, validation_results - validation_scores_std,
                         validation_scores_mean + validation_scores_std, alpha=0.1, color="g")

        plt.plot(train_sizes, train_results, 'o-', color="r",
                 label="Training score")
        plt.plot(train_sizes, validation_results, 'o-', color="g",
                 label="Cross-validation score")

        plt.legend(loc="best")
        return plt

    def filtered_learning_curve(self, train, validation, params, loss, acc, auc):
        csv_name_sorted = sorted(glob.glob(self.csv_logs_folder + '/*.log'))
        csv_train_sorted = sorted(glob.glob(self.train_logs_folder + '/*.csv'))
        uid_sample = os.path.split(csv_name_sorted[1])[1].split('.')[0]
        final = pd.DataFrame(
            index=pd.read_csv(list(filter(lambda x: uid_sample in x, csv_train_sorted))[0], header=None)[0].tolist())
        final.index.name = None
        plt.figure(figsize=(8, 6))
        for csv_file in csv_name_sorted:
            if os.path.getsize(csv_file) > 0:
                data = pd.read_csv(csv_file)
                uid = os.path.split(csv_file)[1].split('.')[0]

                train_csv = pd.read_csv(list(filter(lambda x: uid in x, csv_train_sorted))[0], header=None)
                train_csv.columns = ['parameters', uid[8:25]]
                train_csv.set_index('parameters', inplace=True)
                final = final.join(train_csv)
                current_model = final
                current_model_axes = current_model.axes
                current_model_rows = current_model_axes[0]
                current_model_cols = current_model_axes[1].values
                current_model_params = dict()
                for rows in current_model_rows:
                    for cols in current_model_cols:
                        current_model_params[rows] = current_model.get_value(rows, cols)

                if validation:
                    if loss:
                        plt.plot(data['epoch'], data['val_loss'], label=uid + ' Val Loss')
                    if acc:
                        plt.plot(data['epoch'], data['val_acc'], label=uid + ' Val Acc')
                    if auc:
                        plt.plot(data['epoch'], data['val_auc'], label=uid + ' Val AUC')
                if train:
                    if loss:
                        plt.plot(data['epoch'], data['loss'], label=uid + ' Train Loss')
                    if acc:
                        plt.plot(data['epoch'], data['acc'], label=uid + ' Train Acc')

        plt.ylabel('Accuracy')
        plt.ylim([0, 1])
        plt.xlabel('Epoch')
        plt.legend()
        # plt.close()
        plt.savefig(self.logs_folder + '/out.pdf', transparent=True)
        if params:
            display(final.drop(['data_ID', 'data_id'], axis=0))
        return plt