import sys

module_root = '..'
sys.path.append(module_root)

from utils.history import History

if __name__ == '__main__':
    logs_dir = 'DeepNetArch1-Div'
    model_history = History(logs_dir)

    # model_history.plot_results(train=True, validation=False, params=False, model_visualization=False,
    #                            loss=True, acc=False, auc=False, min_acc=0.4)
    model_history.filtered_learning_curve(train=True, validation=False, params=False,
                                          loss=True, acc=False, auc=False)
