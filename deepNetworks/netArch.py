from keras.layers import LSTM, GRU, SimpleRNN, Dense
from keras.models import Sequential
from keras.regularizers import l2


class DeepNetArch1:  # 2 Layers LSTM + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch1"
        model = Sequential()
        model.add(LSTM(units=self.sl, return_sequences=True, dropout=self.dropout, recurrent_dropout=self.rec_dropout,
                       input_shape=(self.sl, 1), stateful=False))
        model.add(LSTM(units=self.sl, dropout=self.dropout, recurrent_dropout=self.rec_dropout, return_sequences=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


class DeepNetArch2:  # 2 Layers GRU + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch2"
        model = Sequential()
        model.add(GRU(units=self.sl, return_sequences=True, dropout=self.dropout, recurrent_dropout=self.rec_dropout,
                      input_shape=(self.sl, 1), stateful=False))
        model.add(GRU(units=self.sl, dropout=self.dropout, recurrent_dropout=self.rec_dropout, return_sequences=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


class DeepNetArch3:  # 2 Layers RNN + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch3"
        model = Sequential()
        model.add(SimpleRNN(units=self.sl, return_sequences=True, dropout=self.dropout,
                            recurrent_dropout=self.rec_dropout,
                            input_shape=(self.sl, 1), stateful=False))
        model.add(SimpleRNN(units=self.sl, dropout=self.dropout, recurrent_dropout=self.rec_dropout, return_sequences=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


class DeepNetArch1L1:  # 1 Layers LSTM + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch1l1"
        model = Sequential()
        model.add(LSTM(units=self.sl, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.rec_dropout,
                       input_shape=(self.sl, 1), stateful=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


class DeepNetArch2L1:  # 1 Layers GRU + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch2l1"
        model = Sequential()
        model.add(GRU(units=self.sl, return_sequences=False, dropout=self.dropout, recurrent_dropout=self.rec_dropout,
                      input_shape=(self.sl, 1), stateful=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name


class DeepNetArch3L1:  # 1 Layers RNN + Dense
    def __init__(self, sl, initial_lr, l2_reg, dropout, rec_dropout, optimizer, summary):
        self.sl = sl
        self.summary = summary
        self.l2_reg = l2(l2_reg)
        self.dropout = dropout
        self.rec_dropout = rec_dropout
        self.initial_lr = initial_lr
        self.optimizer = optimizer

    def arch_generator(self):
        model_name = "arch3l1"
        model = Sequential()
        model.add(SimpleRNN(units=self.sl, return_sequences=False, dropout=self.dropout,
                            recurrent_dropout=self.rec_dropout,
                            input_shape=(self.sl, 1), stateful=False))
        model.add(Dense(1, activation="sigmoid", kernel_initializer="he_normal", kernel_regularizer=self.l2_reg))
        model.compile(loss='binary_crossentropy', optimizer=self.optimizer, metrics=['accuracy'])
        if self.summary:
            print(model.summary())
        return model, model_name

