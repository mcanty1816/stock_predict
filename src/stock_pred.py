# -*- coding: utf-8 -*-
"""
Created on Mon Jun 25 13:38:31 2018

@author: User
"""

import os
#경로지정
os.chdir('C:/Users/User/Desktop/stock_picking/stock_pred_model/src')

import pandas as pd
import numpy as np
import copy
import scipy.stats
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import matplotlib.pyplot as plt


import keras
from keras.models import Model, Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, Lambda, Average, Input, Conv2D, MaxPooling2D, BatchNormalization, LSTM, GRU, Bidirectional
from keras import backend as K
from sklearn.metrics import classification_report, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from numpy.random import shuffle 
from sklearn.metrics import r2_score
from matplotlib import pyplot as plt
from sklearn import preprocessing
from keras.models import load_model
from keras.models import model_from_json
from shutil import copyfile
import xlwings as xw
import time
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping

MIN_FUND_DAY = 450
MIN_VALUE = 50000000000

PRED_MONTH = 1
LOOK_BACK_MONTH = 12
DEP_PERCENILTE = 30


#MLP 변수
TRAIN_RATIO = 0.8
TREE_NUM=5

DEPTH = 2                 # Depth of a tree
N_LEAF  = 2 ** (DEPTH + 1)  # Number of leaf node
N_LABEL = 2                 # Number of classes
N_BATCH = 32                # Number of data points per mini-batch
TREE_NUM = 5

MLP1 = 14
MLP2 = 7
MLP3 = 4

MAX_VAL_ACCU = 0.65                # Initial Max Train Data Accuracy
TR_RATIO = 1  



def to_integer(dt_time):
    return 10000*dt_time.year + 100*dt_time.month + dt_time.day

def to_datetime(integer):
    return pd.to_datetime(str(integer), format='%Y%m%d')

def intersect(a, b):
     return list(set(a) & set(b))    

def intersect3(a, b, c):
     return list(set(a) & set(b) & set(c)) 

def monthly_end(date_time):
    dateRange = []  
    tempYear = None  
    dictYears = date_time.groupby(date_time.year)
    for yr in dictYears.keys():
        tempYear = pd.DatetimeIndex(dictYears[yr]).groupby(
                pd.DatetimeIndex(dictYears[yr]).month)
        for m in tempYear.keys():
            dateRange.append(max(tempYear[m]))
    dateRange.append(date_time[0])        
    return pd.DatetimeIndex(dateRange).sort_values()


def read_data():
    price_wb = pd.ExcelFile('../data/stock_price.xlsx')
    price_ts = price_wb.parse("price", header=9, index_col=0, 
                                  skiprows=list(range(10, 14)))
    value_ts = price_wb.parse("value", header=9, index_col=0, 
                                  skiprows=list(range(10, 14)))
    return price_ts, value_ts


def read_kospi_const():
    kospi200_const_wb = pd.ExcelFile('../data/kospi200_constituent.xlsx')
    const_ts = kospi200_const_wb.parse("Sheet1", header=6, index_col=0)
    return const_ts

def read_kospi_price():
    kospi200_price_wb = pd.ExcelFile('../data/kospi200_price.xlsx')
    kospi200_ts = kospi200_price_wb.parse("Sheet1", header=9, index_col=0, skiprows=[10,11,12,13])
    return kospi200_ts['코스피 200']

def filter_stock(cur_date_i, price_ts, value_ts):
    cur_date_t = to_datetime(cur_date_i)
    price_to_cur = price_ts[:cur_date_t]
    value_to_cur = value_ts[:cur_date_t]
    cur_existing_stocks =  []
    cur_not_existing_stocks = []
    satisfy_len_stocks = []
    not_stisfy_len_stocks = []
    satisfy_value_stocks = []
    not_satisfy_value_stocks = []
    for i in list(price_to_cur.columns):
        individual_stock = price_to_cur[i]
        individual_value = value_to_cur[i]
        if np.isnan(individual_stock.loc[cur_date_t]) == False:
            cur_existing_stocks.append(i)
        else:
            cur_not_existing_stocks.append(i)
        if len(individual_stock[-MIN_FUND_DAY:].dropna()) >= MIN_FUND_DAY:
            satisfy_len_stocks.append(i)
        else:
            not_stisfy_len_stocks.append(i)
        if individual_value.loc[cur_date_t] >= MIN_VALUE:     
            satisfy_value_stocks.append(i)
        else:
            not_satisfy_value_stocks.append(i)
    
    filtered_stocks = list(set(cur_existing_stocks) & set(satisfy_len_stocks) & set(satisfy_value_stocks))    
    return filtered_stocks, cur_not_existing_stocks, not_stisfy_len_stocks, not_satisfy_value_stocks


def kospi200_filter_stock(cur_date_i, price_ts, value_ts, cur_kospi200):
    cur_date_t = to_datetime(cur_date_i)
    price_to_cur = price_ts[:cur_date_t]
    value_to_cur = value_ts[:cur_date_t]
    cur_existing_stocks =  []
    cur_not_existing_stocks = []
    satisfy_len_stocks = []
    not_stisfy_len_stocks = []
    satisfy_value_stocks = []
    not_satisfy_value_stocks = []
    for i in cur_kospi200:
        individual_stock = price_to_cur[i]
        individual_value = value_to_cur[i]
        if np.isnan(individual_stock.loc[cur_date_t]) == False:
            cur_existing_stocks.append(i)
        else:
            cur_not_existing_stocks.append(i)
        if len(individual_stock[-MIN_FUND_DAY:].dropna()) >= MIN_FUND_DAY:
            satisfy_len_stocks.append(i)
        else:
            not_stisfy_len_stocks.append(i)
        if individual_value.loc[cur_date_t] >= MIN_VALUE:     
            satisfy_value_stocks.append(i)
        else:
            not_satisfy_value_stocks.append(i)
    
    filtered_stocks = list(set(cur_existing_stocks) & set(satisfy_len_stocks) & set(satisfy_value_stocks))    
    return filtered_stocks, cur_not_existing_stocks, not_stisfy_len_stocks, not_satisfy_value_stocks



def get_dd(ind):
    drawdown = ind[-1]/np.max(ind)-1 
    return drawdown


def print_full(x):
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.max_rows')

def calculate_var(cur_date_t, input_start, input_end, output_start, output_end,
                  individual_price, individual_value, end_of_month):
    one_year_price = individual_price.loc[input_start:input_end]
    monthly_price = one_year_price[end_of_month].dropna()
    monthly_ret = monthly_price.pct_change().dropna()
    one_month_ret = monthly_price[-1]/monthly_price[-2] -1
    two_month_ret = monthly_price[-1]/monthly_price[-3] -1
    six_month_ret = monthly_price[-1]/monthly_price[-7] -1
    nine_month_ret = monthly_price[-1]/monthly_price[-10] -1
    twelve_month_average_ret = monthly_ret.mean() 
    standard_dev = monthly_ret.std() * np.sqrt(12)
    ann_ret = (1+(one_year_price[-1]/one_year_price[0] -1))**(12/len(monthly_price))-1
    #sharpe_ratio = ann_ret/standard_dev
    skewness = scipy.stats.skew(monthly_ret)
    kurtosiss = scipy.stats.kurtosis(monthly_ret)
    if monthly_ret[monthly_ret<0].sum() == 0:
        omega_ratio = monthly_ret[monthly_ret>0].sum()/0.005
    else:    
        omega_ratio = monthly_ret[monthly_ret>0].sum()/-monthly_ret[monthly_ret<0].sum()
    auto_corr1 = monthly_ret.autocorr(lag=1)
    auto_corr2 = monthly_ret.autocorr(lag=2)
    auto_corr3 = monthly_ret.autocorr(lag=3)
    drawdown = get_dd(one_year_price)
    #nav = individual_nav.loc[cur_date_t]
    #nav = individual_value.loc[input_end]
    #if nav == 0:
    #    log_nav=0
    #else:    
    #    log_nav = np.log(nav)
    #fund_age = one_year_price.index[-1].year - individual_price.dropna().index[0].year
    try:
        future_ret = individual_price.loc[output_end]/ individual_price.loc[output_start] -1 
    except TypeError:
        future_ret = None
    variables = [one_month_ret, two_month_ret , six_month_ret, nine_month_ret, twelve_month_average_ret, standard_dev, ann_ret,
                 skewness, kurtosiss, omega_ratio, auto_corr1, auto_corr2, auto_corr3, drawdown, future_ret]    
    variable_row = pd.DataFrame(variables).T     
    return variable_row



def get_high_low_percentile(score):
    low_percentile = list(score[score < np.percentile(score, DEP_PERCENILTE)].index)
    high_percentile = list(score[score>=np.percentile(score, 100-DEP_PERCENILTE)].index)
    return low_percentile, high_percentile 


def make_model_var(cur_date_i, filtered_stocks, price_ts, value_ts):
    end_of_month = monthly_end(price_ts.index)[1:]
    cur_date_t = to_datetime(cur_date_i)
    filtered_price = price_ts.loc[:cur_date_t, filtered_stocks]
    filtered_value = value_ts.loc[:cur_date_t]
    
    cur_date_month_index = list(end_of_month).index(cur_date_t)
    input_start = end_of_month[cur_date_month_index - PRED_MONTH - LOOK_BACK_MONTH]
    input_end = end_of_month[cur_date_month_index - PRED_MONTH]
    output_start = end_of_month[cur_date_month_index - PRED_MONTH]
    output_end = end_of_month[cur_date_month_index]                            
   
    cur_input_start = end_of_month[cur_date_month_index - LOOK_BACK_MONTH]
    cur_input_end = end_of_month[cur_date_month_index]
     
    variables_df = pd.DataFrame()
    cur_variable_df = pd.DataFrame()

    for i in filtered_price.columns:
        individual_price = filtered_price[i]
        individual_value = filtered_value[i]
        variable_row = calculate_var(cur_date_t, input_start, input_end, output_start, output_end,
                                     individual_price, individual_value, end_of_month)
        cur_variable_row = calculate_var(cur_date_t, cur_input_start, cur_input_end, 0, 0,
                                     individual_price, individual_value, end_of_month)
        variables_df = pd.concat([variables_df, variable_row], axis=0)  
        cur_variable_df = pd.concat([cur_variable_df, cur_variable_row])
        
    variables_df.columns = ['one_month_ret', 'two_month_ret', 'six_month_ret', 'nine_month_ret', 'twelve_month_average_ret',  'standard_dev',
        'ann_ret' ,'skewness', 'kurtosis', 'omega_ratio', 'auto_corr1', 'auto_corr2', 'auto_corr3', 'drawdown',  'future_ret'] 
    variables_df.index = filtered_price.columns 
    cur_variable_df.columns = ['one_month_ret', 'two_month_ret', 'six_month_ret', 'nine_month_ret', 'twelve_month_average_ret', 'standard_dev',
         'ann_ret', 'skewness', 'kurtosis', 'omega_ratio', 'auto_corr1', 'auto_corr2', 'auto_corr3', 'drawdown', 'future_ret'] 
    cur_variable_df.index = filtered_price.columns 
    
    low_percentile, high_percentile = get_high_low_percentile(variables_df['future_ret'])
    dep_class = []
    for i in list(variables_df.index):
        if i in low_percentile:
            dep_class.append(0)
        elif i in high_percentile:
            dep_class.append(1)
        else:
            dep_class.append(None)
    
    variables_df['future_ret'] = dep_class  
    
    data_sets_abnormal_stocks = list(variables_df[pd.isnull(variables_df).any(axis=1)].index)
    
    cur_data = copy.deepcopy(cur_variable_df.loc[:, cur_variable_df.columns != 'future_ret'])
    cur_indep_var_abnormal_stocks = list(cur_data[pd.isnull(cur_data).any(axis=1)].index)
    
    removed_stocks = list(set(data_sets_abnormal_stocks + cur_indep_var_abnormal_stocks))
    
    variables_df = variables_df.drop(removed_stocks)
    cur_variable_df = cur_variable_df.drop(removed_stocks)

    data_sets = variables_df    
    cur_indep_data =  cur_variable_df
    return data_sets, cur_indep_data, removed_stocks




def leaf_prob(node_prob):
    # 노드 번호가 0부터 시작된다고 가정, 노드번호가 node_num인 leaf로 라우팅 될 확률
    def _leaf_prob(node_num, node_prob):
        if node_num <= 0:
            return 1
        elif node_num % 2 == 1:
            return node_prob[:, (node_num-1)//2] * _leaf_prob((node_num-1)//2, node_prob)
        else:
            return (1-node_prob[:, (node_num-1)//2]) * _leaf_prob((node_num-1)//2, node_prob)
    ret = []
    node_nm = int(node_prob.get_shape()[1])
    for i in range(node_nm, node_nm*2+1):
        ret.append(_leaf_prob(i, node_prob))
    return K.transpose(K.stack(ret))


def mlp_model(model):
       
    model = Dense(MLP1, activation='elu', kernel_initializer='he_normal')(model)
    model = Dropout(0.1)(model)
    model = BatchNormalization()(model)
    model = Dense(MLP2, activation='elu', kernel_initializer='he_normal')(model)
    model = Dropout(0.1)(model)
    model = BatchNormalization()(model)
    model = Dense(MLP3, activation='elu', kernel_initializer='he_normal')(model)
    model = Dropout(0.2)(model)
    #model = BatchNormalization(scale=False)(model)
    #model = Dense(mlp3, activation='relu', kernel_initializer='glorot_normal')(model)
    #model = Dropout(0.2)(model)    
    return model


def tree_e_model(tree_num, train_x):
    i = Input(shape=(train_x.shape[1],))
    
    models = []
        
    for _ in range(tree_num):
        #model = Reshape((x, y))(i)
        model = mlp_model(i)
        model = BatchNormalization(scale=False)(model)
        model = Dropout(0.2)(model)
        model = Dense(N_LEAF-1, activation='relu', kernel_initializer='he_normal')(model)
        model_t = Lambda(leaf_prob)(model)
        model_t = Dense(N_LABEL, kernel_initializer='he_normal', activation = 'softmax')(model_t)
        models.append(model_t)
    
    o = Average()(models)
    model = Model(inputs=i, outputs=o)
    opt = keras.optimizers.Adam(lr = 0.001) #0.001, 0.9
    model.compile(optimizer=opt,
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    
    model.summary()
    
    return model


def split_data(data_sets, train_ratio):
    indep_var = data_sets.iloc[:,:-1]
    dep_var = data_sets.iloc[:,-1]
    #scaler = StandardScaler()
    #scaler = scaler.fit(indep_var)
    #scaled_indep_var = scaler.transform(indep_var)
    #scaled_indep_var_df = pd.DataFrame(scaled_indep_var)
    sequence = [i for i in range(len(indep_var))]
    shuffle(sequence) 
    train_no = int(len(indep_var) * train_ratio)
    train_idx = sequence[:train_no]
    test_idx = sequence[train_no:]
    train_x = np.array(indep_var.iloc[train_idx])
    test_x = np.array(indep_var.iloc[test_idx])
    train_y = np.array(dep_var.iloc[train_idx])
    test_y = np.array(dep_var.iloc[test_idx])
    return train_x, train_y, test_x, test_y



class PlotLosses(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        
        self.fig = plt.figure()
        
        self.logs = []

    def on_epoch_end(self, epoch, logs={}):    
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.i += 1
        return
    
        #clear_output(wait=True)
    def on_train_end(self, logs={}):   
        plt.plot(self.x, self.losses, label="train loss")
        plt.plot(self.x, self.val_losses, label="val. loss")
        plt.legend()
        plt.show();



class EarlyStoppingByAccu(keras.callbacks.Callback):
    def __init__(self, monitor='val_acc', value= MAX_VAL_ACCU, verbose=0):
        super(keras.callbacks.Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            print("\n\nEarly stopping requires %s available!\n" % self.monitor)
            exit()

        if current > self.value:
            if self.verbose > 0:
                print("\n\nEpoch %d: Early Stopping THR by acc %.3f\n" % (epoch, current))
            self.model.stop_training = True
            


def load_mlp_model():
    json_file = open('../model_file/mlp_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("../model_file/mlp_model.h5")
    return loaded_model


def create_rf_model(data_sets, cur_indep_data):
    #ss = ShuffleSplit(n_splits=10, test_size=0.2, random_state=150)
    #for train_index, test_index in ss.split(data_sets):
    #    print("%s %s" % (train_index, test_index))
    #regr = RandomForestRegressor(n_estimators=1000, oob_score = True, random_state=12345)
    regr = RandomForestClassifier(n_estimators=1000, oob_score = True, random_state=12345)
    X = data_sets.iloc[:,:-1]
    y = data_sets.iloc[:,-1]
    regr.fit(np.array(X), np.array(y))
  
    feature_importance = regr.feature_importances_
    #important_vars = np.array(X.columns)[list((-feature_importance).argsort()[:5])]
    feature_importance_df = pd.DataFrame(data = feature_importance, index = X.columns)
    
    pred_ret_df = pd.DataFrame(index = cur_indep_data.index, columns = ['pred ret'])
    pred_prob_df = pd.DataFrame(index = cur_indep_data.index, columns = ['pred ret'])
    for i in list(cur_indep_data.index):
        pred_ret = regr.predict(np.array(cur_indep_data.loc[i].dropna()).reshape(1,-1))[0]
        pred_prob = regr.predict_proba(np.array(cur_indep_data.loc[i].dropna()).reshape(1,-1))[0][1]
        pred_prob_df.loc[i] = pred_prob
        pred_ret_df.loc[i] = pred_ret 
    
    accuracy = regr.oob_score_
    #print('accuracy:', accuracy)
    return accuracy, feature_importance_df, pred_prob_df, pred_ret_df


def create_mlp_model(data_sets, cur_indep_data):
    train_x, train_y, test_x, test_y = split_data(data_sets, 0.8)
    
    model = tree_e_model(TREE_NUM, train_x)
  
    #plot_losses = PlotLosses()
    train_y_category = to_categorical(train_y, num_classes=2)
    test_y_category = to_categorical(test_y, num_classes=2)
    
    #es_tr_acc = EarlyStoppingByAccu('val_acc', MAX_VAL_ACCU, 1)
    early_stop = EarlyStopping(monitor='val_acc', patience=20, mode='max')
    
    hist = model.fit(train_x, train_y_category, batch_size=N_BATCH, epochs=500, shuffle=shuffle, validation_data=(test_x, test_y_category),
              callbacks=[early_stop])
    
    fig, loss_ax = plt.subplots()
    acc_ax = loss_ax.twinx()
    
    loss_ax.plot(hist.history['loss'], 'y', label='train loss')
    loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
    
    acc_ax.plot(hist.history['acc'], 'b', label='train acc')
    acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
    
    loss_ax.set_xlabel('epoch')
    loss_ax.set_ylabel('loss')
    acc_ax.set_ylabel('accuray')
    
    loss_ax.legend(loc='upper left')
    acc_ax.legend(loc='lower left')
    
    plt.show()
    
    
    model_json = model.to_json()
    with open("../model_file/mlp_model.json", "w") as json_file:
        json_file.write(model_json)
    model.save_weights("../model_file/mlp_model.h5")
 
    mlp_loaded_model = load_mlp_model()
    pred_prob_df = pd.DataFrame(None, index = list(cur_indep_data.index), columns = ['pred_ret'])
    pred_ret_df = pd.DataFrame(None, index = list(cur_indep_data.index), columns = ['pred_ret'])
    for k in list(cur_indep_data.index):
        mlp_pred_ret = mlp_loaded_model.predict(np.array(cur_indep_data.loc[k].dropna()).reshape(1,-1))[0]
        pred_prob_df.loc[k] = mlp_pred_ret[1]
        pred_ret_df.loc[k] = np.argmax(mlp_pred_ret) 
    accuracy = hist.history['val_acc'][-1]
    return accuracy, pred_prob_df, pred_ret_df


def save_pred_file(cur_date_i, pred_prob_df):
    with open('../files/pred_result.dump', "rb") as f:
        pred_result_dict = pd.read_pickle(f)
        pred_result_dict[cur_date_i] = pred_prob_df    
    with open('../files/pred_result.dump', "wb") as f:
        pd.to_pickle(pred_result_dict, f)


def save_pred_result():
    price_ts, value_ts = read_data()
    const_ts = read_kospi_const()
    end_of_month = monthly_end(price_ts.index)[1:]
    for i in end_of_month[24:-1]:
    #for i in end_of_month[53:]: 
    #for i in end_of_month[53:]:
    #for i in end_of_month[82:]:
        cur_date_i = to_integer(i)
        prev_cur_date_i = to_integer(end_of_month[list(end_of_month).index(i)-1])
        more_prev_cur_date_i = to_integer(end_of_month[list(end_of_month).index(i)-2])
        cur_kospi200 = list(const_ts.loc[i]['Name'].values)
        kospi200_filtered_stocks, _, _, _ = \
                                        kospi200_filter_stock(cur_date_i, price_ts, value_ts, cur_kospi200)           
        
        filtered_stocks, cur_not_existing_stocks, not_stisfy_len_stocks, not_satisfy_value_stocks = \
                                                    filter_stock(cur_date_i, price_ts, value_ts)                                           
        prev_filtered_stocks, _, _, _ = filter_stock(prev_cur_date_i, price_ts, value_ts)   
        more_prev_filtered_stocks, _, _, _ = filter_stock(more_prev_cur_date_i, price_ts, value_ts)                                                                   
        
        data_sets, cur_indep_data, removed_stocks = make_model_var(cur_date_i, filtered_stocks, price_ts, value_ts)
        _, cur_kospi200_indep_data, _ = make_model_var(cur_date_i, kospi200_filtered_stocks, price_ts, value_ts)
        
        prev_data_sets, _, _ = make_model_var(prev_cur_date_i, prev_filtered_stocks, price_ts, value_ts)
        more_prev_data_sets, _, _ = make_model_var(more_prev_cur_date_i, more_prev_filtered_stocks, price_ts, value_ts)
    
        tot_data_sets = pd.concat([data_sets, prev_data_sets, more_prev_data_sets], axis=0)
        #accuracy, feature_importance_df, pred_prob_df, pred_ret_df = create_rf_model(tot_data_sets, cur_indep_data)
        
        mlp_accuracy, mlp_pred_prob_df, mlp_pred_ret_df = create_mlp_model(tot_data_sets, cur_indep_data)
        K.clear_session()
        rf_accuracy, _, rf_pred_prob_df, rf_pred_ret_df = create_rf_model(tot_data_sets, cur_indep_data)
        K.clear_session()
        pred_prob_df = pd.concat([rf_pred_prob_df, mlp_pred_prob_df], axis=1)
        pred_prob_df.columns = ['rf', 'mlp']
        print(cur_date_i, len(tot_data_sets), 'rf accu: ', rf_accuracy, 'mlp accu: ', mlp_accuracy)
        save_pred_file(cur_date_i, pred_prob_df)
 
    
def get_four_percentile(score):
    first_percentile = list(score[(score>=np.percentile(score,0)) & (score < np.percentile(score,25))].index)
    second_percentile = list(score[(score>=np.percentile(score,25)) & (score < np.percentile(score,50))].index)
    third_percentile = list(score[(score>=np.percentile(score,50)) & (score<= np.percentile(score,75))].index)
    fourth_percentile = list(score[(score>np.percentile(score,75)) & (score <= np.percentile(score,100))].index)
    return first_percentile, second_percentile, third_percentile, fourth_percentile     
 
    
def make_dual_momentum():
    price_ts, value_ts =read_data()
    end_of_month = monthly_end(price_ts.index)[1:]
    dual_ret_df = pd.DataFrame(None, index = end_of_month[24:], columns = ['best', 'worst'])
    for cur_date_t in end_of_month[24:]:
        if cur_date_t ==  end_of_month[24:][-1]:
            break
        cur_date_i = to_integer(cur_date_t)
        next_date_t = sorted(end_of_month)[sorted(end_of_month).index(cur_date_t) +1]
        next_date_i = to_integer(next_date_t)
        
        cur_twelve_mon_date_t = list(end_of_month)[list(end_of_month).index(cur_date_t)-12]
        
        filtered_stocks, cur_not_existing_stocks, not_stisfy_len_stocks, not_satisfy_value_stocks = filter_stock(cur_date_i, price_ts, value_ts)
        filtered_stocks_df = price_ts[filtered_stocks]
        
        cur_twelve_mon_ret = filtered_stocks_df.loc[cur_date_t]/filtered_stocks_df.loc[cur_twelve_mon_date_t]-1
        
        first_percentile, second_percentile, third_percentile, fourth_percentile = get_four_percentile(cur_twelve_mon_ret)
        
        cur_winner_stocks = list(cur_twelve_mon_ret[cur_twelve_mon_ret>0].index)
        cur_loser_stocks = list(cur_twelve_mon_ret[cur_twelve_mon_ret<=0].index)
         
        dual_best_ten = cur_twelve_mon_ret.loc[cur_winner_stocks].sort_values(ascending=False)[0:10].index
        dual_worst_ten = cur_twelve_mon_ret.loc[cur_loser_stocks].sort_values(ascending=True)[0:10].index
        
        dual_best_ret = (price_ts.loc[next_date_t,dual_best_ten]/price_ts.loc[cur_date_t,dual_best_ten]-1).mean()
        dual_worst_ret = (price_ts.loc[next_date_t,dual_worst_ten]/price_ts.loc[cur_date_t,dual_worst_ten]-1).mean()
        print(cur_date_i, len(dual_best_ten), len(dual_worst_ten), list(dual_best_ten))
        dual_ret_df.loc[next_date_t] = [dual_best_ret, dual_worst_ret]
    
    dual_mom_index = 100 * np.cumprod(1+dual_ret_df.fillna(0))
    dual_mom_index.plot()


def adjust_date(start_date, end_date, ret_index):
    start_date_t = to_datetime(start_date) 
    end_date_t = to_datetime(end_date)
    adjust_date = ret_index.loc[start_date_t:end_date_t]
    adjust_result = np.cumprod(1+adjust_date.pct_change().fillna(0))*100
    return adjust_result


def backtest():
    price_ts, value_ts = read_data()
    const_ts = read_kospi_const()
    kospi_200_ts = read_kospi_price()
    with open('../files/pred_result.dump', "rb") as f:
        pred_result_dict = pd.read_pickle(f)
     
    ret_df = pd.DataFrame(None, index = [to_datetime(i) for i in sorted(pred_result_dict)], 
                                columns = ['first', 'second', 'third', 'fourth', 'top_ten','top_ten_with_value'])
    accu_df = pd.DataFrame(None, index = [to_datetime(i) for i in sorted(pred_result_dict)], columns = ['accuracy'])
    for cur_date_i in sorted(pred_result_dict):
        if cur_date_i ==sorted(pred_result_dict)[-1]:
            break
        cur_date_t = to_datetime(cur_date_i)
        next_date_i = sorted(pred_result_dict)[sorted(pred_result_dict).index(cur_date_i) +1]
        next_date_t = to_datetime(next_date_i)
        
        cur_pred_ret = pred_result_dict[cur_date_i]
        cur_pret_result =  cur_pred_ret.mean(axis=1)
        
        cur_kospi200_const = const_ts.loc[cur_date_t]
        
        cur_kospi200_pret = cur_pret_result.loc[list(cur_kospi200_const['Name'])].dropna()
        
        #first_percentile, second_percentile, third_percentile, fourth_percentile = get_four_percentile(cur_pret_result)
        first_percentile, second_percentile, third_percentile, fourth_percentile = get_four_percentile(cur_kospi200_pret)
        
        if len(price_ts.loc[next_date_t,cur_kospi200_pret.index]) != len(price_ts.loc[next_date_t,cur_kospi200_pret.index].dropna()):
            nan_list = list(price_ts.loc[next_date_t,cur_kospi200_pret.index][pd.isnull(price_ts.loc[next_date_t,cur_kospi200_pret.index])].index)
            price_ts.loc[next_date_t,nan_list] = price_ts.loc[cur_date_t,nan_list].values
        
        first_ret = (price_ts.loc[next_date_t,first_percentile]/price_ts.loc[cur_date_t,first_percentile]-1).mean()
        second_ret = (price_ts.loc[next_date_t,second_percentile]/price_ts.loc[cur_date_t,second_percentile]-1).mean()
        third_ret = (price_ts.loc[next_date_t,third_percentile]/price_ts.loc[cur_date_t,third_percentile]-1).mean()
        fourth_ret = (price_ts.loc[next_date_t,fourth_percentile]/price_ts.loc[cur_date_t,fourth_percentile]-1).mean()
        
        """
        if len(price_ts.loc[next_date_t,third_percentile]) != len(price_ts.loc[next_date_t,third_percentile].dropna()):
            nan_list = list(price_ts.loc[next_date_t,third_percentile][pd.isnull(price_ts.loc[next_date_t,third_percentile])].index)
            price_ts.loc[next_date_t,nan_list] = price_ts.loc[cur_date_t,nan_list].values
        
        if len(price_ts.loc[next_date_t,first_percentile]) == len(price_ts.loc[next_date_t,first_percentile].dropna()) and \
           len(price_ts.loc[next_date_t,second_percentile]) == len(price_ts.loc[next_date_t,second_percentile].dropna()) and \
           len(price_ts.loc[next_date_t,third_percentile]) == len(price_ts.loc[next_date_t,third_percentile].dropna()) and \
           len(price_ts.loc[next_date_t,fourth_percentile]) == len(price_ts.loc[next_date_t,fourth_percentile].dropna()) :
             print(cur_date_i, 'ok')
        else:
            print(cur_date_i, 'not ok')
        """
     
        
        top_ten_stocks = list(cur_kospi200_pret.sort_values(ascending=False)[0:10].index)
        cur_value = value_ts.loc[cur_date_t].sort_values(ascending=False)[0:400]
        value_result_intersect = intersect(list(cur_value.index), list(cur_kospi200_pret.index)) 
        top_ten_stocks_with_value = list(cur_kospi200_pret.loc[value_result_intersect].sort_values(ascending=False)[0:20].index)
        
        top_ten_stocks_ret = (price_ts.loc[next_date_t,top_ten_stocks]/price_ts.loc[cur_date_t,top_ten_stocks]-1)
        accuracy = len(top_ten_stocks_ret[top_ten_stocks_ret>0])/len(top_ten_stocks_ret)
        accu_df.loc[next_date_t] = accuracy
        top_ten_ret = (price_ts.loc[next_date_t,top_ten_stocks]/price_ts.loc[cur_date_t,top_ten_stocks]-1).mean()
        top_ten_with_value_ret = (price_ts.loc[next_date_t,top_ten_stocks_with_value]/price_ts.loc[cur_date_t,top_ten_stocks_with_value]-1).mean()
        ret_df.loc[next_date_t] = [first_ret, second_ret, third_ret, fourth_ret, top_ten_ret, top_ten_with_value_ret]
        print(cur_date_i, top_ten_stocks)
    
    ret_index = np.cumprod(1+ret_df.fillna(0)) * 100
    kospi_200_index = np.cumprod(1+kospi_200_ts.loc[ret_index.index].pct_change().fillna(0)) * 100
    kospi_200_index.name = 'KOSPI200'
    ret_index = pd.concat([ret_index, kospi_200_index], axis=1)
    
    adjust_ret_index = adjust_date(20071228, 20181228, ret_index)
    adjust_ret_index.plot()   
    
    cum_ret = ret_index.iloc[-1]/ret_index.iloc[0]-1    
    annualized_ret = (1+cum_ret)**(12/len(ret_index))-1  
    vol = ret_index.pct_change().std()*np.sqrt(12)   
    sharpe = annualized_ret/ vol    
    

        