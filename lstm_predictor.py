import numpy as np
import pandas as pd

import os
import sys
import getopt
import time
import math

from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger
from keras import optimizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# バッチサイズからはみ出た要素を削除する
def trim_dataset(input_data, batch_size):
    no_of_rows_drop = input_data.shape[0]%batch_size
    if no_of_rows_drop > 0:
        return input_data[:-no_of_rows_drop]
    else:
        return input_data


# numpyの要素ごとに正規化を行う
def min_max_scaler_converter(input, output_index):
    output = []
    data_range = []
    data_shift = []

    for row in input:
        scaler = MinMaxScaler()
        output.append(scaler.fit_transform(row))
        data_range.append(scaler.data_range_[output_index])
        data_shift.append(scaler.data_min_[output_index])
    
    return np.array(output), data_range, data_shift


#インプットと正解データのセットを作る
def build_timeseries(input_data, y_col_index, time_steps, output_gap):
    dimention_0 = input_data.shape[0] - time_steps - output_gap
    dimention_1 = input_data.shape[1]
    x = np.zeros((dimention_0, time_steps, dimention_1))
    y = np.zeros((dimention_0,))
    for i in range(dimention_0):
        x[i] = input_data[i:time_steps+i]
        y[i] = input_data[time_steps+output_gap+i, y_col_index]
    print("length of time-series i/o",x.shape,y.shape)
    return x, y


# LSTMモデルを作成する
def create_model(learning_rate, batch_size, time_steps, feature_columns):
    lstm_model = Sequential()
    lstm_model.add(LSTM(100, batch_input_shape=(batch_size, time_steps, len(feature_columns)),
                        dropout=0.0, recurrent_dropout=0.0, stateful=True, return_sequences=True,
                        kernel_initializer='random_uniform'))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(LSTM(60, dropout=0.0))
    lstm_model.add(Dropout(0.4))
    lstm_model.add(Dense(20,activation='relu'))
    lstm_model.add(Dense(1,activation='sigmoid'))
    optimizer = optimizers.RMSprop(lr=learning_rate)
    lstm_model.compile(loss='mean_squared_error', optimizer=optimizer)
    return lstm_model


""" train(input_path, output, logdir)

    Description:
        指定したCSVファイルから学習を行う。
        
    Args:
        input_path(str): CSVのフルパス
        output(str): モデルデータの保存先ファイル名
        logdir(str): ログデータの保存先ファイル名

    Returns:
        None
"""
def train(input_path, output, logdir, epochs=10, learning_rate=0.0001, 
            time_steps=15, output_gap=5, batch_size=10, feature_columns=[], 
            output_index=0):
    if os.path.isfile(input_path):
        print("Input path is single file.")
        
        # ファイル形式のチェック
        if not input_path.endswith(".csv"):
            print("Input file is not CSV.")

        # 学習データの読み込み
        df_input = pd.read_csv(input_path, engine='python')
        print(df_input.dtypes)
        
        # 入力データを学習用とテスト用に分ける
        df_train, df_test = train_test_split(df_input, train_size=0.8, test_size=0.2, shuffle=False)
        print("Train-Test size", len(df_train), len(df_test))

        # 入力データを正規化
        min_max_scaler = MinMaxScaler()
        x_train = min_max_scaler.fit_transform(df_train.loc[:,feature_columns].values)
        x_test = min_max_scaler.transform(df_train.loc[:,feature_columns])

        del df_input
        del df_test
        del df_train

        # 入力データからインプットとアウトプットのセットを作成
        x_t, y_t = build_timeseries(x_train, output_index, time_steps, output_gap)
        x_t = trim_dataset(x_t, batch_size)
        y_t = trim_dataset(y_t, batch_size)
        print("Batch trimmed size",x_t.shape, y_t.shape)

        # テスト用データをさらにvalidateとtestに分ける
        x_temp, y_temp = build_timeseries(x_test, output_index, time_steps, output_gap)
        x_val, x_test_t = np.split(trim_dataset(x_temp, batch_size),2)
        y_val, y_test_t = np.split(trim_dataset(y_temp, batch_size),2)
    else:
        print('Input path is incorrect.')
        return

    # 学習を開始
    from keras import backend as K
    model = create_model(learning_rate, batch_size, time_steps, feature_columns)
    
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
                       patience=40, min_delta=0.0001)
    
    mcp = ModelCheckpoint(output, monitor='val_loss', verbose=1,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)

    csv_logger = CSVLogger(os.path.join(logdir, 'training_log_' + time.ctime().replace(" ","_") + '.log'), append=True)
    
    history = model.fit(x_t, y_t, epochs=epochs, verbose=2, batch_size=batch_size,
                        shuffle=False, validation_data=(trim_dataset(x_val, batch_size),
                        trim_dataset(y_val, batch_size)), callbacks=[es, mcp, csv_logger])
    
    # 学習結果を元に予測のテスト
    predict_time_series(trim_dataset(x_test_t, batch_size), trim_dataset(y_test_t, batch_size), logdir, output, time_steps, output_gap, batch_size, output_index, False)


# 与えられた入力データから連続で予測を行い、正解データとの比較を行う。
def predict_time_series(x_real, y_real, logdir, input_modelfile, time_steps, output_gap, batch_size, output_index, plot_graph):
    # モデルをロード
    model = load_model(input_modelfile)

    # 入力データを正規化
    x_real, data_range, data_shift = min_max_scaler_converter(x_real, output_index)
    
    y_pred = model.predict(trim_dataset(x_real, batch_size), batch_size=batch_size)
    y_pred = y_pred.flatten()
    y_real = trim_dataset(y_real, batch_size)
    error = mean_squared_error(y_real, y_pred)
    print("Error is", error, y_pred.shape, y_real.shape)

    # 予測結果の集計
    y_pred_org = []
    gain = []
    total_gain = []
    total_gain_temp = 0
    win_count = 0
    lose_count = 0
    
    print(str(len(data_range)))
    for i in range(len(data_range) - 1):
        # 正規化したデータを元の値に戻す
        pred_temp = y_pred[i] * data_range[i] + data_shift[i]
        y_pred_org.append(pred_temp)
        if i > output_gap:
            # 予想した[output_gap]分後の価格が現在価格より高いなら、ロングで注文を入れると仮定
            if pred_temp > y_real[i - output_gap]:
                # [output_gap]分後の予想価格から実際の価格を引いて、差額を利益として計上
                gain_temp = pred_temp - y_real[i]
            # 予想価格が現在価格より低いなら、ショートで注文を入れると仮定
            else:
                gain_temp = y_real[i] - pred_temp
            
            # 勝率計算に使うため、カウントを行う。
            if gain_temp > 0:
                win_count += 1
            else:
                lose_count += 1
                
            gain.append(gain_temp)
            total_gain_temp += gain_temp
            total_gain.append(total_gain_temp)
    
    if plot_graph:
        # pyplotによる結果のグラフ化(予測値と実績値)
        from matplotlib import pyplot as plt
        fig = plt.figure()
        plt.plot(y_pred_org)
        plt.plot(y_real)
        plt.title('Bitcoin pred-vs-real(Time steps:' + str(time_steps) + ', Batch_size:' + str(batch_size) + ')')
        plt.ylabel('Price')
        plt.xlabel('Minutes')
        plt.legend(['Prediction', 'Real'], loc='upper left')
        plt.savefig(os.path.join(logdir, 'pred_vs_real_BS'+str(batch_size)+"_"+time.ctime()+'.png'))
        plt.close(fig)

        # 毎分ごとの利益の計測
        from matplotlib import pyplot as plt
        fig = plt.figure()
        plt.plot(gain)
        plt.plot(total_gain)
        plt.title('Gain graph(Time steps:' + str(time_steps) + ', Batch_size:' + str(batch_size) + ')')
        plt.ylabel('Price')
        plt.xlabel('Minutes')
        plt.legend(['gain', 'total'], loc='upper left')
        plt.savefig(os.path.join(logdir, 'gain_BS'+str(batch_size)+"_"+time.ctime()+'.png'))
        plt.close(fig)
        
    print("WIN RATE:", str(win_count / (win_count + lose_count)))


# inputのCSV全データを使用して予測し、正解データとの比較を行う。
def predict_all(input_csv, input_modelfile, logdir, time_steps, output_gap, batch_size, feature_columns, output_index):
    # モデルのロード
    model = load_model(input_modelfile) 

    # 入力データの読み込み
    df_input = pd.read_csv(input_csv, engine='python')
    
    x_predict = df_input.loc[:,feature_columns].values
    
    x_t, y_t = build_timeseries(x_predict, output_index, time_steps, output_gap)
    x_t = trim_dataset(x_t, batch_size)
    y_t = trim_dataset(y_t, batch_size)

    predict_time_series(x_t, y_t, logdir, time_steps, output_gap, batch_size, output_index, False)


# inputのCSVの末尾のデータから、次の1つの数値を予測する
def predict_next(input_csv, input_modelfile, logdir, time_steps, output_gap, batch_size, feature_columns, output_index):
    # モデルのロード
    model = load_model(input_modelfile) 

    # 入力データの読み込み
    df_input = pd.read_csv(input_csv, engine='python')
    
    df_predict = df_input[-time_steps:]

    # データの正規化
    min_max_scaler = MinMaxScaler()
    x_predict = min_max_scaler.fit_transform(df_predict.loc[:,feature_columns].values)

    # 予測
    y_pred = model.predict(np.tile(x_predict, (batch_size,1,1)), batch_size=None)
    y_pred = y_pred.flatten()

    # 正規化したデータを元に戻す
    y_pred_org = (y_pred * min_max_scaler.data_range_[output_index]) + min_max_scaler.data_min_[output_index]
    
    final_value = y_pred_org[0]
    print("PREDICTED VALUE: ", final_value)
    
    return final_value


def main(argv):
    input = ''
    filename = ''
    logdir = ''
    is_train = False
    try:
        opts, args = getopt.getopt(argv,"i:o:l:t:",["input=","output=","logdir=","train="])
    except getopt.GetoptError:
        print('test.py -i <input> -o <output> -l <logdir> -t <train>\n\nex)\npython3 lstm_predictor.py -i input/fx_btc_jpy.csv -o best_model.h5 -l outputs -t false')
        sys.exit(2)
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print('test.py -i <input> -l <logdir> -t <train>')
            sys.exit()
        elif opt in ("-i", "--input"):
            input = arg
        elif opt in ("-o", "--output"):
            filename = arg
        elif opt in ("-l", "--logdir"):
            logdir = arg
        elif opt in ("-t", "--train"):
            # Trueなら学習を行ってモデルを保存。FalseならCSVの末尾データだけを使って未来の予測をする
            is_train = True if arg == "true" else False

    if is_train:
        train(input, filename, logdir, epochs=10, learning_rate=0.0001, 
            time_steps=15, output_gap=5, batch_size=10, 
            feature_columns=["best_bid", "best_ask", "total_bid_depth", "total_ask_depth", "volume_by_product"], 
            output_index=0)
    else:
        predict_next(input, filename, logdir, time_steps=15, output_gap=5, 
            batch_size=10, feature_columns=["best_bid", "best_ask", "total_bid_depth", "total_ask_depth", "volume_by_product"], 
            output_index=0)

if __name__ == "__main__":
    main(sys.argv[1:])
