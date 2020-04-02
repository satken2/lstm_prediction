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

""" trim_dataset

    Description:
        BATCH_SIZEからあふれた余分データを削除する。
        
    Args:
        input_data(numpy.ndarray): 整形したいnumpyデータセット
        batch_size(int): バッチサイズ

    Returns:
        バッチサイズで割り切れるように整形されたデータセットを返却します。
        元々割り切れる場合は入力値をそのまま返却します。
"""
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

""" build_timeseries

    Description:
        一続きの時系列データから入力データと結果データの組を作成します
        
    Args:
        input_data(numpy.ndarray): 2次元の時系列データ
        y_col_index(int): 結果のインデックス

    Returns:
        バッチサイズで割り切れるように整形されたデータセットを返却します。
        元々割り切れる場合は入力値をそのまま返却します。
"""
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

# モデルの作成
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

# 入力データを元に予測を行う
def predict_test(x_real_build, y_real, logdir, filename, time_steps, output_gap, batch_size, output_index, plot_graph):
    # モデルをロード
    model = load_model(filename)

    # 入力データを正規化
    x_real_build, data_range, data_shift = min_max_scaler_converter(x_real_build, output_index)
    
    y_pred = model.predict(trim_dataset(x_real_build, batch_size), batch_size=batch_size)
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
        plt.figure()
        plt.plot(y_pred_org)
        plt.plot(y_real)
        plt.title('Bitcoin pred-vs-real(Time steps:' + str(time_steps) + ', Batch_size:' + str(batch_size) + ')')
        plt.ylabel('Price')
        plt.xlabel('Minutes')
        plt.legend(['Prediction', 'Real'], loc='upper left')
        plt.savefig(os.path.join(logdir, 'pred_vs_real_BS'+str(batch_size)+"_"+time.ctime()+'.png'))

        # 毎分ごとの利益の計測
        from matplotlib import pyplot as plt
        plt.figure()
        plt.plot(gain)
        plt.plot(total_gain)
        plt.title('Gain graph(Time steps:' + str(time_steps) + ', Batch_size:' + str(batch_size) + ')')
        plt.ylabel('Price')
        plt.xlabel('Minutes')
        plt.legend(['gain', 'total'], loc='upper left')
        plt.savefig(os.path.join(logdir, 'gain_BS'+str(batch_size)+"_"+time.ctime()+'.png'))
        
    print("WIN RATE:", str(win_count / (win_count + lose_count)))

def __multi_file_reader(input_dir, batch_size, feature_columns, output_index):
    print("[__multi_file_reader] start")
    
    x_concat = None
    y_concat = None
    first_loop = True
    
    for file in os.listdir(input_dir):
        if file.endswith(".csv"):
            # 学習データの読み込み
            input_csv = os.path.join(input_dir, file)
            print("[__multi_file_reader] processing " + input_csv)
            df_input = pd.read_csv(input_csv, engine='python')
            print(df_input.dtypes)

            # 入力データを正規化
            min_max_scaler = MinMaxScaler()
            x_input = min_max_scaler.fit_transform(df_input.loc[:,feature_columns].values)

            del df_input

            # 入力データからインプットとアウトプットのセットを作成
            x_t, y_t = build_timeseries(x_input, output_index, time_steps, output_gap)
            
            del x_input
            
            # リストに追記
            if first_loop:
                x_concat = x_t
                y_concat = y_t
                first_loop = False
            else:
                x_concat = np.concatenate((x_concat, x_t))
                y_concat = np.concatenate((y_concat, y_t))
            
    
    x_concat = trim_dataset(x_concat, batch_size)
    y_concat = trim_dataset(y_concat, batch_size)
    print("[__multi_file_reader] row count x_concat is " + str(x_concat.shape))
    print("[__multi_file_reader] row count y_concat is " + str(y_concat.shape))
    
    return x_concat, y_concat

def __trim_dataframe(df_input, batch_size):
    print("[__trim_dataframe] start")
    no_of_rows_drop = len(df_input.index) % batch_size
    print("[__trim_dataframe] no_of_rows_drop: " + str(no_of_rows_drop))
    if no_of_rows_drop > 0:
        return df_input[:-no_of_rows_drop]
    else:
        return df_input

""" train(input_path, output, logdir)

    Description:
        指定したCSVファイルから学習を行います。
        単一のCSVを指定した場合はそのファイルから学習を行い、ディレクトリを指定した場合は配下のCSVを結合して学習します。
        
    Args:
        input_path(str): CSVまたはディレクトリのフルパス
        output(str): モデルデータの保存先ファイル名
        logdir(str): ログデータの保存先ファイル名

    Returns:
        None
"""
def train(input_path, output, logdir, epochs=10, learning_rate=0.0001, 
            time_steps=15, output_gap=5, batch_size=10, feature_columns=[], 
            output_index=0):
    if os.path.isdir(input_path):
        print("Input path is directory. Train with multi file mode")
        
        # 学習データの読み込み
        x_input, y_input = __multi_file_reader(input_path, batch_size, feature_columns, output_index)
        print("Batch trimmed size",x_input.shape, y_input.shape)
        
        # 学習用データとテスト用データに分ける
        train_count = int(y_input.shape[0] * 0.8)
        test_count = y_input.shape[0] - train_count
        
        print("Train count = ",str(train_count))
        print("Test count = ",str(test_count))
        
        # BATCH_SIZEで割り切れるようにする
        x_t = trim_dataset(x_input[:train_count], batch_size)
        y_t = trim_dataset(y_input[:train_count], batch_size)

        # テスト用データをさらにvalidateとtestに分ける
        x_val, x_test_t = np.split(trim_dataset(x_input[train_count:], batch_size), 2)
        y_val, y_test_t = np.split(trim_dataset(y_input[train_count:], batch_size), 2)
        

        print("Train data size",x_t.shape, y_t.shape)
        print("Test data size",x_test_t.shape, y_test_t.shape)
        print("Validate data size",x_val.shape, y_val.shape)
    
        print("Test size", x_test_t.shape, y_test_t.shape, x_val.shape, y_val.shape)
        
    elif os.path.isfile(input_path):
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
    print("checking if GPU available", K.tensorflow_backend._get_available_gpus())
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
    predict_test(trim_dataset(x_test_t, batch_size), trim_dataset(y_test_t, batch_size), logdir, output, time_steps, output_gap, batch_size, output_index, False)

def predict_next(input_csv, input_file, logdir, time_steps, output_gap, batch_size, feature_columns, output_index):
    # モデルのロード
    model = None
    try:
        model = load_model(input_file)
        print("Loaded saved model.")
    except FileNotFoundError:
        print("Model not found. Exiting...")
        return

    # 入力データの読み込み
    df_input = pd.read_csv(input_csv, engine='python')
    tqdm_notebook.pandas('Processing...')
    
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
    print("\n==========================")
    print("PREDICTED VALUE: ", final_value)
    print("==========================")
    
    return final_value

def predict_all(input_csv, input_file, logdir, time_steps, output_gap, batch_size, feature_columns, output_index):
    # モデルのロード
    model = None
    try:
        model = load_model(input_file) 
        print("Loaded saved model.")
    except FileNotFoundError:
        print("Model not found. Exiting...")
        return

    # 入力データの読み込み
    df_input = pd.read_csv(input_csv, engine='python')
    tqdm_notebook.pandas('Processing...')
    
    x_predict = df_input.loc[:,feature_columns].values
    
    x_t, y_t = build_timeseries(x_predict, output_index, time_steps, output_gap)
    x_t = trim_dataset(x_t, batch_size)
    y_t = trim_dataset(y_t, batch_size)

    predict_test(x_t, y_t, logdir, time_steps, output_gap, batch_size, output_index, False)
    
def main(argv):
    input = ''
    filename = ''
    logdir = ''
    is_train = False
    try:
        opts, args = getopt.getopt(argv,"i:o:l:t:",["input=","output=","logdir=","train="])
    except getopt.GetoptError:
        print('test.py -i <input> -o <output> -l <logdir> -t <train>\n\nex)\npython3 stock_pred_main.py -i inputs/ge.us.txt -o best_model.dat -l outputs -t false')
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
            is_train = True if arg == "true" else False

    if is_train:
        train(input, filename, logdir, EPOCHS)
    else:
        predict_next(input, filename, logdir, time_steps, output_gap, batch_size, feature_columns, output_index)
        #predict_all(input, filename, logdir, time_steps, output_gap, batch_size, feature_columns, output_index)

if __name__ == "__main__":
    main(sys.argv[1:])
