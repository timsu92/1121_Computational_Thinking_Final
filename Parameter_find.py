# %%
import os

if not os.path.isdir("1121_Computational_Thinking_Final"):
  os.system("git clone https://github.com/timsu92/1121_Computational_Thinking_Final.git")
else:
  os.system(f"cd 1121_Computational_Thinking_Final && git pull")

# %%
import json
import pandas as pd
import numpy as np
import os
# from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop


def translate_one(path: str):
    with open(path, 'r') as f:
        data = json.load(f)
    station = data["records"]["Station"][0]
    ele = station["WeatherElement"]
    return {
        "DateTime": station["ObsTime"]["DateTime"],
        "Precipitation": ele["Now"]["Precipitation"], #降雨量
        "AirTemperature": ele["AirTemperature"], #當日氣溫
        "RelativeHumidity": ele["RelativeHumidity"], #當日濕度
    }

def preprocess_data(data):
    # 處理無效值
    data.replace(-99.0, 0.0 , inplace=True)

    # 選擇需要的特徵( 降水量 / 氣溫 / 濕度 )
    features = ["Precipitation", "AirTemperature", "RelativeHumidity"]

    # 將時間設為索引
    data['DateTime'] = pd.to_datetime(data['DateTime'])
    data.set_index('DateTime', inplace=True)

    # 選擇需要的特徵
    data = data[features]

    # 補充缺失值
    data.interpolate(inplace=True) # 插值法

    return data

def create_sequences(data, time_window):
    sequences, labels = [], []
    for i in range(len(data) - time_window):
      sequence = data.iloc[i:i+time_window].values
      label_temperature = data.iloc[i+time_window, 1]  # 氣溫
      label_humidity = data.iloc[i+time_window, 2]  # 濕度
      sequences.append(sequence)
      labels.append([label_temperature, label_humidity])  # 將氣溫和濕度作為一個列表添加到 labels 中
    return np.array(sequences), np.array(labels)


# def build_rnn_model(input_shape):
#     model = Sequential()
#     model.add(layers.SimpleRNN( 6 , input_shape= input_shape ))
#     model.add(layers.Dropout(0.2)) # 增加dropout曾，以防止過你合
#     model.add(layers.Dense(units=2, activation='relu'))
#     model.compile(optimizer=RMSprop(learning_rate=0.02, momentum=0.9, epsilon=1e-07), loss='mean_squared_error' , metrics=["mae"] )
#     return model


def main():
    # 轉換 json 為 dataframe
    folder_path = "1121_Computational_Thinking_Final/raw_weather_data"
    df = pd.DataFrame([translate_one(os.path.join(folder_path, file_name)) for file_name in os.listdir(folder_path) if file_name.endswith(".json")])

    df = df.sort_values(by="DateTime")
    df_processed = preprocess_data(df)
    time_window = 3  # 3小時

    # print(df_processed)
    split_index = int(len(df_processed) * 0.95)  # 假設80%的數據用於訓練集
    X_train, X_test = df_processed[:split_index], df_processed[split_index:]
    y_train, y_test = df_processed[:split_index], df_processed[split_index:]

    #-----------------------------------------

    # 創建時間序列資料
    sequences , labels = create_sequences(df_processed, time_window)
    X_train_sequences, y_train_labels = create_sequences(X_train, time_window)
    X_test_sequences, y_test_labels = create_sequences(X_test, time_window)


    #------------------------------------------
    # 設定TensorBoard
    # %load_ext tensorboard

    # Remove any logs from previous runs
    # os.system(f"rm -rf ./logs/")

    # Create TensorBoard log directory
    from datetime import datetime
    from tensorflow.keras.callbacks import TensorBoard

    logdir = os.path.join("logs", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = TensorBoard(logdir, histogram_freq=1)

    # Start the TensorBoard
    # %tensorboard --logdir logs


    #------------------------------------------
    # 超參數尋優
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    # Setup the HParams directory
    hparams_dir = os.path.join(logdir, 'validation')

    # Define Hyper-parameters
    HP_RNN_UNITS = hp.HParam('rnn_units', hp.IntInterval(30, 100))
    HP_DROPOUT = hp.HParam('dropout', hp.RealInterval(0.1, 0.3))
    HP_LEARNING_RATE = hp.HParam('learning_rate', hp.RealInterval(0.0001, 0.003))

    HPARAMS_SEARCH = [HP_RNN_UNITS, HP_DROPOUT, HP_LEARNING_RATE]

    # Set the Standard of Evaluation
    METRIC_MAE = "mae"

    # Setup the HParams directory for search
    hparams_search_dir = os.path.join(logdir, 'hparams_search')

    # Creating & configuring a log file writer for search
    with tf.summary.create_file_writer(hparams_search_dir).as_default():
        hp.hparams_config(
            hparams=HPARAMS_SEARCH,
            metrics=[hp.Metric(METRIC_MAE, display_name='MAE')]
        )

    # Count the total run
    total_run = 60

    curr_run = 1
    for curr_run in range(1, total_run+1):
        hparams_search = {
            HP_RNN_UNITS: HP_RNN_UNITS.domain.sample_uniform(),
            HP_DROPOUT: HP_DROPOUT.domain.sample_uniform(),
            HP_LEARNING_RATE: HP_LEARNING_RATE.domain.sample_uniform()
        }

        # Show something on Standard Output for progress
        print('--- Starting HParams Search trial: {} of {}'.format(curr_run, total_run))
        print({h.name: hparams_search[h] for h in hparams_search})
        search_run_name = "_".join(map(lambda item: f"{item[0]}({item[1]})", {h.name: hparams_search[h] for h in hparams_search}.items()))

        # Write information into log
        with tf.summary.create_file_writer(os.path.join(hparams_search_dir, search_run_name)).as_default():
            hp.hparams(hparams_search)

            # Create & Train the model
            model = Sequential()
            model.add(layers.SimpleRNN(hparams_search[HP_RNN_UNITS], input_shape=(time_window, df_processed.shape[1]), kernel_initializer="glorot_normal"))
            model.add(layers.Dropout(hparams_search[HP_DROPOUT]))
            model.add(layers.Dense(units=2, activation="linear", kernel_initializer="glorot_normal"))
            model.compile(
                optimizer=RMSprop(learning_rate=hparams_search[HP_LEARNING_RATE]),
                loss="mse",
                metrics=["mae"])
            model.fit(X_train_sequences, y_train_labels, epochs=160, batch_size=20,
                        validation_data=(X_test_sequences, y_test_labels),
                        callbacks=[tensorboard_callback])

            # Evaluate the Accuracy & Write to log
            test_loss, test_acc = model.evaluate(X_test_sequences, y_test_labels)
            tf.summary.scalar(METRIC_MAE, test_acc, step=10)

        curr_run += 1

    #------------------------------------------

    # 建立RNN模型
    # model = build_rnn_model(input_shape=(time_window, df_processed.shape[1]))
    # print("safe\n")

    #------------------------------------------
    # 模型訓練
    # model.fit(X_train_sequences, y_train_labels, epochs=100 , batch_size=20, validation_data=(X_test_sequences, y_test_labels), callbacks=[tensorboard_callback])


    # # Model Evaluation
    # test_loss = model.evaluate( X_test_sequences ,y_test_labels )
    # print("Loss of Test Set:", test_loss[0])

    # # 進行預測
    # print( " predicting : \n")

    # # 預測降雨機率
    # predictions = model.predict( X_test_sequences )
    # print( predictions )

    # print(y_test)


if __name__ == "__main__":
    main()



