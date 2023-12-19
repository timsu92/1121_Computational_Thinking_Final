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


def build_rnn_model(input_shape):
    model = Sequential()
    model.add(layers.SimpleRNN(94, input_shape= input_shape, kernel_initializer="glorot_normal"))
    model.add(layers.Dropout(0.11901741186214937))
    model.add(layers.Dense(units=2, activation="linear", kernel_initializer="glorot_normal"))
    model.compile( optimizer=RMSprop(learning_rate=0.002416477391612136),loss="mse", metrics=["mae"])

    return model


def main():
    # 轉換 json 為 dataframe
    folder_path = "1121_Computational_Thinking_Final/raw_weather_data"
    df = pd.DataFrame([translate_one(os.path.join(folder_path, file_name)) for file_name in os.listdir(folder_path) if file_name.endswith(".json")])

    df = df.sort_values(by="DateTime")
    df_processed = preprocess_data(df)
    time_window = 3  # 3小時

    split_index = int(len(df_processed) * 0.99) # 假設80%的數據用於訓練集
    X_train, X_test = df_processed[:split_index], df_processed[split_index:]
    y_train, y_test = df_processed[:split_index], df_processed[split_index:]

    #-----------------------------------------

    # 創建時間序列資料
    sequences , labels = create_sequences(df_processed, time_window)
    X_train_sequences, y_train_labels = create_sequences(X_train, time_window)
    X_test_sequences, y_test_labels = create_sequences(X_test, time_window)

    #------------------------------------------

    # 建立RNN模型
    model = build_rnn_model(input_shape=(time_window, df_processed.shape[1]))
    print("safe\n")

    #------------------------------------------
    # 模型訓練
    model.fit(X_train_sequences, y_train_labels, epochs=40 , batch_size=20)


    # Model Evaluation
    test_loss, test_acc = model.evaluate(X_test_sequences, y_test_labels)
    print("loss of test set : " , test_loss )
    print("accuaracy of test set : " , test_acc )

    # 進行預測
    print(" predicting : \n")

    # 預測降雨機率
    predictions = model.predict(X_test_sequences)
    print(predictions[-1])

    # print(y_test_labels[-1])



if __name__ == "__main__":
    main()



