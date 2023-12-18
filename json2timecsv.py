import os
import typing as tp
import dateutil.parser as psr
from itertools import product
from functools import reduce
import pandas as pd

from util import translate_one

def fixDatetimeFormat(oneHistory: tp.Dict[str, tp.Any]):
  oneHistory["DateTime"] = psr.parse(oneHistory["DateTime"]).strftime(r"%d/%m/%Y %H:%M")
  oneHistory["DailyHigh_DateTime"] = psr.parse(oneHistory["DailyHigh_DateTime"]).strftime(r"%d/%m/%Y %H:%M")
  oneHistory["DailyLow_DateTime"] = psr.parse(oneHistory["DailyLow_DateTime"]).strftime(r"%d/%m/%Y %H:%M")
  return oneHistory

def preprocess_data(data: pd.DataFrame):
  # 處理無效值
  data.replace(-99.0, 0.0 , inplace=True)

  # 選擇需要的特徵( 降水量 / 氣溫 / 濕度 )
  features = ["DateTime", "Precipitation", "AirTemperature", "RelativeHumidity"]
  data = data[features]

  # 排序
  data.loc[:, 'DateTime'] = pd.to_datetime(data['DateTime'])
  data = data.sort_values(by="DateTime")

  # 將時間設為索引
  data.set_index('DateTime', inplace=True)

  # 補充缺失值
  data.interpolate(inplace=True) # 插值法

  return data

def append_hours(data: pd.DataFrame):
  features = ["Precipitation", "AirTemperature", "RelativeHumidity"]
  hourAnnotation = ["3hr", "2hr", "1hr", "now"]
  return pd.DataFrame(
    data=(reduce(
      lambda hr0,hr1 : hr0+hr1.values.tolist(),
      (data.iloc[hour, :] for hour in range(old_hour, old_hour + 4)),
      []) for old_hour in range(0, len(data) - 3)),
    columns=list(map(lambda column: "_".join(column), product(hourAnnotation, features)))
  )

def write_csv(data: pd.DataFrame):
  with open("weather.csv", "w") as file:
    file.write(",".join(data.columns) + "\n")
    for row in data.values:
      file.write(",".join(map(str, row)) + "\n")

def main():
  df = pd.DataFrame (translate_one(os.path.join(".", "raw_weather_data", file_name)) for file_name in os.listdir(os.path.join(".", "raw_weather_data")) if file_name.endswith(".json"))
  df_processed = preprocess_data(df)
  df_sequence = append_hours(df_processed)
  write_csv(df_sequence)

main()