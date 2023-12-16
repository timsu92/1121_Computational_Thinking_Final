import os

from util import translate_one

def main():
    """這邊示範把當前目錄下的所有json檔合併成weather.csv，並且會把已經存在的weather.csv覆蓋"""

    history = [translate_one(os.path.join(".", "raw_weather_data", file_name)) for file_name in os.listdir(os.path.join(".", "raw_weather_data")) if file_name.endswith(".json")]

    with open("weather.csv", "w") as f:
        f.write(",".join(history[0].keys()) + "\n")
        f.writelines(map(lambda oneDay: ",".join(map(str, oneDay.values())) + "\n", history[1:]))

if __name__ == "__main__":
    main()
