import pandas as pd
import os

from util import translate_one

def main():
    """這邊示範把當前目錄下的所有json檔合併成一個DataFrame，然後每3個column print一次"""
    
    df = pd.DataFrame([translate_one("./raw_weather_data/" + file_name) for file_name in os.listdir("./raw_weather_data") if file_name.endswith(".json")])

    for i in range(0, len(df.columns), 3):
        print(df.iloc[:, i:i+3])

# 如果你直接用python執行這個檔案的話，main函式才會被執行
if __name__ == "__main__":
    main()
