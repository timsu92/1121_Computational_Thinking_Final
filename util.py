import json

def translate_one(path: str):
    """將一個json檔轉為字典。
    裡面除了有包含觀測時間、降雨量、氣溫、相對濕度，還有一些應該用不到的欄位：氣壓、風向、風速、日高溫、日高溫時間、日低溫、日低溫。
    與此同時，我已經丟棄一些欄位了。如果對那些欄位有興趣的話，我再把他們加入。
    資料有時候是無效的，例如我有注意到降雨量會出現-99.0的情形。我猜是代表那時後沒有下雨的意思

    Args:
        path (str): json檔路徑
    """
    with open(path, 'r') as f:
        data = json.load(f)
    station = data["records"]["Station"][0]
    ele = station["WeatherElement"]
    return {
            "DateTime": station["ObsTime"]["DateTime"],
            "Precipitation": ele["Now"]["Precipitation"],
            "AirTemperature": ele["AirTemperature"],
            "RelativeHumidity": ele["RelativeHumidity"],
            "AirPressure": ele["AirPressure"],
            "WindDirection": ele["WindDirection"],
            "WindSpeed": ele["WindSpeed"],
            "DailyHigh_AirTemperature": ele["DailyExtreme"]["DailyHigh"]["TemperatureInfo"]["AirTemperature"],
            "DailyHigh_DateTime": ele["DailyExtreme"]["DailyHigh"]["TemperatureInfo"]["Occurred_at"]["DateTime"],
            "DailyLow_AirTemperature": ele["DailyExtreme"]["DailyLow"]["TemperatureInfo"]["AirTemperature"],
            "DailyLow_DateTime": ele["DailyExtreme"]["DailyLow"]["TemperatureInfo"]["Occurred_at"]["DateTime"],
            }
