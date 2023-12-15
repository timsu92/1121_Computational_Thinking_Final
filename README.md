# 1121 Computational Thinking Final

這個repo包含著我抓到的天氣資料，可以抓下來去訓練模型。
目前這些內容是用GitHub Actions自動抓的。但是他們伺服器的負載頗高，時不時會有些時間沒抓到的情形。因此我還有一個手動的備份伺服器在抓，但是只會在我發現有缺漏的資料的時候才會來手動加上。

`raw_weather_data`資料夾內包含的是抓下來的原始天氣資料，可以用以下的兩種方式利用：

### 使用`json2dataframe.py`
這個檔案適合Python模型組使用，可以將那些JSON原始檔案轉換成Pandas DataFrame。

### 使用`json2csv.py`
這個檔案適合rapidminer組使用，可以在執行Python程式後得到csv檔：

1. 執行Python程式
   ```sh
   python3 json2csv.py
   ```
