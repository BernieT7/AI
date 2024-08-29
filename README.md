__投資理財顧問__
===
## 目標:
創造一個專屬於金融領域的聊天機器人
## Approch:
### 1.Intoduce OpenAI API:
(0) Deep learning, Transformer架構:參考台大李宏毅教授的深度學習課程(https://hackmd.io/@abliu/BkXmzDBmr)。
   
(1) GPT-3.5-turbo-instruct：
   當用戶輸入一個問題或指令後，模型通過理解該文本中的關鍵信息，然後在它的知識範圍內生成一個連貫且相關的回應。是一個利用深度學習中
   藉由Transformer的架構訓練出來的模型，也就是會根據上下文來決定每個單詞的重要性，從而更好地捕捉上下文關係。

(2) whisper-1:
   主要用於將音訊檔案中的語音轉換為文字。這項功能廣泛應用於語音轉錄、自動字幕生成等場景。一樣是由Transformer的架構訓練出來的模型。

(3) text-embedding-ada-002 
   將文字數據轉換為數值向量，這些向量可以用於多種自然語言處理任務。一樣是由Transformer的架構訓練出來的模型。
### 2.架構：
Step1: request夯鼠兄弟YouTube撥放清單網址
Step2: 將其撥放清單中所有影片的音訊以及標題透過whisper-1轉換為文字並儲存下來
Step3:
### 3.Code
引入OpenAPI套件以及.evn套件

.evn套件用於保存OpenAI的API Key，將其寫入自己電腦中的.env文檔中儲存，可以避免被其他人使用
```python
from openai import OpenAI
from dotenv import dotenv_values
config = dotenv_values('.env')
client = OpenAI(api_key=config["API_KEY"])
```
引入pytubefix中的YouTube, Playlist，定義一個將影片音訊轉換為文檔的function
```python
from pytubefix import YouTube, Playlist
def get_audio_text(audio_path, title):
  audio = open(audio_path, "rb")               # 讀取音檔
  res = client.audio.transcriptions.create(          # 利用whisper-1將音檔轉換為文字
      model="whisper-1",
      file=audio,
      prompt=title                      # 以影片標題引導模型的指示
  )
  return res.text                        # 回傳影片音檔文字
```
定義一個回傳影片標題及文字的function
```python
def get_video_title_text(video_url, audio_name):
  video = YouTube(video_url)                  # 創建YT影片
  stream = video.streams.filter(only_audio=True).first()    # 取得YT影片音檔
  audio_path = stream.download(filename=audio_name)       # 將音檔下載至本地端
  text = get_audio_text(audio_path, video.title)        # 取得文字
  return video.title, text                   # 回傳影片標題及文字
```
利用前兩個函式，定義一個回傳所有撥放清單中的所有影片標題及內容文字的函式
ˋˋˋ
def get_playlist_info(playlist_url):
  playlist = Playlist(playlist_url)                  # 取得playlist裡所有影片的網址
  videos_info = {}
  for idx, video_url in enumerate(playlist):
    title, text = get_video_title_text(video_url, f"{idx}.mp3")  # 取得所有影片標題及文字
    videos_info[title] = text                    # 存放所有影片標題及文字
  return videos_info                          # 回傳所有影片標題及文字
ˋˋˋ

## 使用資料：
YouTuber夯鼠兄弟的YouTube撥放清單影片
