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

(3) text-embedding-ada-002: 

   將文字數據轉換為數值向量，這些向量可以用於多種自然語言處理任務。一樣是由Transformer的架構訓練出來的模型。
### 2.架構：
Step1: request夯鼠兄弟YouTube撥放清單網址
Step2: 將撥放清單中所有影片的音訊以及標題透過whisper-1轉換為文字並儲存下來
Step3:
### 3.Code
引入OpenAPI套件以及.evn套件

.evn套件用於保存OpenAI的API Key，將其寫入自己電腦中的.env文檔中儲存，可以避免被其他人使用
pandas跟numpy基本上是ML, DL必備套件
```python
from openai import OpenAI
from dotenv import dotenv_values
import pandas as pd
import numpy as np
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
```python
def get_playlist_info(playlist_url):
  playlist = Playlist(playlist_url)                  # 取得playlist裡所有影片的網址
  videos_info = {}
  for idx, video_url in enumerate(playlist):
    title, text = get_video_title_text(video_url, f"{idx}.mp3")  # 取得所有影片標題及文字
    videos_info[title] = text                    # 存放所有影片標題及文字
  return videos_info                          # 回傳所有影片標題及文字
```
text-embedding-ada-002有設置token數的限制(8191個token)，原本一段影片的總token數遠超模型限制，所以要將每句話分割出來，分次embedding轉換
```python
def split_text(all_text, title):                
  text_list = all_text.split(' ')              # 把每句話分開
  text = title                        # 分開後前面加上title也就是後面引導模型的指示
  new_text_list = []                     # 接收加上title後的新text_list
  for idx, i in enumerate(text_list):            # 利用for loop昨上述任務
    text += f",{i}"
    if (idx+1)%50==0 or idx==len(text_list)-1:       # 每50句話一組或是做到最後一組就將結果存入new_text_list
      new_text_list.append(text)
      text = title
  return new_text_list
```
利用text-embedding-ada-002將文字向量化
```python
def get_embedding(text):
  res = client.embeddings.create(            # 利用模型將文字向量化
      model="text-embedding-ada-002",
      input=text                    # 輸入欲轉換文字
  )
  return res.data[0].embedding              # 回傳轉換結果
```
取得柴鼠兄弟YouTube playList的request
```python
playlist_url = "https://www.youtube.com/playlist?list=PLrZrfGLGySzcZoVhb4idy5B0XI25ZhnF7" # 引用YouTuber柴鼠兄弟的playList
playlist_info = get_playlist_info(playlist_url)
```
將playlist的所有影片音訊透過完成的函數轉換成文字後，轉換成CSV檔
```python
df = pd.DataFrame(list(playlist_info.items()), columns=["title", "text"])
df.to_csv("video_text.csv", index=False)
df = pd.read_csv("video_text.csv")
```
利用for迴圈跑過所有df中每部影片的文字並且使用split_text將其分割，避免token數超過模型限制的問題
```python
split_text_list = []
for idx, text in enumerate(df["text"].values):        # 把playlist的所有影片分割
  split_text_list += split_text(text, df["title"][idx])

df = pd.DataFrame(split_text_list, columns=['split_text_list'])
```
利用get_embedding將上面分割好的文字轉換成向量後做為此投資理財顧問的資料庫，存入df中
```python
split_text_embeddings = [get_embedding(i) for i in df["split_text_list"]]   # 將每段文字轉換成向量
df["embeddings"] = split_text_embeddings          # 創建新的column，將向量化後的結果展現出來
```
計算輸入文字轉換為向量後與資料庫中所有向量的距離，並且將這些向量由近到遠排序。
distances_from_embeddings: 計算輸入文字轉換為向量後與資料庫中所有向量的距離，並將結果存儲在列表中。
indices_of_nearest_neighbors_from_distances: 根據列表中的值從小到大進行排序，並返回排序後的索引。
```python
from typing import List  # 引入 List 類型，用於標示列表中元素的類型
from scipy import spatial  # 引入 scipy 的 spatial 模組，其中包含空間距離計算的工具

def distances_from_embeddings(
    query_embedding: List[float],  # 查詢嵌入向量，表示為一個浮點數列表
    embeddings: List[List[float]],  # 多個嵌入向量的列表，每個嵌入向量也是一個浮點數列表
    distance_metric="cosine",  # 距離度量方法，預設為 "cosine"（餘弦距離）
) -> List[List]:  # 函式返回一個二維列表，儲存每個嵌入向量與查詢嵌入向量之間的距離
    """返回查詢嵌入與嵌入列表之間的距離。"""

    # 定義一個字典，將不同的距離度量方法名稱與對應的距離計算函式進行映射
    distance_metrics = {
        "cosine": spatial.distance.cosine,  # 餘弦距離，用於度量兩個向量之間的夾角餘弦值
        "L1": spatial.distance.cityblock,  # L1 距離（曼哈頓距離），即各坐標軸上的距離總和
        "L2": spatial.distance.euclidean,  # L2 距離（歐幾里得距離），即直線距離
        "Linf": spatial.distance.chebyshev,  # L∞ 距離（切比雪夫距離），即各坐標軸上距離的最大值
    }

    # 使用列表生成式，計算查詢嵌入向量與每個嵌入向量之間的距離，並將距離存入列表
    distances = [
        distance_metrics[distance_metric](query_embedding, embedding)
        for embedding in embeddings
    ]

    return distances  # 返回計算出的距離列表

def indices_of_nearest_neighbors_from_distances(distances) -> np.ndarray:
    """根據距離列表返回最近鄰的索引列表。"""
    
    # 使用 numpy 的 argsort 函式，將距離從小到大排序，並返回對應的索引
    return np.argsort(distances)
```








## 使用資料：
YouTuber夯鼠兄弟的YouTube撥放清單影片
