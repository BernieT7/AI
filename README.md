__投資理財顧問__
===
## 致謝：
感謝Grandmacan團隊的小白老師，對我在Python學習的過程中提供了極大的幫助和支持。無論是在解決技術難題還是理解複雜的概念，小白老師
總是耐心細緻地為我解答，讓我在每一次學習中都有所收穫。特別是在我遇到困惑的時候，老師的講解讓問題迎刃而解，使我對Python有了更加
深刻的認識。真的非常感謝小白老師，這段學習之旅有您的指導，讓我成長了許多。此專案是在小白老師的教導下完成的
## 目標:
創造一個專屬於金融領域的聊天機器人
## Approch:
### 1.Intoduce OpenAI API:   
(1) GPT-3.5-turbo-instruct：

   當用戶輸入一個問題或指令後，模型通過理解該文本中的關鍵信息，然後在它的知識範圍內生成一個連貫且相關的回應。是一個利用深度學習中
   藉由Transformer的架構訓練出來的模型，也就是會根據上下文來決定每個單詞的重要性，從而更好地捕捉上下文關係。

展示使用方式：
```python
res = client.completions.create(              # AI會依據我的訊息給我回應
    model="gpt-3.5-turbo-instruct",
    prompt="訊息",                      # 傳訊息給AI
    max_tokens=300,                    # 設定AI輸出最大值
    stop="",                       # 設定遇到特定值就停止回應
    n=3,                         # 設定n個回應
    echo=True,                      # 設定是否要覆誦我的訊息，預設為false, 不會包含在tokan裡面
    temperature=1,                   # 設定AI的回應的創意度(0-2)，預設為1，數字越小越古板，數字越大越有創意
    top_p=1,                      # 設定AI的回應的創意度(0-1)，預設為1，數字越小越古板，數字越大越有創意
    frequency_penalty=0,                # (懲罰會累積)避免或增加使用重複的字(-2-2)，預設為0，>0懲罰，會減少重複;<0鼓勵，會增加重複
    presence_penalty=0,                 # (只會懲罰一次)避免或增加使用重複的字(-2-2)，預設為0，>0懲罰，會減少重複;<0鼓勵，會增加重複
    stream=False                     # 是否流式(一個步驟一個步驟)輸出，把過程顯示出來，要搭配for loop
)
```

(2) whisper-1:

   主要用於將音訊檔案中的語音轉換為文字。這項功能廣泛應用於語音轉錄、自動字幕生成等場景。一樣是由Transformer的架構訓練出來的模型。

展示使用方式：
```python
audio_file = open("音檔檔名", "rb")               # 開啟並讀取語音檔案
res = client.audio.transcriptions.create(               # 做語音辨識並且轉換成文字，輸出輸入的語言相同
    model="whisper-1",
    file=audio_file,
    prompt="提示"                         # 提示，可以提升模型輸出的準確度，提示的語言必須和語音的語言相同
)
```

(3) text-embedding-ada-002: 

   將文字數據轉換為數值向量，這些向量可以用於多種自然語言處理任務。一樣是由Transformer的架構訓練出來的模型。

展示使用方式：
```python
res = client.embeddings.create(
      model="text-embedding-ada-002",
      input=text                         # 輸入欲轉換文字
  )
```
補充：Deep learning的Transformer架構，可參考台大李宏毅教授的深度學習課程(https://hackmd.io/@abliu/BkXmzDBmr)。
### 2.架構：
Step1: request柴鼠兄弟YouTube撥放清單網址。

Step2: 將撥放清單中所有影片的音訊以及標題透過whisper-1轉換為文字並儲存下來。

Step3: 透過text-embedding-ada-002將這些文字轉換為向量並且儲存為此投資理財顧問的資料庫。

Step4: 搭建投資理財顧問，將輸入的文字轉換為向量後，計算該向量與資料庫中所有的向量的距離並找出最近的那一個。

Step5: 利用此最近的向量作為gpt-3.5-turbo-instruct回答輸入問題的基礎，並回傳模型判斷最適合的答案。
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
    query_embedding: List[float],  # 輸入向量向量，表示為一個浮點數列表
    embeddings: List[List[float]],  # 資料庫向量的列表，每個嵌入向量也是一個浮點數列表
    distance_metric="cosine",  # 距離度量方法，預設為 "cosine"（餘弦距離）
) -> List[List]:  # 函式返回一個二維列表，儲存輸入向量與每個資料庫向量之間的距離
    """返回輸入向量與每個資料庫向量之間的距離。"""

    # 定義一個字典，將不同的距離度量方法名稱與對應的距離計算函式進行映射
    distance_metrics = {
        "cosine": spatial.distance.cosine,  # 餘弦距離，用於度量兩個向量之間的夾角餘弦值
        "L1": spatial.distance.cityblock,  # L1 距離（曼哈頓距離），即各坐標軸上的距離總和
        "L2": spatial.distance.euclidean,  # L2 距離（歐幾里得距離），即直線距離
        "Linf": spatial.distance.chebyshev,  # L∞ 距離（切比雪夫距離），即各坐標軸上距離的最大值
    }

    # 使用列表生成式，計算查詢輸入向量與每個資料庫向量之間的距離，並將距離存入列表
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
接著，就可以搭建一個投資理財的顧問了
```python
def finance_consult(question):
    # 獲取投資理財問題的文字並且轉換為向量
    question_embeddings = get_embedding(question)

    # 計算並儲存問題的向量與所有資料庫的向量的距離
    dist = distances_from_embeddings(question_embeddings, df["embeddings"])

    # 找到最近鄰的索引
    nearest_idx = indices_of_nearest_neighbors_from_distances(dist)

    # 初始化變量來儲存最近的文本片段
    nearest_text = ""
    
    # 從最近鄰中提取前兩個最接近的文本片段，並將它們合併到一個字符串中
    for i in range(2):
        nearest_text += df["split_text_list"][nearest_idx[i]] + '\n'

    # 構建要發送給 GPT-3.5-turbo-instruct 的提示語，包含問題和最接近的文本片段
    prompt = f"""
    你是我的投資理財顧問，請根據以下內容回答此問題:{question}
    如果沒有100%確定，就回答'我不知道'

    ###
    內容:
    {nearest_text}
    ###

    """
    
    # 使用 OpenAI API 來獲取 GPT-3.5-turbo-instruct 的回應
    res = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=500,
        temperature=0
    )

    # 返回 GPT 模型生成的文本回應
    return res.choices[0].text
```
最後，透過gradio快速建立一個互動介面，將模型變成網頁應用
```python
import gradio as gr

demo = gr.Interface(
    fn=finance_consult,
    inputs="text",
    outputs="text",
    title="投資理財顧問",
    description="輸入您的問題:",
    allow_flagging="never"
)

demo.launch(debug=True)
```
### Test our consultant's capability

## Pros & Cons
優點：
   1. 使用OpenAI API建構聊天機器人，無需自行開發或訓練模型，就能使用許多相當優秀的模型，這大大縮短了開發時間
      和成本。
   2. 支持多種語言，能夠滿足全球市場的需求，適合開發國際化應用。
   3. 可以通過設置不同的參數（如創意度、輸出數量等）來調整聊天機器人的行為，從而定制其回應風格和語氣。
   4. 由於資料庫僅限於金融知識(柴鼠兄弟YouTube撥放清單)，運算效率較其他資料量的的聊天機器人快   
缺點：
   1. 使用OpenAI API是需要錢的，不同的模型也有不同計費方式，因此，隨著使用量的增加，API 請求的成本可能會變得
      很高，特別是對於大型或高頻率使用的應用，這可能會導致成本增加。
   2. 使用 OpenAI API 意味著對第三方服務的依賴，如果 OpenAI API 出現故障或變更，可能會影響到應用的正常運行。
   3. GPT 模型對於一般知識和常識性問題表現優秀，但對於非常專業或技術性的問題，它的回應可能不如專業模型那麼精確。
## 使用資料：
YouTuber夯鼠兄弟的YouTube撥放清單影片
