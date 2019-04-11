from selenium import webdriver
import numpy as np
import pandas as pd
import datetime
import sys
sys.setrecursionlimit(10000)
today=datetime.date.today()

#csvファイル（既存データ）の読み込み
# df=pd.read_csv("既存データ先")
df=pd.DataFrame(index=[],columns=["id","name","date"])

#ブラウザ開く(most-popular today)
driver=webdriver.Edge('/home/riku/anaconda3/envs/tik/lib/python3.6/site-packages/chromedriver_binary/chromedriver')
for i in np.arange(1,11):
    if i==1:
        driver.get("https://www.tiktok.com/share/music/6655934560741739265?language=ja&utm_campaign=client_share&app=tiktok&utm_medium=ios&iid=6548133476875372289&utm_source=copy")
    else:
        a="https://www.tiktok.com/share/music/6655934560741739265?language=ja&utm_campaign=client_share&app=tiktok&utm_medium=ios&iid=6548133476875372289&utm_source=copy/%s/" % str(i)
        driver.switch_to.default_content()  #最上位に戻す
        driver.get(a)
#    time.sleep(5)

    #動画のリスト取得
    l=driver.find_element_by_class_name("video-cover")
    # l=l.find_element_by_class_name("main")
    # l=l.find_element_by_class_name("row row3")
    # l=l.find_element_by_class_name("content")
    # l=l.find_element_by_class_name("thumbs-items")

    #そのページの動画一覧
    for i in l.find_elements_by_class_name("thumb"):
        a=i.find_element_by_tag_name("a")
        index=a.get_attribute("href").find('/',27)
        #id,nameをDataFrameに追加
        df=df.append(pd.DataFrame([[int(a.get_attribute("href")[27:index]),a.get_attribute("href")[index+1:-1],today.isoformat()]],columns=["id","name","date"]),ignore_index=True)
        #df.append(df2,ignore_index=True)

#重複データの削除（前残し）
df=df.drop_duplicates(["id"])
#dataframeの出力
df.to_csv("./",index=False)


driver.close()
driver.quit()
