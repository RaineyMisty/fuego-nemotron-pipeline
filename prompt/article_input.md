# article input

## article input 结构
```{json}
{
  "Record_ID":"20260912234500-81",
  "Publication_Date":1789256700000,
  "Source_Name":"idahostatejournal.com",
  "Title":"AP News in Brief at 6:04 p.m. EDT",
  "Article_Link":"",
  "Article_Text":""
}
```

## aritcle input 转化后的存储结构
```
articles
--------------------------------
id                  PRIMARY KEY
published_at
source
title
url
summary
semantic_text
```
bucket拆一个表单独存
```
article_buckets
--------------------------------
article_id
bucket_id
similarity
```

## 处理input
- 根据给定的input结构提取出两个部分
    - text+megadata
        - 这部分作为cache存储，给article_processing.py使用，生成summary等数据之后就覆盖
    - 文章相关信息
        - ID，链接之类，ID作为唯一识别手段（PK）
    - > 这部分写成json为了方便阅读，实际使用我们用SQLite来存，后续也用SQLite来查询
    ```{json}
    {
    "id": "20260912234500-81",
    "published_at": 1789256700000,
    "source": "idahostatejournal.com",
    "title": "AP News in Brief at 6:04 p.m. EDT",
    "url": "...",

    "summary": "...",
    "semantic_text": "...",

    "buckets": [
        {
        "bucket_id": "technology",
        "similarity": 0.78
        }
    ]
    }
    ```
- 调用article_processing.py进行处理
- 结果以SQLite的形式储存到/db里面

## 文章处理特例
> 长度大于10000字符的文章是那种很多篇的合集，会污染数据库。阈值设置在10000左右，太长的文章直接过滤掉