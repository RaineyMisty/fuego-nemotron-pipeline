# named bucket

## named bucket
- 这一部分是关于固定的bucket查询
- 我们的网页首页会提供20个固定的bucket
    - 关于固定bucket的设计，我们可以采用新闻界的习惯，找出20个相距比较远并且比较大众的话题
    - 每一个话题作为semantic map里面的一个独立点，设计一套key然后用相同的embedder生成vector。描述固定bucket definitions 储存为静态文件，vectors 储存为可重建 cache，加载时生成
- 通过map模块给出的矩阵方法得到距离固定话题最近的10篇文章，输出编号

## Bucket definition
每个 bucket 保存：
- id
- name
- semantic definitions / prototype texts
- optional description
例如 Technology:
    - "technology, software, computing, digital products and infrastructure"
    - "artificial intelligence, semiconductors, cloud computing and robotics"
    - "consumer electronics and emerging technologies"
然后这个东西再经过embedder生成vector。

## 阈值
> 找到top10只是一般情况。
- 如果整个空间都没有10篇文章，也不用硬找，有多少给多少。
- 在空间质量很差的时候，设定阈值，比如说相关性只有0.3的东西就不需要硬凑数了。

## 测试
```{bash}
python -B -m integration.smoke_named_buckets
python -B -m unittest discover -s test -p 'test_named_buckets.py' -v
```

## Fixed buckets

- Politics
- World Affairs
- Economy
- Business
- Technology
- Science
- Health
- Environment
- Energy
- Education
- Crime and Justice
- Transportation
- Housing
- Sports
- Entertainment
- Arts and Culture
- Gaming
- Food
- Travel
- Weather
