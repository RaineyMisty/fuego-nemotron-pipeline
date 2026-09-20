# query bucket

## query bucket模块
> 跟name bucket差不多，不同的是单点查询
- 会获取来自backend端的用户具体topic查询
- 对用户的查询进行embedding然后生成一个vector
- 用map里面给定的接口进行查询与该topic相近的10个新闻
- 输出文章列表

## 阈值
- 同样也有阈值限制，找不到不用硬找

## 测试
```{bash}
python -B -m integration.smoke_query_buckets
python -B -m unittest discover -s test -p 'test_query_buckets.py' -v
```