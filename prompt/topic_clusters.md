# topic clusters

## K-mean
- 从map获取完整向量
- 进行k-mean
    - 此处只关注算法本身。数学上欧式距离和 cosine similarity 有直接关系：\[||a-b||^2 = 2 - 2\cos(a,b)\]
- 输出存储为
    1. cluster centers
    2. article → cluster assignment

## 注意事项
- 后面map模块在添加新的新闻时不用立马更新k-mean，囤100个再更新（或者显式调用更新）

## 测试
```{bash}
python -B -m integration.smoke_topic_clusters
python -B -m unittest discover -s test -p 'test_topic_clusters.py' -v
```

## 问题
- smoke测试没有数据，未经检验。