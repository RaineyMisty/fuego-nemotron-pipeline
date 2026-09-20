# article_processing

## article_processing模块核心功能
- 将文本转化成需要的数据

## 设计步骤
1. 流水线式提示词撰写
    - 用英文写。大概就是：你是一个新闻审稿人，请对这篇文章进行分析「article」，选出20个词你觉得这篇文章最重要的和最能概括的，或者根据文章意思生成，但不能选介词或者冠词。并且根据新闻内容和选出来的词，生成一个新闻摘要供读者阅读，不包括广告。相关数据附上：「magedata」等等。
2. 丢给ai
3. 取出结果
    - 输出应该是json之类的东西，取出其中需要的数据：文章概括和文章摘要。

## 测试方法
1. 调用前配置环境变量
```{bash}
export NVIDIA_API_KEY='你的key'
export NVIDIA_MODEL='nvidia/nemotron-3.5-lightning-30b-a3b'
```
2. 运行测试文件
```{bash}
python3 -m unittest discover -s tests -p "test_*.py" -v
```
3. 运行Smoke测试文件
```{bash}
python3 -B -m integration.smoke_article_processing \
  --article integration/article_sample.txt \
  --timeout 20
```

## 后续改进
- `article_processing`的第100行有一个判断是关于输出的key数量是否是20个，我改成了至少是20个，但是实际上这样写还是不太好。同时如果小于20个要重新跑，这个在最后的逻辑里面要考虑到。
- 关于输出格式其实没有一个规范，后期可以加上比如说key多少个，summary长度多少，这个可以作为config写到外面来。
- 文章的生成有时候会出现api访问失败，要在后续的总和模块中考虑到重新生成，不要丢失数据。