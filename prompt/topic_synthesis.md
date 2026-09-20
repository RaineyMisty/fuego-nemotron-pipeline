# topic synthesis

## 桶分析
这个模块是关于前面分出来的桶，每个桶里面都有默认是10篇文章，要用ai来进行分析总结这十篇文章，总结出一个桶标题（一个词或词组的topic），和一个桶summary。

## 工具
- `nemotron-3.5-lightning-30b-a3b`

## 实现方案
- 把收到的文章列表提取出之前生成好的summary
- 设计提交给`Nemotron`的prompt
    -提示词用英文写，固定流水线，把10个（或者几个）文章summary+那些megadata嵌进去。提示词大概是这样：你是一个新闻主编，以下是你今天收到的十篇新闻，你是一个对社会动向很敏锐的主编，你一看到这十篇文章就能想到这十篇文章合起来讲了一个什么内容，不一定要限定为一个词，但要Identify the strongest shared topic supported by the articles. Do not force unrelated articles into a common narrative. 你要写以下东西：1、这十篇文章相关的一个主题，不能太大众化，也不能太生僻；2、关于十篇文章，围绕这个主题，写一个主题的summary（大概五行字这样，自然语言）。
- 丢给ai
    - 使用之前写的ai模块
- 输出的结果整理为桶标题和桶summary

## 测试
```{bash}
python -B -m integration.smoke_topic_synthesis --articles /path/to/summaries.json
```