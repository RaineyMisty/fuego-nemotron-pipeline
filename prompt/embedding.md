# embedding

## embedding模块
1. 输入text
2. 经过Embedder
3. 生成normalized float vector
> 这一部分主要是embedder的设计

## Embedder设计
- 使用`all-MiniLM-L6-v2`模型作为Embedder，向量空间大小是384维。
- 向量输出前进行标准化，所有点其实都在一个 384-dimensional space 的 unit sphere 上，具有很好的几何结构。
- 直接用python库里面的这个模型就可以了。

## 测试
1. 单元测试
```{bash}
python -B -m unittest discover -s test -p 'test*embedding.py' -v
```
2. smoke测试
```{bash}
# python -B -m integration.smoke_embedding --cache-folder work/models
python -B -m integration.smoke_embedding \
  --cache-folder work/models \
  --local-files-only
```

## 注意事项
- 下载安装fastembed
- 输出结果除了向量还有一堆东西，后面整合的时候要注意。