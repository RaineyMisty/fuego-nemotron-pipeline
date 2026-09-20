# write output

## write output
- 输出的标准格式在`/output/fuego_output_example.json`里面
- 需要topic_synthesis的输出来打包
- 需要访问数据库来获取article和bucket的信息
- 构建一个json打包函数，就是可以方便修改输出的结构
- 返回符合 schema 的 Python dict，并提供 JSON serialization；实际传输由上层接口负责

## 测试
```
python -B -m integration.smoke_write_output
python -B -m unittest discover -s test -p 'test_write_output.py' -v
```