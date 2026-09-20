# AI模块

## ai模块功能
1. 读取 AI 配置
2. 调 NVIDIA / Nemotron API
3. 返回原始或标准化后的模型响应

## 使用方法
1. 调用前配置环境变量
```{bash}
export NVIDIA_API_KEY='你的key'
export NVIDIA_MODEL='nvidia/nemotron-3.5-lightning-30b-a3b'
```
2. 运行测试文件
```{bash}
python3 -m unittest discover -s tests -p "test_ai.py" -v
```