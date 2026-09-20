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
3. 运行Smoke测试文件
```{bash}
python3 -B -m integration.smoke_ai --timeout 20
```

## Nemotron测试
> 有时候因为各种原因测试失败了，有可能是API远端的问题，可以跑以下代码来测试哦
```{bash}
curl -v \
  --connect-timeout 5 \
  --max-time 30 \
  https://integrate.api.nvidia.com/v1/chat/completions \
  -H "Authorization: Bearer $NVIDIA_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
    "messages": [
      {
        "role": "user",
        "content": "Say hello"
      }
    ],
    "chat_template_kwargs": {
      "enable_thinking": false
    },
    "max_tokens": 64,
    "stream": false
  }'
```