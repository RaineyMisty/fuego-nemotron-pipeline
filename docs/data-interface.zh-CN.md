# 接口说明入口

当前接口以 [English API guide](data-interface.md) 为准，README 和代码注释使用简单 English。

新主流程分为两个阶段：

1. `/v1/ingest` 接收文章，Nemotron 提取 scheme，再按 scheme 分配 bucket，将正文和来源一起存入 SQLite。
2. `/v1/query` 将自然语言映射到 bucket IDs；`/v1/summary` 此时读取 bucket 内容并调用 Nemotron 生成 digest。`/v1/digest` 可以一次完成第二阶段。

`/v1/gdelt` 继续支持四列 CSV/JSON，默认服务会保存 metadata-only 记录并返回统计。没有正文时不会编造摘要；可用相同 ID 和 URL，通过 `/v1/ingest` 补入正文。

旧的 `run`、`feed` 命令和预生成 catalog 流程已移除。详见英文文档中的完整请求、返回格式、时间语义和限制。
