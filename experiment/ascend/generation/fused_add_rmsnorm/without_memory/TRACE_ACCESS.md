# Trace 发布范围

trace.jsonl：Codex CLI逐事件结构，保留事件顺序、类型、状态、退出码及用量。
session.jsonl：Codex Session逐记录结构，保留原时间戳和序号。
events.jsonl：API请求/响应时间、HTTP状态、Token用量、工具调用名称与类别、全部固定评测及resume事件。
三个文件保留原记录数量；trace-export-audit.json提供数量核验。版本源码和测量见versions/及result.json。

为保护私有资料，所有自由文本（模型正文、Prompt、工具命令/参数/输出、错误正文等）统一脱敏。工具类别为粗粒度分类，不是原命令。stderr.log和final_message.txt也仅保留脱敏占位，不能视为原文。
本公开包是结构化脱敏Trace，支持时序、调用与测量核验，但不支持还原完整推理/调试正文。原始API请求/响应、Codex Trace及Session完整保留本地；未上传任何私有专家资料。
