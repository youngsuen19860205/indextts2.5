# gen_txt/ 随机克隆目标文本目录

供 `python -m indextts_batch.random_clone` 使用的目标文本池：每个 `.txt`
文件是一条待合成文本（UTF-8 编码），空文件会被跳过并打印警告。运行时会从
本目录中随机选择一条文本作为待合成内容。

可以用 `<名称>.<语种>.txt` 的形式指定该条文本的语种，覆盖 `--lang` 默认值，
例如 `hello_en.en.txt`。支持的语种：`zh`、`en`、`zhen`（中英混读）、`ja`、
`es`、`ar`。

示例文件：`hello_zh.txt`。
