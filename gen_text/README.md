# gen_text/ 待合成文本目录

- 一个 `.txt` 文件代表一条待合成文本，UTF-8 编码，空文件会报错。
- 文件名（basename）会作为输出 WAV 名称的一部分。
- 可以用 `<名称>.<语种>.txt` 的形式指定该条文本的语种，覆盖 `--lang` 默认值，
  例如 `hello_en.en.txt`、`hello_ja.ja.txt`。
  支持的语种：`zh`、`en`、`zhen`（中英混读）、`ja`、`es`、`ar`。

示例文件：`hello_zh.txt`、`hello_en.en.txt`、`hello_ja.ja.txt`。
