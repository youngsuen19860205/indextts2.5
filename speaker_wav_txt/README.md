# speaker_wav_txt/ 随机克隆参考音色目录

供 `python -m indextts_batch.random_clone` 使用的参考音色池。按 speaker 分子目录，
每个 speaker 下放置若干**同名**的 wav/txt 配对（UTF-8 编码）：

```
speaker_wav_txt/
├── alice/
│   ├── a.wav          # 参考音频
│   ├── a.txt          # a.wav 里实际说出的内容
│   ├── b.wav
│   └── b.txt
└── bob/
    ├── c.wav
    └── c.txt
```

要求与行为：

- `.wav` 与 `.txt` 的 basename 必须完全一致才算一个有效配对；缺少任意一边，
  或参考文本为空，该条目会被跳过并打印警告（不会导致整个 speaker 失败）。
- 一个 speaker 目录下只要存在至少一个有效配对即可被随机选中；完全没有有效
  配对的 speaker 会被跳过（并打印警告）。
- 参考音频建议：单人、无背景噪声/音乐、无混响，时长 3–15 秒。
- 本仓库不提供真实人声样例：请只使用**已获授权**的声音，禁止克隆未经许可的
  他人声音。

> 出于仓库体积与隐私考虑，`speaker_wav_txt/*/*.wav` 与 `speaker_wav_txt/*/*.txt`
> 已在 `.gitignore` 中忽略，请自行放置数据。
