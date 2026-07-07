# genie-meeting

PDF 簡報 + 會議錄影 → 按頁歸類的會議記錄:vision 模型把語音逐句對齊到 PDF 每一頁,產出含「語音時間戳 + PDF 頁碼」雙來源引用的報告。適用會議與課程。

## 需求

- genie-core(`[pdf][mlx]` extras)+ `ffmpeg`
- LM Studio:文字模型 + vision 模型(qwen3.6-35b 一顆兼任亦可)

## 用法

```bash
genie-meeting slides.pdf recording.mp4                # 輸出 slides_meeting_report/(與 PDF 同目錄)
genie-meeting slides.pdf recording.mp4 -o report/ \
    --text-model qwen3.6-35b-a3b-turboquant-mlx       # 不指定則自動挑選
```

| 參數 | 預設 | 說明 |
|---|---|---|
| `pdf` | — | 簡報 PDF |
| `video` | — | 會議錄影(影片或音檔) |
| `-o, --output` | PDF 同目錄 `<pdf>_meeting_report/` | 輸出目錄 |
| `--language` | zh | whisper 語言碼 |
| `--whisper-model` | medium | whisper 模型大小 |
| `--text-model` | 自動挑選 | 綜合報告用文字模型 |
| `--vision-model` | 自動挑選 | 逐頁對齊用 vision 模型 |
| `--url` | `http://localhost:1234/v1` | LM Studio API |

## 輸出

```
report/
  page_analyses/page_NNN.json   # 逐頁分析 checkpoint(resume 用)
  report.json                   # topics + 每頁討論內容 + relevance 分級
  report.md / report.html
```

## 斷點續跑(resume)

逐頁 vision 分析結果即時落盤;中斷後重跑同一指令,已分析的頁直接載入跳過 vision 呼叫,只補缺的頁與最終綜合。

## 行為說明

- 每頁只餵「估計時間點附近」的語音窗口給 vision 模型(長會議不會爆 context)
- 綜合報告走剝引句 → 樹狀合併 → 程式回填來源的流程(同 genie-transcript)
- 報告 parse 失敗 retry 一次後非零退出,不會產出只有 appendix 的空報告
