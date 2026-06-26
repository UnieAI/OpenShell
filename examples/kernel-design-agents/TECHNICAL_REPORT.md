# Kernel Design Agents 技術報告

## 1. 目標

本工作將 Kernel Design Agents（KDA）整合進 OpenShell，形成一套以
definition 為中心的 FlashInfer-Bench 任務優化流程。

這次整合的目標不是做一個泛用聊天迴圈，而是建立一條可重現的任務執行
管線，能夠：

- scaffold 一個真實 submission workspace
- 由平面化 config 產生 task contract
- 執行 draft-only 規劃
- 在真實 workload 子集合上執行 baseline 驗證
- 在已驗證的 source candidate 之上執行一輪或多輪 optimization
- 將 immutable 的每次 attempt artifact 與 machine-readable summary
  持久化到 host 端

目前整合聚焦於 MLSys26 FlashInfer contest dataset，以及其中的
3 個 task family / 5 個 concrete definition：

- `moe`
- `dsa_paged`
- `gdn`

## 2. 設計目標

這次整合以以下限制與原則為核心：

- 使用者入口盡量簡單：一個 host-side `optimize.sh`
- 盡量把修改收斂在 `examples/kernel-design-agents/`
- 使用真實 FlashInfer starter kit，而不是 toy scaffold
- 保留 Docker 作為執行邊界，但讓 host-side 檔案直接持久化
- 將 orchestration state 與 candidate implementation 檔案分離
- 讓執行結果可以從 host-side `results/` 恢復與續跑
- 讓優化重心落在 `solution/`，尤其是 `solution/cuda/kernel.cu`

## 3. 高層架構

KDA 整合目前可以分成四層。

### 3.1 Host 端入口

`optimize.sh` 是主要的 host-side 入口。

責任包括：

- 載入平面 YAML config
- 解析 workspace 路徑
- scaffold 或 refresh task workspace
- 產生 `TASK_CONTRACT.md`
- 產生 `config.toml`
- 將 dataset 與 workspace mount 進 Docker
- 將 execution mode、workload scope、optimization 控制參數傳入 container

### 3.2 Starter-kit workspace

starter kit vendored 在：

- `starter-kit/`

它提供：

- `scripts/pack_solution.py`
- `scripts/check_cuda_extension.py`
- `scripts/run_local.py`
- `solution/` submission 目錄結構

OpenShell 並沒有取代 starter-kit 的執行模型，而是在其外層包上 task
orchestration 與 result persistence。

### 3.3 Container-side runner

`scripts/run-kda-draft.sh` 是 container 內的 orchestrator。

責任包括：

- 執行 draft mode 或 execute mode
- materialize immutable attempt workspace
- 組裝最終模型 prompt
- 強制執行 execution contract
- 每輪執行後正規化 artifact
- 將結果 canonicalize 成 `record.json`、`current.json`、
  `benchmark.csv` 與 `candidates.jsonl`

### 3.4 Prompt / contract 層

prompt template 位於：

- `prompts/basic-flow.md`

動態生成的 task contract 位於：

- `TASK_CONTRACT.md`

prompt 提供模型流程層面的行為指引。
task contract 提供模型當前任務的具體內容、validation command、
evaluation command，以及 promotion criteria。

目前整合把 `TASK_CONTRACT.md` 視為 active task 的 source of truth，
但 orchestration layer 會在執行前先正規化 contract，使 artifact path
保持明確且一致。

## 4. 以 Definition 為中心的 Preset 模型

目前的模型是 definition-centric，而不是 benchmark-script-centric。

Preset 路徑格式：

```text
config/presets/<family>/<definition>-<profile>.yml
```

每個 preset 固定：

- 一個 concrete dataset `definition`
- 一個 result `workspace`
- 一個 workload profile
- 一條 build path
- 一組預設 contract

核心 workload profile 有兩種：

- `w1`
  使用一個固定 UUID 作為 smoke validation workload
- `wa`
  對該 definition 的全部 workload 執行 benchmark

這點很重要，因為 contest dataset 裡面有很多 task，也有很多 workload。
orchestration 不應該 hard-code 成只支援單一 MoE task。

## 5. 執行模式

### 5.1 Draft mode

Draft mode 是純規劃模式。

預期輸出：

- `docs/draft.md`

不會記錄任何真實 implementation attempt。

### 5.2 Execute mode，但不帶 `--kernel-optimize`

這是 baseline smoke / baseline revalidation 模式。

行為：

- 若尚無 validated baseline，就先 bootstrap 一個 baseline
- 若 baseline 已存在，就只 revalidate baseline
- 不進入更深層的 optimization branch

這個模式主要用於 correctness bring-up 與 `w1` smoke test。

### 5.3 Execute mode，且帶 `--kernel-optimize`

這是 optimization 路徑。

行為：

- 若尚無 source candidate，先 bootstrap baseline
- 然後從 current source candidate 繼續優化
- 只有當當前 attempt 被 promoted，才會推進到下一層 depth
- 用 `--max-depth` 限制最多推進幾層 promoted depth
- 用 `--branch` 限制每一層 depth 最多可嘗試幾個 attempt

在使用者可見的命名模型中：

- baseline 使用 `baseline/bN`
- optimization depth 使用 `dN/bM`

active source candidate 透過 `current.json` 追蹤。

## 6. Immutable Attempt Workspace 模型

每個 execute attempt 都在自己的 immutable workspace 中執行。

例如：

- `results/<task>/baseline/b1`
- `results/<task>/baseline/b2`
- `results/<task>/d1/b1`
- `results/<task>/d2/b1`

這個模型取代了 restore-over-write 的舊做法。

好處是：

- 先前證據完整保留
- rejected candidate 仍可回頭檢查
- summary 可從 record 重新生成
- source candidate 可由 immutable history 重新推導

Attempt materialization 規則如下：

- `scripts/`、`images/` 與 contract doc 等 support files
  從 result root 複製
- candidate code 與 `config.toml`
  從被選中的 source attempt 複製
- `solution/` 因此會繼承自 parent candidate
- orchestration support scripts 不會再從舊 attempt 繼承

這樣可以避免舊 attempt-local helper script 汙染後續 run。

## 7. Task Contract 與 Artifact Contract

這次整合的一個核心要求，是 benchmark artifact path 必須明確。

目前 contract 明確要求：

- `python scripts/check_cuda_extension.py --log-file runs/check_cuda_extension.txt`
- `python scripts/run_local.py --log-file runs/run_local.txt --results-json runs/run_local_results.json`

系統已不再依賴 orchestration 注入的 log-path environment variable
來決定這兩個 artifact 路徑。

這樣設計的原因是：直接用明確 command argument，
比靠隱含 env 更容易稽核、更穩定，也比較不容易把 artifact 寫到錯誤的
workspace root。

## 8. 結果模型

orchestration layer 會同時保存 human-readable 與 machine-readable state。

### 8.1 Attempt-local artifacts

每個 attempt 會有：

- `docs/draft.md`
- `docs/plan.md`
- `runs/check_cuda_extension.txt`
- `runs/run_local.txt`
- `runs/run_local_results.json`
- `outputs/execution-summary.agent.md`
- `outputs/last-message.md`

### 8.2 Workspace-level derived state

每個 task result root 會有：

- `current.json`
- `benchmark.csv`
- `candidates.jsonl`
- `outputs/execution-summary.md`

### 8.3 Canonical attempt record

每個 attempt 會整理成：

- `<step>/<branch>/record.json`

重要欄位包括：

- `step`
- `branch`
- `parent_step`
- `parent_branch`
- `status`
- `correctness`
- `metrics`
- `solution_snapshot_dir`
- `config_snapshot`
- `task_contract_snapshot`

Promotion 語意如下：

- `validated`
  baseline 驗證通過
- `promoted`
  optimization candidate 通過 correctness，且 latency 優於 source
- `rejected`
  optimization candidate correctness 通過，但 latency 沒有優於 source
- `failed`
  candidate correctness 或 runtime 失敗

## 9. Source Candidate 語意

Source candidate 不是「最新一次 attempt」。
它是 validated baseline 的最新 promoted descendant。

目前 source candidate 會從 immutable record 重新推算，
而不是只單純信任 `current.json`。

這可以避免以下 stale-pointer 問題：

- 使用者手動刪除部分目錄
- 舊 run 寫入了過時的 `current.json`
- current source 之後還存在 rejected candidate

對於 `w1 + max-depth=1`，這可提供預期語意：

- `d1/b1` 若 promoted，就成為新的 source candidate
- `d1/b1` 若 rejected，source candidate 仍維持 baseline

## 10. Dataset 與 Task Context

當 `fib_dataset_path` 可用時，host runner 會把 MLSys26 dataset
以 read-only 方式 mount 進 container，並匯出 `FIB_DATASET_PATH`。

整合也會自動生成：

- `docs/task-context.md`

此檔案會摘要：

- active dataset definition
- selected workloads
- baseline solution reference context

這讓模型在深入讀 library internals 之前，先有一份本地 task summary。

## 11. Build 模型

目前 generic preset 共用同一條主要 build path：

- `language: cuda`
- `binding: torch`
- `entry_point: kernel.cu::kernel`
- `source_dir: cuda`

這表示 active optimization target 通常就是：

- `solution/cuda/kernel.cu`

orchestrator 也會明確告知模型：
不能悄悄把 active benchmark path 改成 `binding.py::...`。

## 12. W1 驗證行為

目前 smoke-validation workflow 為：

1. pack solution
2. 若 active build 為 CUDA + `binding=torch`，先做 compile-smoke
3. 只 benchmark 被選中的 `w1` workload UUID
4. 比較 candidate latency 與 source latency

對於 `w1`，單一 workload UUID 現在已被納入證據模型，
應該從：

- `run_local_results.json`

流入：

- `record.json`
- `current.json`
- `benchmark.csv`

這樣才能保證每個結果都綁定到實際使用的 smoke workload。

## 13. 分析工具

`solution.sh` 是結果彙整工具。

責任包括：

- 走訪一個或多個 result root
- 讀取 immutable `record.json`
- 解析 current source candidate
- 輸出平面 performance table
- 輸出對應 candidate 的 solution file path

這讓使用者不需要手動打開所有 JSON，也能快速檢查結果。

## 14. 目前已實作的主要整合修正

目前版本已包含幾個重要 orchestration fix。

### 14.1 顯式 artifact path

Evaluation command 會被正規化，確保 benchmark artifact 永遠使用：

- `runs/run_local.txt`
- `runs/run_local_results.json`

### 14.2 Python 3.9 相容性

整合層與 starter-kit helper script 已調整，
避免在 execution-critical code path 使用 Python 3.10 專屬 union type syntax。

### 14.3 Immutable attempt workspace

Attempt 目錄採 immutable materialization，不再覆寫舊 run。

### 14.4 Root 與 attempt script 汙染修正

新的 attempt 會從 result root 複製 orchestration support scripts，
而不是從前一個 attempt 複製，避免把 stale support script 向前傳遞。

### 14.5 Source candidate 重新推算

`current_record_field()` 現在會優先從 immutable record 重新推算
source candidate，而不是直接信任 stale 的 `current.json`。

### 14.6 W1 workload UUID 傳遞

Canonicalized metrics path 現在會保留單一 workload 的 UUID，
讓 `w1` 的證據鏈保持完整。

## 15. 目前範圍與邊界

目前這套整合已經能穩定解決的問題包括：

- definition-centric preset 選擇
- host 可見且可持久化的結果
- draft / baseline / optimize 三種 execution mode
- immutable baseline 與 depth attempt history
- source-candidate tracking
- `w1` smoke workload 執行
- 在 MoE、DSA、GDN 三個 family 間共用同一套 generic orchestration

這套整合本身尚未直接解決：

- task-level kernel optimization 品質
- `wa` 大範圍 performance validation 策略
- 跨 workload generalization 的自動保證
- 模型在困難 kernel task 上的 search quality

也就是說，orchestration layer 的目的是讓真實 kernel optimization
變得可執行、可稽核，而不是保證模型一定能找到更快的 kernel。

## 16. 後續建議重點

後續主要工作重心應繼續放在 task-level optimization，
而不是再擴張 orchestration 介面。

建議優先順序：

- 強化各 task family 的 solution template
- 補強仍偏弱的 baseline implementation
- 在 `w1` 穩定後，再逐步擴大到可控的 `wa` 驗證
- 持續把優化重心放在 `solution/cuda/kernel.cu`
- 除非控制力真的不足，否則避免再增加新的 user-facing flag

## 17. 總結

OpenShell 中的 KDA 整合，現在已經是一套真實 execution system，
而不是只有 prompt 的展示範例。

它的核心特性包括：

- 以 definition 為中心的設定模型
- host-side 可持久化結果
- immutable attempt history
- 明確的 validation / evaluation artifact path
- 可跨 depth / branch 追蹤的 source candidate
- 能直接接上真實 FlashInfer contest dataset 與 starter kit

這讓 `examples/kernel-design-agents/` 已能作為 MLSys26 contest task 的
實際 kernel optimization baseline。
