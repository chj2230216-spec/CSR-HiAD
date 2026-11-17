# CSR-HiAD 首轮核心机制验证实验设计：MVTec-2K bottle / capsule

状态：v1.0；静态审查完成，实验未运行。审查基准 CSR-HiAD main @ b83ce679a4f47f84c02bd7ba3346f346032acee3；HiAD upstream @ 54f7c2e49956cffa231d1ec6e19a373c68a6254f。

范围：仅正常样本训练的异常识别与定位，不研究功能区域语义、Mamba、PaDiM、4K、整板及缓存。bottle、capsule 分别建模。当前 Git 树没有 MVTec-2K 数据，实际本地样本数量和掩码待核验。本轮是探索性机制筛查，不是泛化验证。

## 1. 科学问题与假设

HiAD 先平均尺度特征，再检测异常；CSR 保留尺度差异，以粗尺度预测细尺度，并进行联合最近邻评分。首轮应回答：条件预测是否在相同输入、训练数据、检测器、特征及记忆库预算下提供超出直接拼接、普通标准化的收益？

- H1：固定权重融合可能丢失判别信息，拼接优于融合。
- H2：条件创新相对直接拼接及简单残差具有增量价值。
- H3：该增量不能由普通标准化、容量或 Coreset 重新选择单独解释。
- H4：局部机制有效后，恢复 SCA 和整图低分支仍有收益。
- 任一假设均允许被否证；不以新增模块掩盖失败。

## 2. 设计缺陷及理论边界

### 2.1 本质是学习度量，而非显式条件异常概率模型

源码 hiad/csr_model.py:39–104，行向量约定：

    r = f - c
    W,b = argmin mean_i(||r_i-c_i W-b||²) + lambda ||W||²
    e = f - c(I+W) - b
    z = [c,e]
    z_CSR = (z-mu)/max(sigma,epsilon)

f,c 为已对齐、聚合、降维的细、粗尺度特征。给定 c,e 可恢复 f；多尺度可从粗到细递推。在已投影空间内，原始拼接、残差、条件创新间是可逆仿射变换，不能声称后两者创造新信息。对两条样本，中心化抵消：

    d_CSR² = ||delta_c / sigma_c||²
             + ||(delta_f-delta_c(I+W)) / sigma_e||²

sigma 为标准差下限修正后各分量的尺度。该距离等价于拼接向量上的正定二次型度量，且改变 Coreset 选择。当前 PatchCore 未显式拟合 p(c)p(e|c)，也没有分别计算结构/细节异常分数。应准确称为“跨尺度条件创新特征与联合记忆库”。

**关键缺失对照：concat_std、residual_std。** conditional 对 csr 只能测出标准化在条件表示上的作用，不能排除普通标准化。若 CSR 不优于上述控制组，条件预测独立贡献未成立。后续可比较普通协方差白化、低秩度量，但不是首轮必做。

### 2.2 同总维度不代表细粒度信息保留公平

hr_csr_patchcore.py:61–141：fine 使用 D 维细特征；fused 先在 backbone map 融合后聚合为 D；concat 等每尺度先聚合到 D，再 adaptive_avg_pool1d 至 D/2。D=384 时每尺度仅192维，可能损失细节，投影路径也不同。

主对照固定总 D=384（最终存储预算）；补充相同共享投影 P、每尺度192维的 paired_fine、paired_fused，使同一投影下的 fine/fused/concat/CSR 比较可追溯。前两者总维度192，不属于等总维度组。预注册 D=768 敏感性检查用于容量诊断。原版1024维另报，不能当作同预算对照。

### 2.3 线性条件模型与标准化可能放大正常扰动

每个 task 仅有线性 W,b 和共享对角方差，不能描述多模态条件分布或位置相关异方差。低方差通道可能放大边界、插值、反光误差。std_floor=0.01 为绝对量纲阈值。须报告独立正常样本预测 R²、残差方差、相关性、条件数、floor 命中率。粗尺度来自同一图像，仍包含目标异常信息；不输入 fine 不等于消除异常泄漏。

### 2.4 对齐与定位精度

对齐采用整数 crop、双线性插值及 OpenCV resize。深层特征差分不是严格频域分解。固定2048/512/stride512、尺度0,1，检查16主patch、4个1024上下文区域、坐标和批量一致性。原实现可能重复编码上下文，本轮不改。layer2/layer3 的下采样本身可能令极小缺陷消失。

PatchCore common.py 使用 FAISS IndexFlatL2/GpuIndexFlatL2，输出平方L2；近邻评分取 k 个距离均值，不额外开方。RescaleSegmentor 对 patch 图 sigma=4 平滑，BaseDetector.post_processing 又对整图 sigma=4 平滑；低分支还有放大效应。主实验保持同一后处理，另诊断平滑前后，不只给 CSR 关闭平滑。

## 3. 实现与公平性审查

| 优先级 | 源码位置/问题 | 后果与处理 |
|---|---|---|
| P0 | csr_model.py:58–104 缺 concat_std、residual_std | 正式消融前必须增加；共用拟合行、floor及统计估计方式 |
| P0 | runner:185–200 仅 calibration=tail 时预留正常数据 | none/tail 同时改变训练及聚类样本；阶段A统一none，阶段B固定fit/cal manifest |
| P0 | runner:91–112 目录未含seed、维度、校准等 | 结果易覆盖；显式独立run root，后续config hash/no-overwrite |
| P0 | runner:114–157 eval-only 校验不完整并仍复制YAML；同名config可覆盖 | 完整有效配置、backbone hash、数据manifest；eval-only只读并完整校验 |
| P0 | 现有CI仅NumPy单元测试和syntax compile | 无真实GPU、FAISS、crop/reload测试；先smoke |
| P0 | trainer.py:195–320 常规模式只输出日志/可视化，未保存无损预测 | 增加按原图raw score导出；可先封装return_results_only=True，不以可视化PNG当原始分数 |
| P0 | metrics_gpu.py 默认512，GT最近邻缩小；2048全量堆叠GPU | 小缺陷消失或OOM；主指标用原尺寸，按图导出、分块评价 |
| P1 | 各模式重新Coreset，reservoir候选有限 | 记录同源候选行ID及数量；新增固定bank行ID诊断以隔离度量 |
| P1 | trainer.py:223–228 测试集全局min-max；metrics报告最优阈值 | 排序通常不变但绝对阈值依赖test；最优F1为oracle，部署阈值独立正常校准 |
| P1 | csr_calibration.py 正常像素池化、尾部饱和；融合后未校准图像阈值 | 记录tail参考数、饱和比例与空间相关；保留raw，不称异常概率 |
| P1 | SCA每次重新聚类，实际task数可能<4 | 首轮A2O隔离路由；阶段B冻结task manifest |
| P1 | device/trainer预加载裁切图像，CSR两遍抽特征 | reservoir不等于系统流式；测RSS/VRAM、训练时间 |
| P1 | checkpoint只检查mode、D等有限配置 | reload一致性，校验尺度、权重、投影、版本及有效配置 |

P0表示正式科学结论前需解决，并不代表每项已触发运行错误。此次只提交实验协议，不修改算法。

## 4. 数据核验与冻结

预期目录：

    data/MVTec-2K/
      bottle/train.jsonl
      bottle/test.jsonl
      capsule/train.jsonl
      capsule/test.jsonl
      ... JSONL引用的图像和mask ...

data_root 为 data/MVTec-2K，而非类别子目录。JSONL至少包含 filename、clsname、label、label_name；异常test记录须有mask。路径相对data_root。上游 data/README.md 有下载入口。不得用普通MVTec AD替换。MVTec-2K为HiAD构造的高分辨率benchmark，含超分辨率及合成微小异常；本轮不自行合成训练异常。

只读预检（本地运行，输出实际样本数和manifest）：

~~~bash
python - <<'PY'
import json, hashlib
from pathlib import Path
from collections import Counter
from PIL import Image
root = Path("data/MVTec-2K").resolve()
for cat in ("bottle", "capsule"):
    paths, hashes = {}, {}
    for split in ("train", "test"):
        p = root / cat / (split + ".jsonl")
        rows = [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
        assert rows, p
        paths[split], hashes[split] = set(), set()
        for r in rows:
            assert {"filename","clsname","label","label_name"} <= set(r)
            assert r["clsname"] == cat and r["label"] in (0,1), r
            if split == "train":
                assert r["label"] == 0, r
            f = (root / r["filename"]).resolve()
            assert f.is_file() and f not in paths[split], f
            paths[split].add(f)
            hashes[split].add(hashlib.sha256(f.read_bytes()).hexdigest())
            with Image.open(f) as im:
                assert im.size == (2048,2048), (f,im.size)
            if r["label"] == 1:
                m = (root / r["mask"]).resolve()
                with Image.open(m) as im:
                    assert im.size == (2048,2048) and im.convert("L").getbbox(), m
        print(cat, split, len(rows), Counter((r["label"],r["label_name"]) for r in rows),
              "manifest_sha256", hashlib.sha256(p.read_bytes()).hexdigest())
    assert not paths["train"] & paths["test"]
    assert not hashes["train"] & hashes["test"], "exact duplicate content"
PY
~~~

另检查共同来源原图及合成变体跨split（如有source ID按组）、近重复、mask面积及连通域分布。失败不得静默resize或删样本。训练不得使用test标签/mask决定参数，不假设样本数等于原始MVTec AD。

## 5. 实验阶段与固定参数

### Phase 0：运行前检查

执行pytest和compileall，检查CUDA、FAISS、backbone权重及依赖。真实训练推理smoke：bottle/fused、bottle/csr；检查完整batch及尾batch、16 patch覆盖、权重reload一致、finite scores、bank数量、输出图尺寸。保存环境、配置和日志。失败先修复。当前4项数学测试不能替代此步骤。

### Phase A：局部机制筛查

| 参数 | 固定设置 |
|---|---|
| 类别 | bottle / capsule分别训练 |
| 图像、patch、stride | 2048、512、512 |
| 尺度、融合权重 | 0 1 / 0.5 0.5 |
| Assignment | A2O，1个局部detector |
| thumbnail、calibration | 关闭 / none |
| 数据 | 全部正常train，官方test不变 |
| Backbone | wideresnet50, layer2/layer3，权重固定 |
| PatchCore | merge_size=3, pretrain_embed_dimension=1024, D=384, percentage=0.1, kNN=5 |
| 拟合/候选行上限 | 20000/50000；候选达上限时bank目标约5000 |
| ridge/floor | 0.01/0.01 |
| Batch | 2起试；OOM时对应对照统一降为1 |
| Seed | 42初筛；2025、2026所有模式复核 |
| 指标 | 主2048 P-AP及实例定位；辅512、I-AUC/P-AUC/AUPRO |

A2O并非部署推荐，而是消除专家分数、路由变化和低分支混杂。后续须恢复SCA。

| ID | mode | 机制 | 状态 |
|---|---|---|---|
| A0 | fine | 原始细尺度 | 已有 |
| A1 | fused | 固定融合，同预算HiAD | 已有 |
| A2 | concat | 直接拼接 | 已有 |
| A3 | residual | 粗尺度+相邻差分 | 已有 |
| A4 | conditional | 条件残差 | 已有 |
| A5 | csr | 条件残差+标准化 | 已有 |
| A6 | concat_std | 拼接+同样标准化 | 待实现，必做 |
| A7 | residual_std | 差分+同样标准化 | 待实现，必做 |

配对差值：A2-A1信息保留、A3-A2坐标变换、A4-A3条件预测、A5-A4标准化、A5-A6/A7条件建模在标准化后的独立价值。不得只比较A5和A1。

当前首批可尝试的命令（Bash；真实环境未验证；512指标仅作运行筛查、论文可比辅助指标）：

~~~bash
set -euo pipefail
python -m pytest -q tests/test_csr_model.py
python -m compileall -q hiad/csr_model.py hiad/csr_calibration.py hiad/detectors/hr_csr_patchcore.py runs/run_csr_patchcore.py
GPU=0
for cat in bottle capsule; do
  for mode in fine fused concat residual conditional csr; do
    seed=42
    out="results/round1/A/$cat/$mode/seed$seed"
    if [ -e "$out/console.log" ]; then
      echo "Existing run: $out; choose new run ID" >&2
      exit 1
    fi
    mkdir -p "$out"
    python runs/run_csr_patchcore.py \
      --data_root data/MVTec-2K --category "$cat" \
      --image_size 2048 --patch_size 512 --stride 512 \
      --ds_factors 0 1 --fusion_weights 0.5 0.5 \
      --assignment a2o --detector_number 1 \
      --no-use_thumbnail --calibration none \
      --embedding_dim 384 --batch_size 2 \
      --csr_fit_rows 20000 --csr_bank_rows 50000 \
      --csr_ridge 0.01 --csr_std_floor 0.01 \
      --mode "$mode" --seed "$seed" --gpus "$GPU" \
      --metric_size 512 \
      --checkpoint_root "$out/checkpoints" --log_root "$out/logs" \
      --vis_root "$out/vis" 2>&1 | tee "$out/console.log"
  done
done
~~~

初跑12次（2类×6现有模式）。先单独用上述参数和独立smoke目录运行bottle/fused、bottle/csr，成功后再运行矩阵。补齐A6/A7及无损导出评价后，正式Phase A共2×8×3=48次（满足协议且已保存足够数据的初跑可复用）。runner会在root后追加mode/image_size/assignment/category；报告实际路径。可尝试 --metric_size 2048 但须防OOM，推荐导出后离线评价，不重复训练。

补充固定每尺度投影与共用bank行ID对照。独立Coreset是部署合理设置，却不能单独证明度量收益。

### Phase B：完整HiAD框架验证

仅在Phase A通过后开展。每类别每seed固定相同训练图像及同一SCA task manifest（目标4）。比较 fused、concat_std、residual_std、csr，先仅高分支，再恢复thumbnail=512。预先冻结原图级fit80%/cal20%划分；若拟合尾部及最终图像阈值均需正常数据，预先分离cal-score与cal-decision或另设独立正常验证。样本不足不得宣称可靠低FPR保证。

同一checkpoint比较raw与tail；当前runner直接切换none/tail会改变fit集，不能当公平校准消融。保留raw、局部/全局及最终图，记录饱和比例。部署阈值仅从独立正常最终图像分数确定。

原始HiAD参考：upstream run_patchcore.py默认O2O、1024维、候选无CSR上限、thumbnail等，不等于A1。其列表/布尔CLI有解析局限，采用默认参数或独立wrapper，记录有效设置。原论文与本地指标分开。

## 6. 特征诊断与评价

以下诊断和导出属待开发功能，不是当前runner已输出结果。

- 固定图像/patch/feature-grid坐标，保存paired fine/coarse、fused、concat、residual、innovation及标准化表示，以相同行ID比较；核查批量大小改变时对齐数值一致。
- 独立正常诊断样本R²=1-SSE/SST（SST=0记NA）、残差方差、相关性、floor命中；诊断用交叉拟合，不将test用于拟合。报告正常与异常分数分离度和实际bank行数。
- 按patch边界距离0–8、8–16、>16 px诊断正常响应；分别检查feature-grid、patch插值/平滑、整图输出。平滑敏感性对所有模式同设置。
- 2048 P-AP为主要指标；P-AUC、AUPRO@FPR<=0.30及512版本为辅助；I-AUC/I-AP；保持相同16×16 max-average图像评分；test最优F1标记oracle。
- 原尺寸GT 8连通域。固定面积组1–41、42–419、420–4194、4195–41943、>=41944 px（约0.001%、0.01%、0.1%、1%分界）；小缺陷定义<=419 px。另记录bbox短边/长度；每组实际数量、空组NA。
- 阈值无关实例指标：全测试背景像素FPR 0.1%、1%时，报告各组宏平均PRO（test ROC诊断，非部署阈值）。独立正常阈值可用后，报告实例命中（GT覆盖>=10%）、充分覆盖（>=50%）、正常图像FPR、每图误报连通域数及面积。
- 原尺寸OOM时逐图保存raw float32分数、uint8 mask，外部排序求exact AP；如用固定histogram近似，记录bin与误差、在子集验证。512不代替2048。冻结AUPRO阈值数与实现版本。
- 效率：总训练wall-clock，固定设备/batch，warm-up后CUDA同步计单图preproc/backbone/query/stitch端到端延迟（至少20次，median/p95）；峰值VRAM allocated/reserved、RSS、checkpoint bytes、bank行数/维度。两遍特征提取须计入；参数量不等于速度。

## 7. 保存及统计

拟扩展输出：

    results/round1/
      environment.json
      data_manifest.json
      A/bottle/fused/seed42/
        console.log
        checkpoints/...
        logs/...
        vis/...
        run_manifest.json
        metrics.json
        image_scores.csv
        defect_instances.csv
        profile.json
        diagnostics.json
        predictions/{image_id}.npz
      summary.csv
      paired_deltas.csv

summary.csv建议字段：

    category,phase,mode,seed,assignment,M_actual,thumbnail,calibration,
    feature_dim,scale_dim,fit_n,cal_n,bank_n,metric_size,I_AUC,I_AP,P_AUC,
    P_AP,AUPRO,small_PRO_FPR001,small_PRO_FPR01,peak_vram_mb,peak_rss_mb,
    train_sec,median_ms,p95_ms,status,commit,config_hash

未实现/运行字段记NA，不填0或虚构结果。保留失败日志。逐类别报告、两类宏平均（不混像素池）、同seed配对差值。三seed共享test，反映训练随机性而非独立数据；配对bootstrap以图像为单位而非像素。两类不足以证明普适性。

## 8. 预注册阶段性目标（非实测）

- Gate 0：数据及端到端smoke通过，reload一致；同源fit图像、网格、总维度、候选行及后处理均可核对。
- Gate 1：A5相对A1两类宏平均2048 P-AP目标+2个百分点，且相对A6、A7均+1个百分点；至少2/3配对seed同方向。任一类别下降>2个百分点需解释，不能称全面稳定收益。
- Gate 2：小实例足量（建议每类>=10，报告实际计数）时，PRO@FPR1%目标较A1+5个百分点；样本不足不作通过声明。
- Gate 3：SCA+thumbnail保留核心增量；报告I-AUC及大缺陷指标，推理median增幅目标<=20%，未经实测不声明达成。
- Gate 1失败：若concat_std/residual_std足够，仅支持一般度量/标准化；需审视显式条件方差、局部匹配等而非堆叠模块。若多尺度均不优于fine，审查stride、对齐、容量、粗尺度异常泄漏；若仅tail或平滑改善，归因后处理。
- bottle/capsule用于探索，不在其test反复调参后宣称独立验证。下一轮使用未参与调试类别及真实RealIAD-HD。

## 9. 下一步开发优先级

1. 新增concat_std、residual_std、共享投影及固定bank诊断；数学/形状/checkpoint测试。
2. 冻结数据及任务manifest、完整有效config、禁止覆盖、只读eval；真实GPU smoke与无损预测导出。
3. 实现原尺寸评价、实例分组、特征诊断、profiling及批量汇总。
4. 核心通过后恢复SCA、thumbnail、独立正常校准；暂不扩展其他模型。

## 10. 源码证据

- [CSRTransform](https://github.com/chj2230216-spec/CSR-HiAD/blob/b83ce679a4f47f84c02bd7ba3346f346032acee3/hiad/csr_model.py)
- [CSR detector](https://github.com/chj2230216-spec/CSR-HiAD/blob/b83ce679a4f47f84c02bd7ba3346f346032acee3/hiad/detectors/hr_csr_patchcore.py)
- [Runner](https://github.com/chj2230216-spec/CSR-HiAD/blob/b83ce679a4f47f84c02bd7ba3346f346032acee3/runs/run_csr_patchcore.py)
- [Calibration](https://github.com/chj2230216-spec/CSR-HiAD/blob/b83ce679a4f47f84c02bd7ba3346f346032acee3/hiad/csr_calibration.py)
- [Trainer](https://github.com/cnulab/HiAD/blob/54f7c2e49956cffa231d1ec6e19a373c68a6254f/hiad/trainer/trainer.py)
- [PatchCore FAISS/smoothing](https://github.com/cnulab/HiAD/blob/54f7c2e49956cffa231d1ec6e19a373c68a6254f/hiad/detectors/patchcore/common.py)
- [Pixel metrics](https://github.com/cnulab/HiAD/blob/54f7c2e49956cffa231d1ec6e19a373c68a6254f/hiad/utils/metrics_gpu.py)
- [Dataset instructions](https://github.com/chj2230216-spec/CSR-HiAD/blob/b83ce679a4f47f84c02bd7ba3346f346032acee3/data/README.md)

结论：先证明条件创新度量超出直接拼接和标准化的独立判别价值，再讨论完整高分辨率系统的创新。目前尚无实际性能提升证据。
