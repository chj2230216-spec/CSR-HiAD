# CSR-HiAD 失效可能：基于 HLGFA 与 PyramidFlow 的反向机制审计

状态：2026-11-17，静态数学和源码审查。当前CSR代码基准9842e84bbc457b6d770e49a0d815cb27242c1ea7，未训练/运行真实数据。相关文档：HLGFA_MATH_AND_IMPLEMENTATION.md、PYRAMIDFLOW_MATH_AND_SOURCE.md、../ROUND1_EXPERIMENT_DESIGN.md（原计划）。本审计补充实验设计，不更改算法。两篇论文不支持直接断言CSR有效。

## 1. 最核心的设计偏差

HLGFA用学习后跨尺度误差直接评分；PyramidFlow对每个latent尺度与正常模板计算非负差再合成。当前CSRTransform产生e，再与c拼接做PatchCore，异常输出仅有联合近邻分数。它没有检验“细节偏离正常条件关系”的直接证据。模型可用正常近邻的非零e解释异常；这不等同于预测错误本身。

## 2. 严格推导当前岭回归的归纳偏置

按源码行向量，中心化C,F∈R^{N×d}，R=F-C。lambda>0：
    W=(C^T C/N+lambda I)^-1 C^T (F-C)/N
    e=F-C(I+W)
令B=I+W，则
    B=(S_cc+lambda I)^-1 (S_cf+lambda I)
而不是直接预测F、以||B||²正则的
    B_direct=(S_cc+lambda I)^-1 S_cf.
两者差 lambda (S_cc+lambda I)^-1。等价目标为
    min_B E||F-CB||² + lambda||B-I||².
**当前模型将映射拉向恒等I，而不是零。** 对低方差方向lambda相对较大时B趋近I；这隐含粗细同通道语义及尺度接近的先验。不同感受野、插值和通道池化使该先验不一定合理。Ridge训练残差在拟合样本上满足C^T E/N=lambda W（中心化后一阶条件），一般不严格正交；因此不能说已经数学消除跨尺度相关性。对照 direct_fine_ridge（B正则向0）、residual_ridge（B正则向I）；报告相邻尺度相关、W,B范数、条件数和独立正常预测R²。

## 3. 可证明的失效例子（不是实测）

### F1：跨尺度共同漂移被预测器解释

设细粗异常扰动分别delta_f、delta_c，则
    delta_e=delta_f-delta_c B.
如果delta_f=delta_c B，e完全不变。粗尺度若落入正常支持，联合/直接残差均可能漏检。例如B=I，正常bank含(c=1,e=0)，异常样本(c=1,e=0)是另一位置的正常外观，A2O无法区分；如果投影后的两尺度特征完全等同某个正常点，任何仅使用该表示的评分都无解。必须区分特征不可辨识和评分机制退化。

### F2：同一个条件预测器不等于显式条件分布

假设两类正常上下文c=a,b，条件残差分布分别e~N(0,0.01²)与N(0,1²)。测试e=0.1：在a下为10个标准差、b下仅0.1个。当前每task全局std会把两个模式混合，无法保证正确条件比较。联合kNN在上下文足够分离、bank覆盖充分时可能隐式区分，但有限采样、维度缩放和Coreset改变可使之失败。

### F3：共同近邻限制与正常变异权衡

固定归一化后的u=c、v=e；k=1时
    A_joint=min_j (||u-u_j||²+||v-v_j||²)
    >= min_j ||u-u_j||² + min_j ||v-v_j||² = A_sep.
等号仅当两分量可共享一个最优邻居（或存在交集）。例正常bank两点(0,1),(1,0)，query=(0,0)，joint=1、sep=0。若不同成分独立组合正常，joint可能误报；若搭配关系本身重要，sep可能漏检。例说明不能默认独立评分更优。对k=5，也可用“对任意同一k索引集合，两个分量的均值至少各自k最小均值”推出同类不等式。共同检索和独立评分必须实证对照。

### F4：可逆变换仅改距离几何

在投影后的[ c,f ]上，z=[c,f-cB]，对角标准化S正定。令T为块三角可逆矩阵且对角I，则
    ||delta z_std||² = delta x (T S^-2 T^T) delta x^T
这是结构化Mahalanobis度量。中心化偏置在距离中完全抵消；W影响距离和Coreset，但不产生新信息。若原拼接做精确完整协方差Mahalanobis，任意可逆线性T下距离不变；仅当投影、正则、协方差估计或度量不同才可能有收益。需保留concat_std、residual_std与full-whitening诊断。不能将经典高斯Schur分解或可逆残差本身宣称新颖。

### F5：投影的不可逆信息丢失发生在CSR之前

hr_csr_patchcore.py:111–126，每尺度先以PatchCore聚合到384，再adaptive_avg_pool1d至192。若两个细特征通道扰动+delta,-delta被同一平均bin合并，其投影为0。对任意处于P零空间的异常，后续CSR无解。HiAD fused也压缩但投影路径/每尺度维度不同。必须同投影控制和D=768敏感性，且先在layer2、浅层(可选layer1)及网格score诊断。浅层可能增纹理误报，不能预设有益。

### F6：非标准化残差并不等于高频

独立CNN视野、crop/bilinear、非线性与上下文变化破坏线性金字塔关系。若正常纹理相位轻移，fine-coarse残差可能增大；即便缺陷变化平滑，也可能出现在粗尺度。应测配准扰动及角色分离，不能称严格频率成分。

### F7：正常特征全局标准化可能放大边界噪声

若正常sigma_e=0.01，e噪声变化0.02即2标准差，而sigma=1的通道变化0.2仅0.2。std_floor控制极端值而非验证噪声可辨识性。记录floor命中比例、正常patch边界误报、normal held-out方差与校准尾部饱和。禁止用test调floor。

### F8：局部异常峰值/分数可能被后处理消除

原HiAD两次sigma4平滑；同网格理想无限域卷积等效sigma=sqrt(32)≈5.66，实际patch边界/插值/低分支max使其不严格等价。正峰被平滑不代表AP下降必然，必须同后处理保存raw-grid、patch map、merged map。512指标可能遗漏真实2048小异常。尾部校准有限分辨率，超出正常最大值全部饱和；max无法纠正任何单支高响应。局部机制首轮关闭thumbnail、tail，后续固定同一fit/cal分析。

## 4. 首轮验证的最小修订：先判别失败来源，不增加网络

保持2048、512 patch、stride512、0/1尺度、A2O、no thumbnail、no tail。原协议A0–A7保留，新增针对“直接预测差异是否有价值”的诊断，优先同一cache/相同正常样本计算而非逐模型重复抽特征。

先冻结投影及训练/正常诊断划分：fit训练Ridge及bank；正常诊断holdout只估计分数统计与预测泛化（最终原协议全训练模型可另行复跑，不能混比分数）。严禁test真值选权重或标准差。若需要给直接评分校准，与joint/concat/fused使用相同fit集；先raw机制诊断，再独立normal校准。

| ID | 方法 | 目的 |
|---|---|---|
| R0 | raw e0=f-c 直接平方norm（同正常标准化） | 不学习的跨尺度差异 |
| R1 | e=f-cB-b 直接统计分数 | HLGFA启发的预测差异证据 |
| R2 | c-only正常bank | 结构异常覆盖 |
| R3 | e-only正常bank | 残差正常集合 |
| R4 | c和e各自bank、独立评分再融合 | PyramidFlow启发的先评分后融合 |
| R5 | 当前joint bank | 共用近邻及条件搭配 |
| R6 | direct_fine_ridge vs residual_ridge | 恒等正则先验检验 |
| R7 | local context-conditioned residual distribution | 仅在异方差/多模态证据成立时继续 |

R0/R1：仅正常fit中估计mu_e、sigma²_e；直接统计分数 mean_d ((e-mu_e)/max(sigma_e,floor))²。不能把该分数称为精确条件概率，除非额外高斯和同方差假设成立。R2/R3相同候选与bank预算；R4先固定独立正常score校准再求max或sum，对照二者且不依据test选择。联合和独立bank若各自拥有B行，总容量需按字节及查询FLOPs匹配（两个d维bank vs 一个2d维bank）；分别选择Coreset时再做固定行ID控制。

R7候选：以c从唯一正常配对bank检索K近邻，再按相同索引获取e_i，估计局部均值及收缩方差：
    m(c)=sum_i alpha_i e_i
    v(c)=sum_i alpha_i (e_i-m)^2
    v_shrink=(1-beta)v(c)+beta v_global+eps
    A_cond=mean((e-m)^2/v_shrink)
alpha非负归一化，beta预先固定或仅正常验证。若声称高斯conditional NLL还需加log v_shrink与常数，且要保留coarse/context OOD分数；否则宽方差区域可能过度宽容。有效邻居不足时fallback至全局残差统计，记录覆盖及OOD。此候选未实现，不预设优于现方案。

进一步考虑原始细尺度异常证据作为安全对照，避免共同尺度异常完全被预测。不要同时增加FiLM、flow、Mamba，否则无法定位来源。

## 5. bottle/capsule针对性实验与决策

PyramidFlow附件Supplement Table S2：bottle去pyramid difference AUPRO=83.4 vs full94.0（-10.6 pp）；capsule=94.0 vs full93.1（+0.9 pp）。两者去volume norm或模板均明显下降。这说明保留尺度差异并非两类必然同效；capsule尤其不应预设多尺度优于fine。HLGFA Table3无结构prior P-AP下降19.9pp，而无detail下降1.9pp（全MVTec，不是这两类）。这些只是提出假设的依据。

| 检验 | 正常数据观察 | 冻结后test评价 | 失败时的处理 |
|---|---|---|---|
| 可预测性 | R²、holdout残差方差 | 按大小/类别分数分离 | 若R²接近0或负，暂停复杂预测器 |
| 归纳偏置 | B-I/B范数、W条件数 | R6配对P-AP/PRO | 若direct更好，否决恒等prior |
| 条件残差 | normal分布随c分组的方差变化 | R0/R1/R3/R5 | 若直接分数已足够，不保留joint复杂性为主贡献 |
| 共同比对 | 同/不同bank近邻ID和分量贡献 | R4/R5 | 依据正常组合/异常搭配错误选下一步 |
| 投影保真 | 投影重建误差/方差比例 | fine、shared-P、D=768 | 先修表示而非提高预测器容量 |
| 小缺陷 | 真实GT面积及短边、边界 | 2048 P-AP、small PRO@FPR1% | 若只512指标提高，不宣称微缺陷收益 |

每类分别报告，不混pool；seed42先跑，原协议其余种子复核；所有实验正常数据来源、特征ID、bank尺寸固定。无损导出raw map，原始面积分组沿用ROUND1；如果需要独立校准留出，所有比较统一留出。当前仓库没有实际MVTec-2K图像，不能填任何实验数值。R0–R7是设计与待实现对照，不是当前CLI可运行模式。

## 6. Go/no-go 顺序

1. 首先补齐concat_std/residual_std及同源投影控制；真实GPU smoke。
2. 检验独立正常预测误差与条件统计；同时跑R0/R1/R5，回答“残差本身是否是好的异常证据”。
3. 若R1/R3/R4提升而R5没有，优先修改评分机制，非加深网络。若R5稳定优于独立评分及普通标准化，保留joint设计并准确定位为学习度量。
4. 若所有多尺度变体不优于fine，审查输入尺度、骨干stride和压缩；若正常不可预测、异常亦被粗尺度解释，则停止当前CSR假设。若异方差或多模态明确，才考虑R7；其收益仍需独立验证。
5. 通过后才恢复SCA和thumbnail、normal calibration；实际峰值显存/时间及失效案例均需记录。

结论：两篇文献共同支持“不同分辨率信息应当参与异常判别”，并未证明“线性条件残差拼接+PatchCore”就是合适实现。当前最紧迫创新候选是拆分条件残差的直接证据、独立结构证据与联合近邻证据，通过对照决定保留何种评分，而不是预先宣布新方法一定更好。
