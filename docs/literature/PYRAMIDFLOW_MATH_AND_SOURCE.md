# PyramidFlow 数学推导、公开实现与机制审查

来源：附件 Lei 等 - 2023 - PyramidFlow High-Resolution Defect Contrastive Localization using Pyramid Normalizing Flow.pdf，arXiv:2303.02595v1（14页，含4页补充材料）；CVPR 2023 https://openaccess.thecvf.com/content/CVPR2023/html/Lei_PyramidFlow_High-Resolution_Defect_Contrastive_Localization_Using_Pyramid_Normalizing_Flow_CVPR_2023_paper.html 。

源码：原README链接 https://github.com/gasharper/PyramidFlow 经GitHub API返回404。可读公开版本 https://github.com/FourthM/PyramidFlow ，main commit c463b1cb0c2b084cdbae290234e3f8657298e981。该仓库README自称官方实现并保留原链接；未能证明与原作者最后发布完全一致。读取model.py(335行)、train.py(172行)、util.py(189行)。autoFlow.py为README要求另行下载的依赖，未核验其内部反向传播实现。静态审查，不是运行复现。以下[P]原文，[C]公开代码，[D]数学推导，[H]条件性风险。

## 1. 核心范式

[P] Fig.2：训练使用正常图像对，优化其多尺度latent差异；推理以所有正常训练样本的逐位置latent均值为模板，逐尺度先计算异常，再金字塔合成。不是常规NLL流模型，也不是PatchCore多尺度拼接；无外部先验版本使用RGB的1×1卷积扩维，可选预训练ResNet18浅层。

## 2. Invertible Pyramid (eq.1–2)

定义 x=WI，W∈R^{C×3}，通常C>=3。令 g_0=x, g_{d+1}=D_d(g_d)。各层形状 C×H_d×W_d，H_d、W_d约减半。定义
    p_d = g_d - U_d(g_{d+1})   d=0,...,L-2
    p_{L-1} = g_{L-1}
[P] 对应eq.(1)与D^L(x)=0；eq.(2)将各层逐级上采样相加。原文D/U为线性算子。为避免尺寸歧义，递推合成：
    h_{L-1}=p_{L-1}
    h_d=p_d+U_d(h_{d+1})
[D] 归纳：若h_{d+1}=g_{d+1}，h_d=g_d；最终h_0=x。重建恒等式只需要一致调用U，不依赖D可逆，甚至D本身可非线性。若U线性则可扩写为sum U^d p_d；若U非线性需保留递归，不能直接分配到各项。

[D] 对规则2倍下采样，总元素 C HW sum_{d=0}^{L-1}4^-d < 4/3 C HW（未计batch、梯度和缓存）；分解是单射且有左逆，对所有任意pyramid系数组合不一定是满射，冗余系数空间与原始图像维数不同。不能把这一分解直接当成方形Jacobian的概率密度变换。同源单特征金字塔与CSR两个独立CNN裁切视野不同，不能称等价频带。

[C] model.py:12–29 SemiInvertible_1x1Conv 正交初始化，inverse使用torch.linalg.lstsq；若W满列秩，W^+W=I，原始输入可重建。C>3时不是方形双射，0.5 logdet(W^T W)是子空间体积因子；代码在forward:320–321用no_grad冻结此encoder。预训练版inconv不可逆且输出步长4。

[C] model.py:32–84 LaplacianMaxPyramid：5×5 binomial核（1,4,6,4,1外积/256），zero-padding；先blur后2×2 maxpool down，先nearest resize后blur up；build保存差值及最粗层，compose反向相加。补充材料第11页写Gaussian+nearest down，而公开代码maxpool，真实实现差异。尽管D不线性，重建递归仍成立。U含线性非负卷积与nearest。模型给出特征空间重建，而不是在ResNet模式原图可逆。

## 3. 多尺度仿射耦合 (eq.3–5)

对可互相调制的两个分量 x0,x1：
    y0 = x0
    y1 = exp(s(x0)) ⊙ x1 + t(x0)
    x1 = (y1 - t(y0)) ⊙ exp(-s(y0))
[D] 扁平化后Jacobian为块下三角：
    J=[[I,0],[*,diag(exp(s))]]
    log|det J|=sum s.
s,t可由非线性网络产生，但条件块必须未改变或在逆向已恢复；scale非零。无需对条件网络求逆。这里是对pyramid tuple坐标中的方形耦合，而非端到端原图冗余表示的密度公式。

[C] model.py:106–130 AffineParamBlock：两层无bias Conv2d，中间LeakyReLU，末层零初始化，按通道分s_raw,t：
    s=VN(2*0.636*atan(s_raw/2))
    a=exp(s)
返回a,t和s。clamp默认2；VN减均值之后绝对值可能超过clamp原始范围。

[C] model.py:185–239 FlowBlock，direct=up时粗x1最近邻插值到细尺度x0，预测a0,t0，更新z0=A(a0⊙x0+t0)，x1不变；direct=down时从细x0生成参数，再nearest resize到粗网格，更新z1=A(a1⊙x1+t1)。注意论文正文写bilinear对齐，公开代码使用nearest；down方向对exp后的scale单独插值，不是任意插值下exp与插值交换（nearest时成立）。

[C] model.py:242–276 FlowBlock2 三尺度dual：
    g=concat(resize(x0,target=x1),resize(x2,target=x1))
    z0=x0, z1=A(a(g)⊙x1+t(g)), z2=x2
恢复x1：A^-1 z1后减t除a；条件相邻尺度保持不变。model.py:291–301循环按偶/奇层交替FlowBlock和FlowBlock2，确保多层都能被更新。补充算法1与该流程对应。

## 4. Invertible 1×1 convolution (eq.6–7)

[P] A=P L (U+diag(exp(s_i)))；P置换，L单位下三角，U严格上三角。
[D] |det A|=prod exp(s_i)=exp(sum s_i)，log|det A|=sum s_i。A的这些对角元素是三角因子的主元，不能一概称为A的特征值（原文用了eigenvalue说法）。空间H×W独立乘A，logdet为HW sum s_i。

[C] model.py:132–182 InvConv2dLU：QR初始后LU分解，保存置换、三角mask、主元符号；用sign_i exp(w_s_i)重建主元。volumeNorm时 w_s←w_s-mean(w_s)，每个像素 |det A|=1；逆用double求逆转float再conv2d。不会保证条件数为1。例 diag(100,0.01) 的体积恒定但条件数10000。必须监测逆误差，不能把volume preserving解释为距离保真。

## 5. Volume Normalization (eq.8, Sec.3.2, Supp.Alg.2)

普通flow p(x)=p(z)|det J|；若给定基础高斯，NLL含二次能量减logdet。作者不采用该常规概率密度目标，而以体积约束保留有用的隐式先验并学习正常差异。

[P] affine log-scale减均值，CVN沿channel、SVN沿space；训练更新running mean(momentum .1)，测试用running mean。
[C] model.py:87–104实现CVN dims=(0,1)，按batch与channel、每个空间位置中心化；SVN dims=(0,2,3)，按batch与空间、每个channel中心化。这与补充算法2只写dim=1有区别。

[D] 对训练batch中共享条件不变时，按被归一化的整个分组 sum s'=0，因此对应更新张量的整体体积可保留；不推出每个样本独立sum为0。推理减running mean，当前样本sum s'一般非零；不能断言任何测试输入的affine block严格单位det。down路径在resize后实际目标网格的分组权重也可能改变sum，需逐输出核查。1×1卷积log-scale中心化与affine VN不同：前者对每个权重计算严格成立，后者依赖样本/状态。训练时跨batch统计使单样本映射受同批样本影响；逆向重算是否正确依赖autoFlow的运行状态处理（未核验）。不可把CSR z-score称作VolumeNorm，前者直接缩放观测特征，后者约束仿射变换的log-scale。

## 6. 训练目标：正常样本对与Fourier loss (eq.9)

[P] 对正常样本i,j得到多层latent z_d^i,z_d^j，先逐尺度差异，再Lcom合成，最后对频域差异优化。原文eq.(9)未指定精确范数，Fig.2示意平方差异。

[C] util.py:121–134 BatchDiffLoss对batch所有i<j取元素差的绝对值p次方，p=2。train.py:44–54：
    delta_d^{ij}=|z_d^i-z_d^j|² (elementwise)
    T^{ij}=mean_channel(Lcom({delta_d^{ij}}))
    loss=mean_{pairs,k}|FFT2(T^{ij})_k|.
Adam lr2e-4,eps1e-4,weight_decay1e-5,betas(.5,.9)，clip norm1，15epochs,batch2。固定batch、drop_last=True。补充材料相同优化器参数。默认encoder resnet18；no encoder输入256，ResNet输入1024而特征256。不能当作原像素2K训练。

[D] 若FFT2采用默认非归一化，DC项为sum T>=0；各频率均进入L1均值，显式频率权重不存在。Parseval适用于平方L2而非这里L1频谱；不可宣称仅使用FFT就必然提高高频权重。对给定正常样本独立同分布，E|z_i-z_j|²=2Var(z)，故压缩正常方差有统计意义；傅里叶L1不等价于简单逐通道方差。体积约束避免所有空间任意完全收缩，但允许在正常数据低维流形及法向之间重新分配尺度（非普适异常保证）。

## 7. 正常模板与先评分后合成 (eq.10)

[P] 每尺度模板 bar_z_d=(1/N)sum_n z_d(x_n)。异常为norm(z_d-bar_z_d)，再Lcom合成。代码对应：
[C] train.py:60–68 对val_loader逐样本累加；util.py:69–75说明val实际上是同一正常train/good集合，无增强；不是独立校准。
[C] train.py:73–87 逐元素 abs差异（非先取L2通道范数），compose、mean(channel)，pixel AUROC及image=max像素。公开代码没有P-AP报告。

[D] 对U线性、非负且同一坐标：
    |sum U^d delta_d| <= sum U^d |delta_d|
逐元素三角不等式。左边在先合成latent后取绝对值时可能抵消；右边独立尺度差不可因正负相消。但正值正常纹理响应也会积累，因此无AUROC/AP单调保证。例 delta0=+1,U delta1=-1，则合成差0而分尺度差2。
这条性质与CSR拼接后平方L2的区别：CSR本身在固定邻居的各block平方距离中无符号抵消；风险来自共同近邻、条件化后异常被解释、尺度带宽和标准化分配，而不是简单将CSR也描述为带符号融合。

[D] mean模板是平方L2的最优单点（E||Z-m||²），但推理用L1差时逐维median是L1最优点，均值不必最优。模板需要对齐，多模态正常可能误报；论文Sec4.5也讨论无配准困难。原图模板经inverse及W最小二乘获取，为可视化，非评分必需。

## 8. 公开源码核验与复现风险

| 主机制 | 固定公开源文件 | 对照论文 |
|---|---|---|
| 半可逆扩维及inverse | model.py:12–29 | Sec3.1 |
| 金字塔build/compose | model.py:32–84 | eq1–2 |
| VolumeNorm、affine params | model.py:87–130 | Sec3.2/Supp.Alg2 |
| LU混合 | model.py:132–182 | eq6–7 |
| scale/dual coupling | model.py:185–276 | eq3–5/Supp.Alg1 |
| flow组网、冻结encoder | model.py:279–334 | Fig3 |
| batch差异 | util.py:121–134 | Fig2/eq9 |
| FFT训练、模板、测试 | train.py:44–95 | eq9–10 |

源码URL（固定commit）：
- https://github.com/FourthM/PyramidFlow/blob/c463b1cb0c2b084cdbae290234e3f8657298e981/model.py
- https://github.com/FourthM/PyramidFlow/blob/c463b1cb0c2b084cdbae290234e3f8657298e981/train.py
- https://github.com/FourthM/PyramidFlow/blob/c463b1cb0c2b084cdbae290234e3f8657298e981/util.py

关键风险：
1. train.py:92–95 每epoch按test pixel AUROC/PRO保存best state，存在test驱动模型选择；独立复现实验须固定epoch或normal-only选择，不能直接把best-test当公正验证。
2. util.py:156–188的compute_pro_score_fast未截断到FPR0.30或归一化部分面积，直接auc(fpr,pro)；且constant map会除零。不应和标准AUPRO@0.30混用，原论文指标不能直接归因于该函数。
3. util.py:77–98 torchvision Resize整数保持宽高比，mask resize后round，极小缺陷可能丢失。1024→256的预训练特征有空间限制。
4. model.py:317–334 encoder no_grad但flow.train()可能使ResNet BN running stats更新；应显式encoder.eval()核查。inverse用self.channel!=64判断encoder，若none且channel=64会错误跳过逆；train.py:21用全局numStack/saveMem而非形参。仅静态风险，不声称已触发。
5. README另需autoFlow，未取得依赖实现，无法确认深度独立内存及状态处理。作者Fig5在固定L=8,256²,C=24改变depth，不能推出随H,W恒定显存。正常模板按原图位置，bottle/capsule都应检查配准及背景。

## 9. 原文消融（附件Table2及Supp.Table S2）

单位百分比；每格 P-AUC/AUPRO，均为作者原数据集结果，并非MVTec-2K。

| Variant | 全15类平均 | bottle | capsule |
|---|---:|---:|---:|
| Full |95.5/94.0|95.9/94.0|96.1/93.1|
| w/o Volume Norm |86.6/84.9|76.5/77.8|82.9/81.3|
| w/o Latent Template |91.5/87.1|83.2/82.4|90.0/87.3|
| w/o Pyramid Difference |91.3/88.9|92.8/83.4|96.0/94.0|
| w/o Fourier |92.9/92.2|88.0/88.0|95.1/94.0|

对bottle，删除pyramid difference使AUPRO降10.6pp；对capsule，反而升0.9pp。作者补充材料也指出对象类平均受pyramid difference影响较纹理类小。故不能以全数据平均结论断言capsule必获益。volume与模板影响更强；CVN/SVN自动配置公开代码bottle为SVN、capsule为CVN（与所有对象同一种归一化的说法不符）。

## 10. 对CSR可移植机制

优先借鉴先逐尺度对正常参照评分、再融合，而非直接复制复杂flow或FFT。对于已对齐CSR细/粗特征，应分别建立结构正常性和innovation正常性分数，保留joint bank为并列对照。模板作为位置对应的诊断对照，不能在A2O把所有位置均值当成PyramidFlow的模板。标准化不是体积归一化；位置对齐假设、正常多模态、异常与粗尺度共享变化必须独立测试。
