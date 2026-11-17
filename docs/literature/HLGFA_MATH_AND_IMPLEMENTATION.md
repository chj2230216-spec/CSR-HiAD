# HLGFA 数学推导、实现边界与机制审查

来源：附件 ⭐HLGFA(1).pdf，Zhou et al., HLGFA: High–Low Resolution Guided Feature Alignment for Unsupervised Anomaly Detection，arXiv:2602.09524v3 (2026-05-09)，共16页。公开入口 https://arxiv.org/abs/2602.09524v3 。本文件以附件第3节式(1)–(10)、Fig.2–5、Table1–3 为准。

证据标记：[P] 原论文；[D] 根据原公式严格推导；[H] 条件性解释或风险；[R] 独立参考实现，不是作者源码。检索 arXiv v3、GitHub HLGFA anomaly、作者及题名未发现可核验的官方实现；不把复现猜测当作证据。原文未指定的层结构、权重、参数不自行补成事实。

## 1. 任务与机制定位

[P] 正常样本训练，category-agnostic；冻结共享 ImageNet Wide-ResNet50，HR/LR双视图，学习跨分辨率一致性。推理不使用记忆库或参考样本。不是整图HR和LR分别执行独立异常检测，也不是先融合特征再用PatchCore评分。Fig.2示例 HR=640×640、LR=80×80，比例8；Fig.3以HR layer2 80×80×512、layer3 40×40×1024为例。实验4.1为640输入、Adam、100 epochs、batch32、学习率1e-3至1e-4余弦退火。不能把Fig.2比例或640输入视为已验证2K/4K的结论。

## 2. 双尺度特征与数学维度 (eq.1–3)

令 xh=x, xl=D(xh)，共享冻结骨干 B，层 s 的输出 H_s=B_s(xh), L_s=B_s(xl)。定义 U_s 把 L_s 对齐到 H_s 的空间尺寸。

[P] eq.(1)：Lbar_s=U_s(L_s)。
[P] eq.(2)：Lhat_s=G_s(Lbar_s,H_s)，LR作为承载主体，HR提供引导。
[P] eq.(3)：A(u)=sum_s [1-cos(Lhat_s(u),H_s(u))]。

其中 u 为特征位置。不同层的网格不同，实际合成时需要映射到共同坐标；原论文未给出完整的多层插值/权重/图像级pool细节，因此不能指定某个实现为原版。HR/LR同图像而非跨图像参考，输入几何对应不代表感受野完全一致。

[D] 对非零向量 a,b，令 ua=a/||a||, ub=b/||b||，则
    1-cos(a,b)=0.5||ua-ub||² ∈ [0,2]
若 b=ka, k>0，分数为0，说明对纯幅值差异不敏感（若k<0则为2）。实际零范数需eps或明确掩码；不能把无定义cos当作0。
[D] 对小扰动 b=a+delta，二阶展开
    1-cos(a,a+delta) ≈ ||P_perp(a) delta||²/(2||a||²)
P_perp=I-uu^T。平行分量的一阶信息消失；小异常若只改变特征幅值可能漏检。

## 3. 结构/细节引导 (eq.4–6, Fig.2–3)

[P] 深层HR特征经多尺度depthwise 3×3 dilation 1,3,5提取结构先验；浅层HR特征经downsample和1×1 channel projection得到细节先验，统一空间及通道后:
    g_s = S_s + D_s       (4)
在Fig.3的层级示例中，浅层 H2 为512×80×80，深层 H3 为1024×40×40；若d与s直接相加，两者必须输出相同C,H,W。具体融合卷积/激活和不同s的分工未完整公开。

[D] 单3×3 dilation d有效核边长 2d+1，d=1,3,5 对应3/7/11个特征格点，不等于输入像素感受野，也不能证明结构/细节严格正交、频域分离。g=S+D仍是信息合并，并非可逆分解。

[P] eq.(5) FiLM(Lbar | g)，eq.(6) Lhat=Lbar+G(g,Lmod)。Fig.2示意 gamma,beta 调制及 gate theta 与 delta 分支，画出 fmod=(1+gamma)×f3l+beta、fgl=f3l+theta×beta。注意图中 beta 符号重复，原文广义G没有给出足够实现细节判断最终beta是否复用/重参数化。以下统一用 Delta 指代待注入修正，非断言作者的精确实现：
    Lmod=(1+gamma(g))*Lbar+beta(g)
    Lhat=Lbar+theta(g,Lmod)*Delta(g,Lmod)
gamma,beta,theta,Delta都需与输出广播兼容。Fig.2描述分工 where/how，不足以确认gate sigmoid及具体输入拼接，参考实现可选择sigmoid但需标记假设。

[D] Lhat关于H的灵敏度一般不为零，因为g=G_H(H2,H3)。若残差修正能够学到 H-Lbar 且theta=1，则 Lhat=H，对任意输入分数0。此为容量足够时的退化反例，不是作者模型已发生此退化的实测。冻结B只固定特征，不限制修正网络复制目标。可检验：遮蔽HR-guidance、交换相邻图像guide、观察修正前后 normal/anomaly discrepancy；不得用测试标签拟合gate。

## 4. 结构可靠性评分 (eq.7–8)

[P] 原文虽然称 structure-based reliability，但eq.(7)使用归一化的联合guide g=S+D，而非仅S：
    sim(u)=mean_{v∈N(u)} dot(g(u)/||g(u)||, g(v)/||g(v)||)
    rho(u)=sigmoid((sim(u)-tau)/delta)
    A'(u)=rho(u) A(u)
delta应为正，tau、delta、neighborhood以及边界处理未充分披露，不能捏造具体值。

[D] 若delta>0，rho单调增，且0<rho<1（有限输入），因此A'<=A。导数 d rho/d sim=rho(1-rho)/delta。异常与正常两个像素a,n的排序可能反转：
    A_a>A_n 但 rho_a/rho_n < A_n/A_a  => A'_a<A'_n.
例 A_a=0.8,A_n=0.2,rho_a=0.1,rho_n=0.9，得到0.08<0.18。微小划痕或边缘处结构一致性低，权重可能压制真实异常；不能把空间平滑性当成可靠性已证明。应分析A、rho、rho*A三图及按缺陷尺寸/边界误报的变化。

## 5. Noise-aware augmentation (3.3)

[P] 正常图像中注入点状噪声和条纹结构噪声，HR/LR一致地应用，拟制发丝、污渍、细裂纹状和污染干扰。作者将它们作为 nuisance 而非缺陷监督。不能声称完全无合成扰动。
[H] 在具体工业标准中，点/条纹/裂纹可能是真缺陷。若无物理定义直接作为正常增强，会削弱相同形态异常检测。当前CSR首轮 normal-only 不自行合成异常，暂不移植该机制。

## 6. 训练目标 (eq.9–10)

[P] L_align=E[1-cos(H,Lhat)]，总损失 L_align+lambda_L1 L_L1+lambda_JS L_JS+lambda_Gram L_Gram+lambda_cls L_cls。L1为focal-weighted一致性；JS用通道归一化概率；Gram约束二阶结构；classification为可选辅助。论文未给出所有精确定义和系数。

[R] 可实现的一种标准数学形式（仅作为复现候选，非原文公式）：按位置令 p=softmax(H/T), q=softmax(Lhat/T), m=(p+q)/2，则 JS=0.5 KL(p||m)+0.5 KL(q||m)；通道Gram可取 G(F)=FF^T/(HW)，损失||G(H)-G(Lhat)||_F²；focal-L1可取 E[w(u)||H-Lhat||_1]，需明确权重是否detach、gamma和归一化。不能宣称这些选项就是官方实现。Gram对空间位置置换不敏感，不替代局部对齐；cos忽略幅值，L1补充该信号，但增大一致性约束也可能压缩异常差异。

## 7. 参考代码：明确非官方、只体现公式

下面是对eq.(3),(5),(6),(7),(8),(9)的一种PyTorch参考实现片段。调用者实现结构/细节先验，gamma/beta/gate/delta head及阶段对齐；不声称是完整HLGFA训练网络。邻域使用有效邻居、排除中心，属于参考选择（论文未指定）。输入均BCHW，浮点且非零有效特征；g为结构+细节。

~~~python
import torch
import torch.nn.functional as F

def alignment(lr, hr, g, film, correction):
    low = F.interpolate(lr, size=hr.shape[-2:],
                        mode="bilinear", align_corners=False)
    gamma, beta = film(g).chunk(2, dim=1)
    mod = (1 + gamma) * low + beta
    gate_logits, delta = correction(torch.cat((g, mod), 1)).chunk(2, 1)
    pred = low + gate_logits.sigmoid() * delta
    a = 1 - F.cosine_similarity(hr, pred, dim=1, eps=1e-8)
    return pred, a.clamp_min(0)

def reliability(g, radius, tau, softness):
    assert radius >= 1 and softness > 0
    g = F.normalize(g, dim=1, eps=1e-8)
    k = 2 * radius + 1
    # Sum only in-image neighbors, then remove center contribution.
    total = F.avg_pool2d(g, k, stride=1, padding=radius,
                         count_include_pad=True) * k**2
    n = F.avg_pool2d(torch.ones_like(g[:, :1]), k, stride=1,
                     padding=radius, count_include_pad=True) * k**2 - 1
    sim = ((total - g) * g).sum(1) / n[:, 0].clamp_min(1)
    return ((sim - tau) / softness).sigmoid()

# training: HR feature encoder frozen; optimize learnable heads on normals
# loss = a.mean() + explicitly configured auxiliary losses
# inference: A = sum(stage_maps resized to common grid)
# A_prime = reliability(g, radius, tau, softness) * A
~~~

独立实现尚未运行，未提供缺失的作者配置或伪造完整准确率。应先测试形状、零向量、恒等输入、幅值扰动和边界。参考代码不应直接替代CSR训练。

## 8. 原文实验回读

附件Table1 MVTec AD HLGFA I-AUC97.4、I-AP99.1、P-AUC98.0、P-AP65.3、PRO90.2。VisA分别88.6、90.0、97.9、40.7、83.8；原文后续文字另写88.2/97.8/40.6，摘要另写MVTec 97.5/97.9，存在内部不一致。本报告以明确表格数字为准，不混用；不外推至MVTec-2K。

Table2：
| Loss | I-AUC | P-AP | PRO |
|---|---:|---:|---:|
| Cos |94.2|61.1|88.6|
| Cos+JS |92.7|58.4|87.9|
| Cos+Gram |94.8|62.5|89.0|
| Cos+L1 |97.2|65.0|89.9|
| Full |97.4|65.3|90.2|

Table3：
| Variant | I-AUC | P-AP | PRO |
|---|---:|---:|---:|
| w/o Structure |86.6|45.4|67.7|
| w/o Detail |95.5|63.4|89.6|
| w/o Noise Aug |96.7|57.0|88.2|
| Full |97.4|65.3|90.2|

结构先验P-AP差19.9 pp，detail差1.9，noise差8.3。Cos+JS反而低于Cos，说明不是所有一致性约束都有效；完整组合不能把效益再分配给单独gate或可靠性权重，因为未见独立消融。原实验支持跨尺度对齐有条件有效，不支持简化Ridge+PatchCore必定提升。

## 9. 对CSR的具体启示

HLGFA最值得借鉴的是直接对预测/对齐误差评分，而非将误差再次当成普通向量。建议作为CSR独立的 residual-score control；比较同一预测器下 e 的直接标准化异常评分、联合近邻与二者融合。不能照搬HR目标引导、结构一致性乘权和干扰增强。另需使用浅层特征与位置匹配的消融检查下采样丢失。无作者代码时仅可标记 HLGFA-inspired，不能称严格复现或公平SOTA基线。
