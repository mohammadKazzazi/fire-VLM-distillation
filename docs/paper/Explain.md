Distilling Vision-Language Models for On-Device Fire Understanding

Mohammad Kazzazi 2∗, Zixuan Liu1∗, Siavash Khajavi1†

1Detectium, Finland  
2EPFL, Lausanne, Switzerland  
{Zixuan, }@detectium.io,

Abstract

Vision-language models (VLMs) offer a promising alternative  
to conventional fire detection systems by reasoning about the  
semantic context of a scene and thus reducing false alarms,  
yet their large model size makes deployment on embedded  
fire sensors impractical. In this paper, we study how domain-  
specialized VLMs can be compressed for fully on-device de-  
ployment without losing the safety-critical behavior required  
for fire detection. We develop a teacher-student knowledge  
distillation framework in which large VLMs fine-tuned for  
fire understanding can be distilled into lightweight students.  
Experiments across multiple VLM families and model scales  
show that compact students preserve most of their teachers’  
fire-understanding capability. We further deploy the distilled  
models on our commercial Detectium fire detection sensor  
and jointly evaluate reasoning accuracy, latency, and mem-  
ory usage. The results show that compression and deploy-  
ment affect not only accuracy but also model failure modes,  
with Qwen2.5-0.5B providing the strongest overall deploy-  
ment trade-off. Our findings provide broader guidance for de-  
ploying domain-specialized VLMs in resource-constrained,  
safety-critical settings.

1  
Introduction

Traditional fire detection systems in industrial settings rely  
primarily on smoke detectors or thermal cameras that trigger  
alarms upon detecting smoke or abnormally high tempera-  
tures (Fonollosa, Solórzano, and Marco 2018; Khajavi et al.  
2023). As these sensors respond to low-level physical signals  
rather than the surrounding context, they suffer from high  
false-positive rates (Liu, Yuan, and Huang 2023): benign  
scenarios, such as open flames during cooking or deliber-  
ately controlled high-temperature processes in a factory, can  
raise the same alarms as a genuinely hazardous fire (Mensch,  
Veley, and Cleary 2024). Every false alarm requires human  
verification and may trigger costly interventions, wasting re-  
sources and eroding trust in the system (Tannous 2021; Chag-  
ger and Smith 2014). Recent advances in multi-modal mod-  
els, particularly vision-language models (VLMs) (Liu et al.

∗These authors contributed equally.  
†Corresponding author.  
Copyright © 2027, Association for the Advancement of Artificial  
Intelligence (www.aaai.org). All rights reserved.

2023; Bai et al. 2025; Team et al. 2024; Liu, Jiang, and Kha-  
javi 2025), offer a path beyond signal-level detection. Given  
image frames from a camera, a VLM can reason about the  
scene as a whole, for example, distinguishing a controlled  
flame from an uncontrolled one, and assess whether an ob-  
served fire poses a genuine safety risk. Prior studies show that  
such semantic reasoning substantially reduces false positives  
compared to traditional systems (Gragnaniello et al. 2025;  
Kim et al. 2025), pointing the way toward a more intelligent  
generation of fire detection systems.  
Deploying VLMs for fire detection, however, is hindered  
by a fundamental conflict between model scale and deploy-  
ment constraints. Modern VLMs with billions of parameters  
cannot run on the embedded hardware of fire detection sen-  
sors and must instead be hosted on cloud services, requiring  
every captured frame to be uploaded for analysis. This in-  
troduces two problems. The first is latency. Fire propagates  
rapidly, and a practical system must identify a hazardous  
fire at the very onset of ignition for mitigation to be effec-  
tive (Çetin et al. 2013). Conventional smoke detectors take  
on the order of a minute to trigger an alarm (Bukowski et al.  
2003). To alert before a smoke detector, a VLM-based system  
must therefore complete its entire detection-inference cycle  
within seconds. Cloud inference, however, adds round-trip  
communication delay, and any network failure can render  
the system unavailable at exactly the moment it is needed.  
The second, and more fundamental, is privacy. Fire sensors  
must be installed precisely in the spaces with the strongest  
privacy expectations, such as private apartments, public re-  
strooms, or facilities where transmitting images off-site is  
prohibited by regulation (Regulation 2016; Kansal, Wong,  
and Kankanhalli 2025), making continuous image upload to  
a cloud service unacceptable. Fire detection thus demands the  
semantic reasoning of large modern VLMs, yet requires fully  
on-device inference to guarantee low latency and privacy, re-  
quirements that current systems cannot satisfy at once.  
In this paper, we investigate how a domain-specialized  
VLM can be compressed sufficiently for real-world embed-  
ded deployment without sacrificing the safety-critical behav-  
ior, i.e., reduced false-alarm rate, required for fire detection.  
To this end, we develop a teacher-student knowledge distil-  
lation framework for on-device fire understanding. We first  
fine-tune state-of-the-art VLMs from the Qwen (Yang et al.  
2025), Llama (Grattafiori et al. 2024), and Gemma (Team


---


et al. 2024) families as teacher models to acquire contex-  
tual understanding of fire scenes, and then distill their fire-  
domain knowledge into lightweight student models spanning  
multiple parameter scales. We evaluate the resulting models  
on DetectiumFire (Liu, H Khajavi, and Jiang 2026), mea-  
suring not only their ability for fire understanding, but also  
their behavior on non-fire scenes that determine false-alarm  
performance. Our results show that compact students can  
retain most of their teachers’ capability, while increasing  
student size provides surprisingly limited accuracy gains.  
We then deploy the distilled students on our commercial  
Detectium fire detection sensor and jointly evaluate their  
fire-understanding accuracy, safety-relevant detection errors,  
inference latency, and memory consumption under fully on-  
device inference. This deployment study reveals that model  
deployment involves more than an accuracy-efficiency trade-  
off: different architectural and deployment choices can sys-  
tematically shift the model toward different failure modes.  
Across the tested model families and scales, Qwen2.5-0.5B  
provides the strongest overall deployment trade-off.  
In summary, our contributions are as follows: 1. We  
develop a teacher-student distillation framework that com-  
presses domain-specialized VLMs into lightweight models  
suitable for fully on-device fire understanding while largely  
preserving their semantic reasoning capability. 2. We system-  
atically study the effect of distillation across model families  
and scales, showing that compact students retain most teacher  
performance while revealing important failure modes under  
different distillation objective. 3. We deploy the distilled stu-  
dents on our commercial Detectium fire detection sensor and  
characterize the practical trade-off among reasoning accu-  
racy, safety-relevant detection errors, latency, and memory  
usage. Although our experiments focus on fire detection,  
the resulting insights on domain-specific distillation, failure-  
mode evaluation, and deployment-time study are broadly ap-  
plicable to other safety-critical VLM applications requiring  
resource-constrained on-device inference.

2  
Related Work

2.1  
VLMs for Fire Detection.

Fire detection has traditionally been formulated as image  
classification or object detection (Çetin et al. 2013; Liu 2023;  
Elhanashi et al. 2025; Khajavi et al. 2024), which localizes  
flames but cannot reason about whether a fire is hazardous.  
Recent work instead leverages the reasoning and language  
capabilities of VLMs, moving beyond detection toward se-  
mantic understanding: describing and analyzing fire scenes in  
natural language (Seidel et al. 2025), and assessing whether  
an observed fire poses genuine risk (Gragnaniello et al. 2025;  
Kim et al. 2025). Benchmarks such as SmokeBench (Qi,  
Li, and Barnes 2026) and the multi-modal DetectiumFire  
dataset (Liu, H Khajavi, and Jiang 2026) further show that  
VLMs can describe burning objects, environments, and risk  
levels beyond binary detection. However, existing work eval-  
uates VLMs offline or assumes cloud-scale inference. De-  
ploying fire-domain VLMs on embedded sensors remains  
unexplored, which is the focus of this work.

2.2  
Knowledge Distillation for VLMs.

Knowledge distillation (KD) transfers the capability of a  
large teacher into a smaller student (Hinton, Vinyals, and  
Dean 2015; Kim and Rush 2016), and has become a stan-  
dard route to deployable vision-language models (Jin et al.  
2025; Shu et al. 2025; Cai et al. 2025; Ghonim, Roberto, and  
Bernardi 2026; Kumar et al. 2026). Several distillation frame-  
works have been proposed for VLMs. For example, LLaVA-  
MoD (Shu et al. 2025) progressively distills a large teacher  
into a sparse mixture-of-experts student through mimic and  
preference distillation. LLaVA-KD (Cai et al. 2025) com-  
bines output-distribution distillation with relation distilla-  
tion over visual tokens in a multi-stage training scheme.  
Production models such as Gemma 3 (Team et al. 2024)  
are likewise trained with distillation. Existing VLM distilla-  
tion, however, targets general-purpose capability measured  
on standard multimodal benchmarks. In contrast, we study  
domain-specific distillation for fire understanding from a de-  
ployment perspective, evaluating not only the preservation  
of visual reasoning capability but also its impact on safety-  
critical behaviors, on-device latency, and resource consump-  
tion.

3  
Knowledge Distillation for Fire  
Understanding

In this section, we present our teacher-student knowledge  
distillation framework for on-device fire understanding. We  
first describe how state-of-the-art large VLMs are fine-tuned  
to acquire contextual understanding of fire scenes in Sec-  
tion 3.1. We then introduce the procedure for distilling this  
fire-domain knowledge into lightweight student models in  
Section 3.2, following the standard distillation framework of  
prior work (Shu et al. 2025).

3.1  
Teacher Model

Since general-purpose VLMs are not specifically trained to  
reason about fire scenes, and no existing fire-domain VLM  
is suitable for use as a distillation teacher, we construct our  
own fire-specialized teacher model πT through a two-stage  
process:

Foundational Initialization:  
Given a pre-trained large  
VLM, we keep both the language model and vision en-  
coder frozen, as their pre-trained parameters already encode  
rich linguistic and visual representations. We train only the  
vision-language adaptor, which projects visual features into  
the language embedding space. For this initialization stage,  
we use a curated image-caption dataset covering diverse  
scenes and visual concepts, so that the adaptor learns general-  
purpose visual-language alignment before any fire-specific  
supervision is introduced. Specifically, given a multimodal  
instruction example (x, y), where x = (xv, xi) consists of  
the input image xv and text instruction xi, and y denotes the  
ground-truth response, we optimize the standard next-token  
prediction objective:

LInit(πT ) = −E(yk|y&lt;k,x)∼πT [log πT (yk | y&lt;k, x)] . (1)


---


Here, πT (yk | y&lt;k, x) denotes the probability assigned by  
the teacher model to token yk, conditioned on the multi-  
modal input x and the preceding response tokens y&lt;k =  
(y1, y2, . . . , yk−1).

Task-Specific Fine-Tuning.  
The model is then fine-tuned  
on the fire-specific image-caption dataset to acquire con-  
textual understanding of fire scenes. To preserve its general  
visual grounding during domain specialization, we addition-  
ally include general vision-language instruction data, which  
helps prevent the teacher from losing the ability to interpret  
and describe unfamiliar scenes. We optimize the model using  
the same next-token prediction objective as in Eq. 1.

3.2  
Student Model  
Foundational Initialization:  
Before distillation, we ini-  
tialize the student model using the same procedure as the  
teacher. Specifically, we train only the vision-language adap-  
tor on general image-caption pairs to establish general visual-  
language alignment before introducing fire-specific distilla-  
tion.

Distillation.  
The initialized student model πS is then  
trained to reproduce the teacher model’s behavior on fire-  
understanding tasks. Given a fire-understanding example  
(x, y) presented to both the teacher πT and the student πS,  
we train the student to match the teacher’s output distribu-  
tion at each generated token. Following prior work (Shu et al.  
2025), we measure the discrepancy between the teacher and  
student token distributions using the Kullback–Leibler (KL)  
divergence:

LKL(πS; πT ) = −E(x,yk)∼πT

[︃  
log πT (yk | y&lt;k, x)

πS(yk | y&lt;k, x)

]︃

(2)

Here, πT (yk | y&lt;k, x) and πS(yk | y&lt;k, x) denote the prob-  
abilities assigned by the teacher and student, respectively,  
to token yk, conditioned on the multimodal input x and the  
preceding tokens y&lt;k.  
However, we find that optimizing the KL objective alone  
can cause the distilled student to predict fire for nearly ev-  
ery image (Table 8), even though the corresponding teacher  
does not exhibit this bias. To mitigate this failure mode, we  
additionally incorporate a standard cross-entropy (CE) term  
based on the ground-truth responses, providing direct super-  
vision from the training data:

LCE(πS) = −E(x,yk)∼πS [log πS(yt | x, y&lt;t)] ,  
(3)

The student is therefore trained with the combined objective

L = LKL + LCE.  
(4)

4  
Distillation Evaluation  
4.1  
Experimental Setup  
Models.  
We evaluate our distillation framework across  
multiple VLM families and model scales. For the teacher  
models, we use Qwen2.5-7B-Instruct, Qwen3-8B (Yang et al.  
2025), Llama-3.1-8B-Instruct (Grattafiori et al. 2024), and  
Gemma-2-9B-IT (Team et al. 2024). Each student is distilled  
from a teacher within the same model family. Specifically, we

consider Qwen2.5-0.5B, 1.5B, 3B, Qwen3-0.6B, 4B, Llama-  
3.2-1B, 3B, and Gemma-2-2B as student models. We use the  
CLIP vision encoder (Radford et al. 2021) for most configu-  
rations and additionally evaluate MobileCLIP-S2 (Vasu et al.  
2024) as a more lightweight alternative.

Training Data.  
During the foundational initialization  
stage, both teacher and student models are trained on the  
LLaVA pretraining data (LCS-558K) (Liu et al. 2023, 2024),  
a 558K image-caption subset used for visual-language fea-  
ture alignment. The teacher models are subsequently fine-  
tuned on the DetectiumFire training set (Liu, H Khajavi, and  
Jiang 2026) to acquire contextual fire-understanding capa-  
bility, together with LLaVA-1.5 Mix665K instruction-tuning  
data (Liu et al. 2024) to preserve general visual-language  
capability during domain specialization. The student models  
are then distilled using only the DetectiumFire training set.

Evaluation.  
We evaluate all teacher and student models  
on the DetectiumFire evaluation set, which contains 1,452  
images, including 754 fire images and 698 non-fire images.  
We consider four fire-understanding tasks: fire/non-fire de-  
tection, identification of the burning object, recognition of  
the surrounding environment, and assessment of fire severity.  
The latter three tasks are evaluated on fire images only, while  
fire/non-fire detection is evaluated on the complete evalua-  
tion set. All results for student models are averaged over three  
random seeds.  
Additional implementation details are provided in Ap-  
pendix B.

4.2  
Results on Fire Understanding

Table 1 reports accuracy on the four fire-understanding  
tasks for each teacher and its distilled students. Overall,  
the teacher models achieve consistently strong performance  
across model families, despite differences in backbone and  
parameter scale. In particular, their near-perfect fire/non-fire  
detection accuracy shows that fine-tuned VLMs can effec-  
tively distinguish hazardous fire scenes from benign ones,  
while the more fine-grained fire-scene reasoning tasks re-  
main more challenging.  
Students retain most of the teacher’s capability across ev-  
ery family. For example, relative to its teacher, Qwen2.5-  
0.5B retains 95.2% of the burning-object accuracy, 99.3%  
of the environment and severity accuracy, and 99.96% of the  
fire/non-fire detection accuracy, while using roughly an or-  
der of magnitude fewer language-model parameters (0.5B vs.  
7B). Similar patterns hold across the other model families,  
demonstrating that the distillation framework preserves most  
of the fire-understanding capability of the teacher models.  
However, the distillation gap is not uniform across tasks,  
with the largest degradation occurring in burning-object  
identification. Across families, students typically trail their  
teachers by approximately 2.5-4.3 on this task, whereas the  
gap in fire/non-fire detection is negligible. Environment and  
severity prediction fall between these two extremes, with  
several students matching or even exceeding their teachers.  
This suggests that fine-grained visual discrimination, such as  
identifying the specific object that is burning, is more sen-


---


Fire-scene reasoning  
Detection

Model  
Vision encoder  
Burning object  
Environment  
Severity  
Fire/non-fire

Qwen2.5 family  
Qwen2.5-7B (Teacher)  
CLIP ViT-L/336  
80.77  
86.34  
88.86  
99.79  
Qwen2.5-7B (Teacher)  
MobileCLIP-S2  
81.43  
78.51  
83.95  
99.72  
Qwen2.5-0.5B  
CLIP ViT-L/336  
76.92 ± 1.16  
85.76 ± 1.48  
88.28 ± 1.46  
99.75 ± 0.04  
Qwen2.5-1.5B  
CLIP ViT-L/336  
77.19 ± 1.63  
85.41 ± 1.61  
87.44 ± 1.30  
99.56 ± 0.17  
Qwen2.5-3B  
CLIP ViT-L/336  
78.29 ± 1.58  
85.59 ± 0.41  
88.82 ± 1.03  
98.51 ± 0.46  
Qwen2.5-3B  
MobileCLIP-S2  
68.79 ± 1.33  
68.61 ± 1.58  
73.43 ± 1.22  
96.97 ± 0.52

Qwen3 family  
Qwen3-8B (Teacher)  
CLIP ViT-L/336  
80.37  
84.62  
88.99  
99.86  
Qwen3-8B (Teacher)  
MobileCLIP-S2  
70.69  
84.48  
85.54  
99.72  
Qwen3-0.6B  
CLIP ViT-L/336  
79.66 ± 0.17  
80.69 ± 0.14  
92.56 ± 0.21  
99.68 ± 0.04  
Qwen3-0.6B  
MobileCLIP-S2  
68.57 ± 0.48  
86.74 ± 1.22  
80.46 ± 0.33  
98.74 ± 0.14  
Qwen3-4B  
CLIP ViT-L/336  
80.37 ± 1.48  
86.43 ± 0.68  
89.08 ± 0.50  
99.61 ± 0.28

Llama-3 family  
Llama-3.1-8B (Teacher)  
CLIP ViT-L/336  
82.49  
86.21  
87.27  
99.86  
Llama-3.2-1B  
CLIP ViT-L/336  
78.16 ± 0.15  
85.54 ± 0.13  
87.40 ± 0.58  
99.77 ± 0.04  
Llama-3.2-3B  
CLIP ViT-L/336  
79.09 ± 0.63  
84.97 ± 0.80  
87.22 ± 1.83  
99.77 ± 0.04

Gemma-2 family  
Gemma-2-9B (Teacher)  
CLIP ViT-L/336  
82.23  
87.00  
88.59  
99.93  
Gemma-2-2B  
CLIP ViT-L/336  
79.22 ± 1.31  
84.84 ± 0.15  
88.55 ± 1.25  
99.56 ± 0.24

Table 1: Fire-understanding accuracy (%) across model families on the DetectiumFire evaluation set. The best-performing  
student within each family is shown in bold.

sitive to distillation than coarse fire detection or scene-level  
reasoning.  
Within the student models, increasing model capacity  
provides little additional accuracy. For example, scaling  
Qwen2.5 from 0.5B to 3B improves burning-object accu-  
racy by only 1.37, while the corresponding gains for the  
Llama-3 and Qwen3 families are 0.93 and 0.71, respectively,  
all within or close to one standard deviation across seeds.  
For fire/non-fire detection, larger students provide no bene-  
fit: Qwen2.5 accuracy decreases monotonically from 99.75%  
at 0.5B to 99.56% at 1.5B and 98.51% at 3B, and no model  
family shows a consistent improvement with increasing stu-  
dent size. Since latency and memory usage grow substan-  
tially with parameter count (Section 5, Table 4), the smallest  
student is not merely a deployment compromise, but is often  
the preferred choice in terms of both accuracy and efficiency.  
Finally, the choice of vision encoder has a larger effect on  
accuracy than increasing the language-model size. Replac-  
ing CLIP ViT-L/336 with MobileCLIP-S2 causes substan-  
tially larger performance drops than those observed when  
changing student scale. The effect is already visible in the  
teachers: Qwen2.5-7B loses 7.83 in environment accuracy  
and 4.91 in severity, while Qwen3-8B loses 9.68 in burning-  
object accuracy. This degradation is subsequently inherited  
by the distilled students. Therefore, although a lightweight  
vision encoder can reduce computational cost (Section 5, Ta-  
ble 4), the efficiency gains discussed in Section 5 come at a  
substantial loss in fire-understanding accuracy.

4.3  
Ablation Study  
Table 2 isolates the effect of the distillation objective by com-  
paring students trained with the full objective in Eq. 4 against

those trained with the KL term in Eq. 2 alone. Removing the  
cross-entropy term LCE causes a severe collapse in fire/non-  
fire detection. In particular, Qwen2.5-0.5B, Qwen2.5-1.5B,  
Llama-3.2-1B, and Llama-3.2-3B all achieve 100.00±0.00%  
recall but only 3.82%-21.25% specificity, indicating that they  
classify nearly every image as containing fire. In contrast,  
adding LCE restores balanced detection performance across  
all students, with specificity between 97.52% and 99.90%  
and recall between 99.43% and 99.82%.  
This collapse is not mitigated by increasing model capac-  
ity. Qwen3-4B, the largest student considered in this abla-  
tion, still achieves only 60.60% specificity without the cross-  
entropy term, while Llama-3.2-3B performs even worse than  
the smaller Llama-3.2-1B (7.88% vs. 13.51% specificity).  
We further report teacher-model results in Appendix C, Ta-  
ble 9, where every teacher produces at most two false alarms.  
Thus, the collapse is not inherited from the teacher, but in-  
stead emerges during distillation.  
The complete ablation results on the fire-understanding  
tasks are reported in Appendix C, Table 8. Strikingly, this fail-  
ure is largely invisible on the fire-scene reasoning tasks: re-  
moving LCE leaves burning-object, environment, and sever-  
ity accuracy largely unchanged, and in several cases even  
improves them. Because these tasks are evaluated only on  
images containing fire, a model that predicts fire indiscrim-  
inately is not penalized. A practitioner selecting the distilla-  
tion objective based only on positive-class reasoning perfor-  
mance could therefore favor the KL-only objective despite its  
severe deployment failure. This result highlights a broader  
lesson: domain-specific distillation must be validated on  
negative examples, not only on the positive class that de-  
fines the task.


---


Errors (counts)  
Rates (%)

Model  
Objective  
False alarms  
Missed fires  
Recall  
Specificity  
Accuracy

Qwen2.5 family

Qwen2.5-0.5B  
LKL only  
671.3 ± 10.4  
0.0 ± 0.0  
100.00 ± 0.00  
3.82 ± 1.49  
53.76 ± 0.72  
+ LCE  
2.0 ± 0.0  
1.7 ± 0.6  
99.78 ± 0.08  
99.71 ± 0.00  
99.75 ± 0.04

Qwen2.5-1.5B  
LKL only  
549.7 ± 28.0  
0.0 ± 0.0  
100.00 ± 0.00  
21.25 ± 4.01  
62.14 ± 1.93  
+ LCE  
4.7 ± 2.1  
1.7 ± 0.6  
99.78 ± 0.08  
99.33 ± 0.30  
99.56 ± 0.17

Qwen2.5-3B  
LKL only  
4.0 ± 2.6  
20.0 ± 5.3  
97.35 ± 0.70  
99.43 ± 0.38  
98.35 ± 0.18  
+ LCE  
17.3 ± 8.4  
4.3 ± 2.5  
99.43 ± 0.33  
97.52 ± 1.20  
98.51 ± 0.46

Qwen3 family

Qwen3-4B  
LKL only  
275.0 ± 28.2  
0.3 ± 0.6  
99.96 ± 0.08  
60.60 ± 4.03  
81.04 ± 1.97  
+ LCE  
4.3 ± 3.5  
1.3 ± 0.6  
99.82 ± 0.08  
99.38 ± 0.50  
99.61 ± 0.28

Llama-3 family

Llama-3.2-1B  
LKL only  
603.7 ± 8.4  
0.0 ± 0.0  
100.00 ± 0.00  
13.51 ± 1.20  
58.43 ± 0.58  
+ LCE  
0.7 ± 0.6  
2.7 ± 0.6  
99.65 ± 0.08  
99.90 ± 0.08  
99.77 ± 0.04

Llama-3.2-3B  
LKL only  
643.0 ± 24.0  
0.0 ± 0.0  
100.00 ± 0.00  
7.88 ± 3.44  
55.72 ± 1.65  
+ LCE  
1.3 ± 0.6  
2.0 ± 0.0  
99.73 ± 0.00  
99.81 ± 0.08  
99.77 ± 0.04

Gemma-2 family

Gemma-2-2B  
LKL only  
226.0 ± 10.2  
8.3 ± 6.4  
98.89 ± 0.85  
67.62 ± 1.45  
83.86 ± 1.14  
+ LCE  
5.0 ± 3.0  
1.3 ± 0.6  
99.82 ± 0.08  
99.28 ± 0.43  
99.56 ± 0.24

Table 2: Comparison of safety-relevant detection errors under the two distillation objectives in Eq. 4 and Eq. 2. False alarms  
denote non-fire images incorrectly classified as fire, while missed fires denote fire images incorrectly classified as non-fire. Recall  
is the proportion of fire images correctly identified, and specificity is the proportion of non-fire images correctly identified.

5  
On-device Deployment  
5.1  
Deployment Setup  
To evaluate the real-world performance of the distilled stu-  
dents, we deploy them on a commercial Detectium fire de-  
tection sensor used for industrial fire monitoring 1 (Khajavi  
et al. 2023). The target sensor has limited memory and con-  
tains no GPU or dedicated hardware accelerator. Moreover,  
it operates without network access during inference, so all  
experiments are performed entirely on the sensor CPU. This  
setting reflects the resource and privacy constraints of real-  
world deployment discussed in Section 1.  
For embedded inference, student models using the  
CLIP vision encoder are exported with the llama.cpp  
toolchain 2. The language-model weights are converted to  
GGUF format and quantized to Q8_0, while the vision en-  
coder and vision-language adaptor are packaged separately  
as the multimodal component. Because llama.cpp does  
not support conversion of the MobileCLIP-S2 encoder, stu-  
dents using MobileCLIP-S2 are instead executed directly on  
the device without this export pipeline.  
We evaluate deployment efficiency using three metrics.  
Time to first token (TTFT) measures the latency from re-  
ceiving an input request to generating the first output token.  
End-to-end latency (E2E) measures the total inference time,  
including both prefill and response generation. Peak resi-  
dent set size (RSS) measures the maximum physical memory  
occupied by the inference process. In addition to these effi-  
ciency metrics, we re-evaluate the deployed models on the  
same DetectiumFire evaluation set to measure how deploy-

1https://www.detectium.io/sensor-package  
2https://github.com/ggml-org/llama.cpp

ment affects their fire-understanding capability and detection  
behavior. Additional hardware, export, and measurement de-  
tails are provided in Appendix D.

5.2  
Results

Table 3 reports the accuracy of the exported students mea-  
sured on the Detectium sensor, together with the change  
relative to the corresponding models before export in Ta-  
ble 1. Overall, fire/non-fire detection accuracy is largely  
preserved after deployment, with an absolute change of at  
most 0.89 percentage across all students. In contrast, fine-  
grained fire-scene reasoning is more sensitive to deployment-  
time compression. For example, Llama-3.2-1B loses 4.07 in  
burning-object accuracy and 3.14 in environment accuracy  
after export. The two MobileCLIP-based students, which are  
executed directly without conversion or quantization, exhibit  
much smaller changes, with a maximum absolute difference  
of only 0.66 across the four tasks. This suggests that the  
degradation observed for the CLIP-based students is primar-  
ily associated with the export and quantization pipeline rather  
than with moving inference from GPU to CPU alone. Con-  
sistent with the distillation results in Section 4, fine-grained  
visual discrimination is the capability most sensitive to com-  
pression: it degrades under both distillation and deployment-  
time quantization. Nevertheless, the overall loss remains  
modest. For example, after both distillation and quantiza-  
tion, Qwen2.5-0.5B remains only 5.88 below its 7B teacher  
in burning-object accuracy.  
To better understand changes in detection behavior, Ap-  
pendix E, Table 11, decomposes fire/non-fire errors into false  
alarms and missed fires. For the quantized CLIP-based stu-  
dents, deployment introduces a directional shift toward pre-


---


Fire-scene reasoning  
Detection

Burning object  
Environment  
Severity  
Fire/non-fire

Model  
Vision encoder  
Acc.  
∆  
Acc.  
∆  
Acc.  
∆  
Acc.  
∆

Qwen2.5 family  
Qwen2.5-0.5B  
CLIP ViT-L/336  
75.95 ± 0.85  
−0.97  
85.10 ± 0.43  
−0.66  
88.37 ± 2.48  
+0.09  
99.52 ± 0.12  
−0.23  
Qwen2.5-1.5B  
CLIP ViT-L/336  
75.73 ± 1.38  
−1.46  
84.13 ± 0.28  
−1.28  
87.31 ± 1.20  
−0.13  
99.54 ± 0.20  
−0.02  
Qwen2.5-3B  
CLIP ViT-L/336  
76.88 ± 0.41  
−1.41  
86.34 ± 1.04  
+0.75  
88.82 ± 0.55  
0.00  
97.87 ± 0.62  
−0.64  
Qwen2.5-3B  
MobileCLIP-S2  
68.39 ± 1.86  
−0.40  
68.22 ± 1.89  
−0.39  
73.34 ± 1.39  
−0.09  
96.97 ± 0.59  
0.00

Qwen3 family  
Qwen3-0.6B  
CLIP ViT-L/336  
82.45 ± 0.20  
+2.79  
76.79 ± 0.69  
−3.90  
87.62 ± 0.50  
−4.94  
99.38 ± 0.12  
−0.30  
Qwen3-0.6B  
MobileCLIP-S2  
69.01 ± 1.00  
+0.44  
87.40 ± 0.66  
+0.66  
80.64 ± 0.53  
+0.18  
98.81 ± 0.08  
+0.07  
Qwen3-4B  
CLIP ViT-L/336  
78.87 ± 1.23  
−1.50  
85.76 ± 0.20  
−0.67  
89.74 ± 0.43  
+0.66  
99.29 ± 0.64  
−0.32

Llama-3 family  
Llama-3.2-1B  
CLIP ViT-L/336  
74.09 ± 0.81  
−4.07  
82.40 ± 1.10  
−3.14  
86.12 ± 0.60  
−1.28  
99.47 ± 0.04  
−0.30  
Llama-3.2-3B  
CLIP ViT-L/336  
77.59 ± 0.27  
−1.50  
84.92 ± 0.63  
−0.05  
85.41 ± 2.11  
−1.81  
98.88 ± 0.20  
−0.89

Gemma-2 family  
Gemma-2-2B  
CLIP ViT-L/336  
77.59 ± 0.70  
−1.63  
84.13 ± 0.80  
−0.71  
88.99 ± 1.51  
+0.44  
99.24 ± 0.18  
−0.32

Table 3: Fire-understanding accuracy (%) of the exported student models measured on the commercial Detectium fire detection  
sensor. ∆denotes the change relative to the corresponding student before export, as reported in Table 1. The best-performing  
student within each model family is shown in bold.

Model  
Vision  
TTFT  
E2E  
RSS Fits  
(ms)  
(ms) (MB)

Qwen2.5 family  
Qwen2.5-0.5B CLIP  
3818  
4034  
1880  
✓  
Qwen2.5-1.5B CLIP  
6862  
7419  
4210  
✓  
Qwen2.5-3B  
CLIP  
11598 12519  
7341  
✓  
Qwen2.5-3B  
MobileCLIP  
1757 18976  
8778  
×

Qwen3 family  
Qwen3-0.6B  
CLIP  
4469  
4770  
3050  
✓  
Qwen3-0.6B  
MobileCLIP  
893  
4227  
2952  
✓  
Qwen3-4B  
CLIP  
15253 16645  
9850  
×

Llama-3 family  
Llama-3.2-1B  
CLIP  
5720  
6905  
3624  
✓  
Llama-3.2-3B  
CLIP  
12077 13914  
7548  
✓

Gemma-2 family  
Gemma-2-2B  
CLIP  
9148 10146  
7251  
✓

Table 4: On-device latency and memory usage of the ex-  
ported student models. The final column indicates whether  
each model fits within the memory constraints of the Detec-  
tium sensor. Models marked × exceed the available memory  
and are benchmarked with additional memory for reference  
only. The best overall value for each metric is shown in bold.

dicting fire more frequently, increasing false alarms rather  
than missed fires. Moreover, this shift tends to become more  
pronounced for larger students, suggesting that smaller stu-  
dents are not only faster and more memory-efficient, but  
can also be more robust to deployment-time compression.  
MobileCLIP-based students exhibit the opposite and more  
safety-critical behavior: their errors are more likely to take  
the form of missed fires. In practical fire detection, a missed  
fire is substantially more consequential than a false alarm,  
making this failure mode undesirable despite the computa-

tional efficiency of the lightweight vision encoder.

Table 4 reports on-device latency and memory usage for  
each student. We find that latency is dominated by prefill  
rather than autoregressive generation. For every CLIP-based  
student, TTFT accounts for approximately 83%-95% of the  
total E2E latency. For example, Qwen2.5-0.5B requires 3818  
ms to produce its first token but only an additional 216 ms  
to generate the remainder of the response. We further find  
that the vision encoder is a major determinant of prefill la-  
tency. Replacing CLIP with MobileCLIP-S2 reduces TTFT  
substantially: from 4469 ms to 893 ms for Qwen3-0.6B and  
from 11598 ms to 1757 ms for Qwen2.5-3B. This reduc-  
tion is consistent with the substantially smaller visual rep-  
resentation produced by MobileCLIP-S2. However, because  
MobileCLIP-S2 is not supported by the llama.cpp ex-  
port pipeline and must instead be executed directly on the  
device, these models do not obtain the same optimized in-  
ference path. Consequently, the reduction in TTFT does not  
necessarily translate into lower E2E latency.

Finally, deployment cost grows steeply with student size,  
while accuracy does not. Across the evaluated Qwen models,  
E2E latency increases from 4034 ms for Qwen2.5-0.5B to  
16645 ms for Qwen3-4B, while RSS increases from 1880  
MB to 9850 MB. In contrast, the improvement in fire-scene  
reasoning accuracy from using larger students is limited, and  
the smallest models remain highly competitive in fire/non-  
fire detection (Table 3). Considering fire-understanding ca-  
pability, detection reliability, latency, and memory jointly,  
Qwen2.5-0.5B provides the strongest overall deployment  
trade-off: it is the fastest configuration at 4034 ms E2E la-  
tency, approximately an order of magnitude lower than that  
reported for conventional smoke detectors (Bukowski et al.  
2003). It also has the lowest memory usage and achieves  
overall best fire-understanding accuracy after export.


---


References  
Bai, S.; Cai, Y.; Chen, R.; Chen, K.; Chen, X.; Cheng, Z.;  
Deng, L.; Ding, W.; Gao, C.; Ge, C.; et al. 2025. Qwen3-vl  
technical report. arXiv preprint arXiv:2511.21631.  
Bukowski, R. W.; Peacock, R. D.; Averill, J. D.; Cleary,  
T. G.; Bryner, N. P.; and Reneke, P. A. 2003. Performance of  
Home Smoke Alarms, Analysis of the Response of Several  
Available Technologies in Residential Fire Settings.  
Cai, Y.; Zhang, J.; He, H.; He, X.; Tong, A.; Gan, Z.; Wang,  
C.; Xue, Z.; Liu, Y.; and Bai, X. 2025. Llava-kd: A framework  
of distilling multimodal large language models.  
In 2025  
IEEE/CVF International Conference on Computer Vision  
(ICCV), 239–249. IEEE.  
Çetin, A. E.; Dimitropoulos, K.; Gouverneur, B.; Gramma-  
lidis, N.; Günay, O.; Habiboˇglu, Y. H.; Töreyin, B. U.; and  
Verstockt, S. 2013. Video fire detection–review. Digital  
Signal Processing, 23(6): 1827–1843.  
Chagger, R.; and Smith, D. 2014. The causes of false fire  
alarms in buildings. Briefing Paper.  
Cherti, M.; Beaumont, R.; Wightman, R.; Wortsman, M.;  
Ilharco, G.; Gordon, C.; Schuhmann, C.; Schmidt, L.; and  
Jitsev, J. 2023. Reproducible scaling laws for contrastive  
language-image learning. In 2023 IEEE/CVF Conference on  
Computer Vision and Pattern Recognition (CVPR), 2818–  
2829. IEEE.  
Elhanashi, A.; Essahraui, S.; Dini, P.; and Saponara, S. 2025.  
Early fire and smoke detection using deep learning: A com-  
prehensive review of models, datasets, and challenges. Ap-  
plied Sciences, 15(18): 10255.  
Fonollosa, J.; Solórzano, A.; and Marco, S. 2018. Chemical  
sensor systems and associated algorithms for fire detection:  
A review. Sensors, 18(2): 553.  
Ghonim, K.; Roberto, A.; and Bernardi, D. 2026. When  
Speed Meets Intelligence: Scalable Conversational NER in  
an Ever-evolving World. In Proceedings of the 19th Confer-  
ence of the European Chapter of the Association for Compu-  
tational Linguistics (Volume 5: Industry Track), 366–376.  
Gragnaniello, D.; Greco, A.; Sansone, C.; and Vento, B.  
2025. Video fire recognition using zero-shot vision-language  
models guided by a task-aware object detector. ACM Trans-  
actions on Multimedia Computing, Communications and Ap-  
plications, 21(10): 1–24.  
Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian,  
A.; Al-Dahle, A.; Letman, A.; Mathur, A.; Schelten, A.;  
Vaughan, A.; et al. 2024. The llama 3 herd of models. arXiv  
preprint arXiv:2407.21783.  
Hinton, G.; Vinyals, O.; and Dean, J. 2015.  
Distill-  
ing the knowledge in a neural network.  
arXiv preprint  
arXiv:1503.02531.  
Hoefler, T.; and Belli, R. 2015. Scientific benchmarking of  
parallel computing systems: twelve ways to tell the masses  
when reporting performance results. In Proceedings of the  
international conference for high performance computing,  
networking, storage and analysis, 1–12.  
Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang,  
S.; Wang, L.; and Chen, W. 2021. Lora: Low-rank adaptation  
of large language models. arXiv preprint arXiv:2106.09685.

Ilharco, G.; Wortsman, M.; Carlini, N.; Taori, R.; Dave,  
A.; Shankar, V.; Namkoong, H.; Miller, J.; Hajishirzi, H.;  
Farhadi, A.; et al. 2021. Openclip. Zenodo.

Jin, Y.; Li, J.; Gu, T.; Liu, Y.; Zhao, B.; Lai, J.; Gan, Z.; Wang,  
Y.; Wang, C.; Tan, X.; et al. 2025. Efficient multimodal large  
language models: A survey. Visual Intelligence, 3(1): 27.

Kansal, K.; Wong, Y.; and Kankanhalli, M. 2025. Implica-  
tions of privacy regulations on video surveillance systems.  
ACM Transactions on Multimedia Computing, Communica-  
tions and Applications, 21(7): 1–27.

Khajavi, S. H.; Moshtaghi, M.; Yu, D.; Liu, Z.; Främ-  
ling, K.; and Holmström, J. 2024. Synthetic imagery for  
fuzzy object detection: A comparative study. arXiv preprint  
arXiv:2410.01124.

Khajavi, S. H.; Tetik, M.; Liu, Z.; Korhonen, P.; and Holm-  
ström, J. 2023. Digital twin for safety and security: Perspec-  
tives on building lifecycle. IEEE Access, 11: 52339–52356.

Kim, J.; Lee, Y.; Yoon, D.; Jung, C.; and Lee, G. 2025. An  
integrated yolo and vlm system for fire detection in enclosed  
environments. In I Can’t Believe It’s Not Better: Challenges  
in Applied Deep Learning.

Kim, Y.; and Rush, A. M. 2016. Sequence-level knowledge  
distillation. In Proceedings of the 2016 conference on em-  
pirical methods in natural language processing, 1317–1327.

Kumar, G.; Xi, Q.; Dabral, T. S.; Ghasemi, H.; Krishnamoor-  
thy, A.; Fu, D.; Min, R.; Antunez, E.; Ding, Z.; and Narayana,  
P. 2026. Enhancing User Safety: Context-Aware Detection  
of Offensive Query-Ad Pairs in Multimodal Search Advertis-  
ing. In Proceedings of the 19th Conference of the European  
Chapter of the Association for Computational Linguistics  
(Volume 5: Industry Track), 475–482.

Liu, G.; Yuan, H.; and Huang, L. 2023. A fire alarm judg-  
ment method using multiple smoke alarms based on Bayesian  
estimation. Fire safety journal, 136: 103733.

Liu, H.; Li, C.; Li, Y.; and Lee, Y. J. 2024. Improved baselines  
with visual instruction tuning. In 2024 IEEE/CVF Confer-  
ence on Computer Vision and Pattern Recognition (CVPR),  
26286–26296. IEEE.

Liu, H.; Li, C.; Wu, Q.; and Lee, Y. J. 2023. Visual instruction  
tuning. Advances in neural information processing systems,  
36: 34892–34916.

Liu, Z. 2023. Deep Learning based method for Fire Detec-  
tion.

Liu, Z.; H Khajavi, S.; and Jiang, G. 2026. Detectiumfire:  
A comprehensive multi-modal dataset bridging vision and  
language for fire understanding. Advances in Neural Infor-  
mation Processing Systems, 38.

Liu, Z.; Jiang, G.; and Khajavi, S. 2025. Llava-sco: Teach  
vision language models to self-correct. In 2025 IEEE/CVF  
Conference on Computer Vision and Pattern Recognition  
Workshops (CVPRW), 3406–3415. IEEE.

Mensch, A.; Veley, E.; and Cleary, T. 2024. Performance of  
New Smoke Alarms and Aerosol Measurements for a Range  
of Nuisance Cooking Sources in a Mock Kitchen.


---


Qi, T.; Li, W.; and Barnes, N. 2026. Smokebench: Evaluating  
multimodal large language models for wildfire smoke detec-  
tion. In 2026 IEEE/CVF Winter Conference on Applications  
of Computer Vision (WACV), 1043–1053. IEEE.  
Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh,  
G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark,  
J.; et al. 2021. Learning transferable visual models from  
natural language supervision. In International conference  
on machine learning, 8748–8763. PmLR.  
Regulation, E. 2016. 679 of the European Parliament and  
of the Council of 27 April 2016 on the protection of natural  
persons with regard to the processing of personal data and  
on the free movement of such data, and repealing Directive  
95/46/EC (General Data Protection Regulation).  
Official  
Journal of the European Union L, 119: 1–88.  
Seidel, L.; Gehringer, S.; Raczok, T.; Ivens, S.-N.; Eckardt,  
B.; and Maerz, M. 2025. Advancing early wildfire detection:  
Integration of vision language models with unmanned aerial  
vehicle remote sensing for enhanced situational awareness.  
Drones, 9(5): 347.  
Shu, F.; Liao, Y.; Zhang, L.; Zhuo, L.; Xu, C.; Zhang, G.;  
Shi, H.; Dai, W.; Yu, Z.; He, W.; et al. 2025. Llava-mod:  
Making llava tiny via moe-knowledge distillation. In Inter-  
national Conference on Learning Representations, volume  
2025, 9386–9404.  
Tannous, W. K. 2021. The economic cost of unwanted auto-  
matic fire alarms. Fire safety journal, 124: 103394.  
Team, G.; Mesnard, T.; Hardin, C.; Dadashi, R.; Bhupatiraju,  
S.; Pathak, S.; Sifre, L.; Rivière, M.; Kale, M. S.; Love, J.;  
et al. 2024. Gemma: Open models based on gemini research  
and technology. arXiv preprint arXiv:2403.08295.  
Vasu, P. K. A.; Gabriel, J.; Zhu, J.; Tuzel, O.; and Ranjan, A.  
2023. Fastvit: A fast hybrid vision transformer using struc-  
tural reparameterization. In 2023 IEEE/CVF international  
conference on computer vision (ICCV), 5762–5772. IEEE.  
Vasu, P. K. A.; Pouransari, H.; Faghri, F.; Vemulapalli, R.;  
and Tuzel, O. 2024.  
Mobileclip: Fast image-text models  
through multi-modal reinforced training. In 2024 IEEE/CVF  
Conference on Computer Vision and Pattern Recognition  
(CVPR), 15963–15974. IEEE.  
Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu,  
B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical  
report. arXiv preprint arXiv:2505.09388.


---


A  
Conclusion  
In this paper, we studied how far domain-specialized VLMs  
can be compressed for fully on-device fire understanding  
without sacrificing the behaviors required for reliable de-  
ployment. Across multiple VLM families and model scales,  
our experiments show that compact students preserve most of  
their teachers’ fire-understanding capability, while increas-  
ing student size provides only limited additional benefit. At  
the same time, our ablation study reveals that distillation  
quality cannot be judged solely by positive-class reasoning  
accuracy: distillation objective with only the KL term as in  
Eq. 2 can preserve strong fire-scene reasoning while inducing  
severe false-alarm bias, highlighting the importance of neg-  
ative examples in evaluating domain-specific compression.  
Real-world deployment further shows that compression af-  
fects not only accuracy, but also the type of errors made  
by the system, with different architectural and quantization  
choices shifting the balance between false alarms and missed  
fires. Among the evaluated models, Qwen2.5-0.5B provides  
the strongest overall trade-off between reasoning capability,  
detection reliability, latency, and memory usage on the target  
sensor.  
Although our experiments focus on fire detection, these  
findings are not specific to this domain. Other safety-critical  
and resource-constrained applications of VLMs face the  
same challenge of preserving domain-specific reasoning  
while satisfying strict deployment constraints. Our results  
therefore suggest broader lessons for on-device VLM de-  
ployment: smaller models can be preferable to larger ones,  
negative examples are essential for validating distilled mod-  
els, and deployment-time compression should be evaluated  
in terms of failure modes as well as accuracy and efficiency.

B  
Additional Implementation Details  
B.1  
Model Architecture  
All teacher and student models follow the LLaVA-MoD ar-  
chitecture (Shu et al. 2025), which consists of three main  
components: a vision encoder, a vision-language (VL) adap-  
tor, and a large language model (LLM). We summarize the  
architecture here for completeness. Given a multimodal in-  
struction pair x = (xv, xi), where xv denotes the input image  
and xi the text instruction, the model generates response y  
as  
y = LLM  
(︁  
Proj(ViT(xv)), xi  
)︁  
.  
ViT denotes the vision encoder, which extracts visual  
features from xv. We consider two vision encoders in  
our experiments. The first is the CLIP vision encoder  
(openai/clip-vit-large-patch14-336)  
(Rad-  
ford et al. 2021), which takes a 336 × 336 input image  
and produces 576 visual tokens using 14 × 14 image  
patches. The second is MobileCLIP-S2 (MCi2) (Vasu  
et al. 2024), a lightweight vision encoder based on  
FastViT (Vasu et al. 2023), which takes a 256 × 256  
input image and produces<span style="color:#ff0000"> ???</span> visual tokens. We load  
the apple/MobileCLIP-S2-OpenCLIP checkpoint  
through OpenCLIP (Ilharco et al. 2021).  
Proj denotes the VL adaptor, which maps the visual fea-  
tures produced by the vision encoder into the language-model

embedding space. Following LLaVA-MoD (Shu et al. 2025),  
we implement the adaptor as a two-layer MLP with a GELU  
activation.  
LLM denotes the language-model backbone, which gen-  
erates the response y conditioned on the projected visual  
features and text instruction. For the teacher models, we con-  
sider Qwen2.5-7B-Instruct, Qwen3-8B (Yang et al. 2025),  
Llama-3.1-8B-Instruct (Grattafiori et al. 2024), and Gemma-  
2-9B-IT (Team et al. 2024). Following LLaVA-MoD (Shu  
et al. 2025), each student is distilled from a teacher with a  
compatible vocabulary. Specifically, Qwen2.5-7B is distilled  
into Qwen2.5-0.5B, 1.5B, 3B-Instruct students; Qwen3-8B  
is distilled into Qwen3-0.6B, 4B students; Llama-3.1-8B  
is distilled into Llama-3.2-1B, 3B-Instruct students; and  
Gemma-2-9B is distilled into Gemma-2-2B-IT. Although  
the Llama teacher and students belong to different Llama  
releases, Llama-3.1 and Llama-3.2 use the same tokenizer  
and therefore share the vocabulary required for token-level  
distillation. Finally, unlike LLaVA-MoD (Shu et al. 2025),  
we do not sparsify the student LLMs. Each student retains  
the standard architecture of its corresponding base language  
model.

B.2  
Training Data  
Teacher  
Model.  
For  
the  
foundational  
initialization  
stage,  
we  
train  
the  
vision-language  
adaptor  
on  
the  
liuhaotian/LLaVA-Pretrain dataset (Liu et al.  
2023, 2024), a filtered set of 558K image-caption pairs used  
for visual-language feature alignment. For the subsequent  
task-specific fine-tuning stage, we use 35,486 training exam-  
ples, whose composition is summarized in Table 5.

Subset  
NO.  
Source

<span style="color:#ff0000">Short description</span>  
12,000  
LLaVA mix665k  
<span style="color:#ff0000">Bbox</span>  
2,409  
LLaVA mix665k  
<span style="color:#ff0000">Simple VQA</span>  
3,000  
LLaVA mix665k  
Fire VQA  
13,077  
DetectiumFire train  
Bbox + non-fire  
5,000  
DetectiumFire train

Table 5: Composition of the training data used in the task-  
specific fine-tuning stage.

<span style="color:#ff0000">Please explain how you got the data subset of short descrip-</span>  
<span style="color:#ff0000">tion, Bbox and Simple VQA. These things do not appear in</span>  
<span style="color:#ff0000">the original LLaVA mix665k dataset. We cannot name them</span>  
<span style="color:#ff0000">ourselves. Also, what question and answer/caption do we use</span>  
<span style="color:#ff0000">for each subset?</span>

To equip the teacher model with contextual fire-  
understanding capability, we fine-tune it on the Detectium-  
Fire training set (Liu, H Khajavi, and Jiang 2026), a compre-  
hensive multimodal fire dataset containing visual question-  
answering annotations for burning objects, surrounding envi-  
ronments, and fire severity, together with non-fire images. We  
use the official training split for fine-tuning and reserve the  
validation split for evaluation. For fire images, we directly  
use the provided visual question-answering (VQA) anno-  
tations as supervision. We additionally include 3,000 real-  
world non-fire images from DetectiumFire. This negative  
supervision is particularly important because false alarms


---


Table 6:<span style="color:#ff0000"> Training hyperparameters of each stage for teacher</span>  
<span style="color:#ff0000">models.</span>

Configuration  
Initialization  
Fine-Tuning

LLM  
✗  
✓  
VL Adaptor  
✓  
✓  
ViT  
✗  
✗

LLM sequence length  
Optimizer  
Optimizer hyperparameter  
Learning rate  
Learning rate schedule  
Weight decay  
Training epoch  
Warm-up ratio  
Global batch size  
Numerical precision  
Model parallelism

are a central challenge in practical fire detection (Section 1).  
It encourages the teacher not only to characterize fire scenes,  
but also to correctly reject benign scenes that may superfi-  
cially resemble fire.  
We additional include 2,000 synthetic fire bounding-box  
examples, where the images are generated from synthetic  
generation methods with bounding box annotation for fire.  
<span style="color:#ff0000">For synthetic data and non-fire images, what is the text in-</span>  
<span style="color:#ff0000">struction and caption for fine-tuning? Do we use the same</span>  
<span style="color:#ff0000">question as in the VQA data in DetectiumFire? Then what is</span>  
<span style="color:#ff0000">the answer or caption?</span>

Student Model.  
For the foundational initialization stage,  
similar to the teacher model, we train the adaptor using  
the liuhaotian/LLaVA-Pretrain dataset (Liu et al.  
2023, 2024). As for the distillation stage, we only use the  
DetectiumFire training dataset with fire.

<span style="color:#ff0000">For each fire image, we pair it with seven question types</span>  
<span style="color:#ff0000">used at evaluation (four for fire scenes, three for non-fire</span>  
<span style="color:#ff0000">scenes). This is hard to understand. Please give me the con-</span>  
<span style="color:#ff0000">crete 7 questions used for training and evaluation.</span>

B.3  
Training Configuration  
All models are trained on eight NVIDIA A100 GPUs with  
80,GB of memory per GPU.

Teacher Model.  
We fine-tune the teacher models using  
low-rank adaptation (LoRA) (Hu et al. 2021) rather than full-  
parameter fine-tuning. The detailed training hyperparameters  
are summarized in Table 6.

Student Model.  
The detailed training hyperparameters for  
the student models are summarized in Table 7. All experi-  
ments are conducted using the same three random seeds: 42,  
123, and 456.

C  
Additional Experimental Results  
C.1  
Ablation Study  
Table 8 compares students trained with the full objective in  
Eq. 4 against those trained with the KL term in Eq. 2 alone.

Table 7:<span style="color:#ff0000"> Training hyperparameters of each stage for student</span>  
<span style="color:#ff0000">models.</span>

Configuration  
Initialization  
Distillation

LLM  
✗  
✓  
VL Adaptor  
✓  
✓  
ViT  
✗  
✗

LLM sequence length  
Optimizer  
Optimizer hyperparameter  
Learning rate  
Learning rate schedule  
Weight decay  
Training epoch  
Warm-up ratio  
Global batch size  
Numerical precision  
Model parallelism

Removing LCE causes a severe collapse in fire/non-fire detec-  
tion, with most students losing between 15.70 and 45.99 per-  
centage points in accuracy. This degradation persists across  
model scales. For example, Qwen2.5-0.5B drops to 53.76%  
accuracy, while Qwen3-4B, despite being eight times larger,  
still falls to 81.04%. Similarly, Llama-3.2-3B performs even  
worse than the smaller Llama-3.2-1B (55.72% vs. 58.43%).  
In contrast, the collapse is nearly invisible on the fire-  
scene reasoning tasks.Removing the cross-entropy termLCE  
leaves burning-object, environment, and severity accuracy  
largely unchanged and, in many cases, even nominally im-  
proves performance. For example, burning-object accuracy  
increases for four of the seven students, including gains of  
3.05 points for Llama-3.2-1B and 2.43 points for Qwen2.5-  
1.5B. Because these three tasks are evaluated only on images  
containing fire, a model that predicts fire indiscriminately is  
not penalized by these benchmarks. Consequently, a practi-  
tioner selecting a distillation objective solely on the basis of  
fire-scene reasoning performance could favor the KL-only  
objective even though it produces a model that is unsuitable  
for deployment. This ablation therefore highlights a central  
practical lesson: domain-specific distillation must be vali-  
dated on negative examples.  
Table 9 reports fire/non-fire detection performance for the  
teacher models. All teachers achieve consistently strong re-  
sults, producing at most two false alarms and at most two  
missed fires. This confirms that the collapse observed in stu-  
dents trained without LCE (Table 2) is not inherited from the  
teacher models, but instead emerges during distillation.

D  
Additional Deployment Configuration  
D.1  
Target Sensor  
We evaluate the distilled students on our commercial Detec-  
tium fire detection sensor 3. The sensor is built on a Raspberry  
Pi 5 with 8 GB of memory and runs Raspberry Pi OS Lite 64-  
bit (Debian 13, glibc 2.41). It is equipped with a Broadcom  
BCM2712 SoC containing four Arm Cortex-A76 CPU cores

3https://www.detectium.io/sensor-package


---


Fire-scene reasoning  
Detection

Model  
Objective  
Burning object  
Environment  
Severity  
Fire/non-fire  
∆

Qwen2.5 family

Qwen2.5-0.5B  
LKL only  
77.59 ± 0.48  
86.16 ± 0.28  
88.81 ± 0.31  
53.76 ± 0.72  
+ LCE  
76.92 ± 1.16  
85.76 ± 1.48  
88.28 ± 1.46  
99.75 ± 0.04  
+45.99

Qwen2.5-1.5B  
LKL only  
79.62 ± 0.15  
87.23 ± 0.43  
88.46 ± 0.27  
62.14 ± 1.93  
+ LCE  
77.19 ± 1.63  
85.41 ± 1.61  
87.44 ± 1.30  
99.56 ± 0.17  
+37.42

Qwen2.5-3B  
LKL only  
76.44 ± 1.26  
85.23 ± 0.81  
86.47 ± 0.87  
98.35 ± 0.18  
+ LCE  
78.29 ± 1.58  
85.59 ± 0.41  
88.82 ± 1.03  
98.51 ± 0.46  
+0.16

Qwen3 family

Qwen3-4B  
LKL only  
80.02 ± 0.31  
86.21 ± 0.48  
88.73 ± 0.58  
81.04 ± 1.97  
+ LCE  
80.37 ± 1.48  
86.43 ± 0.68  
89.08 ± 0.50  
99.61 ± 0.28  
+18.57

Llama-3 family

Llama-3.2-1B  
LKL only  
81.21 ± 0.50  
87.97 ± 0.50  
88.11 ± 0.63  
58.43 ± 0.58  
+ LCE  
78.16 ± 0.15  
85.54 ± 0.13  
87.40 ± 0.58  
99.77 ± 0.04  
+41.34

Llama-3.2-3B  
LKL only  
79.44 ± 0.40  
86.61 ± 0.61  
86.69 ± 0.67  
55.72 ± 1.65  
+ LCE  
79.09 ± 0.63  
84.97 ± 0.80  
87.22 ± 1.83  
99.77 ± 0.04  
+44.05

Gemma-2 family

Gemma-2-2B  
LKL only  
79.93 ± 0.80  
86.61 ± 0.93  
79.93 ± 2.28  
83.86 ± 1.14  
+ LCE  
79.22 ± 1.31  
84.84 ± 0.15  
88.55 ± 1.25  
99.56 ± 0.24  
+15.70

Table 8: Effect of the distillation objective on fire-understanding accuracy (%) across model families on the DetectiumFire  
evaluation set. ∆denotes the change in fire/non-fire detection accuracy. Removing LCE leaves performance on the three fire-  
scene reasoning tasks largely unchanged, and in several cases slightly improves it, while reducing fire/non-fire detection accuracy  
by up to 45.99 percentage.

Errors (counts)  
Rates (%)

Teacher  
Encoder  
False alarms  
Missed fires  
Recall  
Specificity  
Accuracy

Qwen2.5-7B  
CLIP  
2  
1  
99.87  
99.71  
99.79  
Qwen2.5-7B  
MobileCLIP  
2  
2  
99.73  
99.71  
99.72  
Qwen3-8B  
CLIP  
1  
1  
99.87  
99.86  
99.86  
Qwen3-8B  
MobileCLIP  
2  
2  
99.73  
99.71  
99.72  
Llama-3.1-8B  
CLIP  
1  
1  
99.87  
99.86  
99.86  
Gemma-2-9B  
CLIP  
1  
0  
100.00  
99.86  
99.93

Table 9: Safety-relevant detection errors for the teacher mod-  
els.

(ARMv8.2-A) clocked at up to 2.4 GHz. No GPU or NPU is  
used during inference. To reserve sufficient memory for the  
operating system and other system processes, we impose a 7  
GiB memory cap on model inference. All experiments use  
four CPU threads, one per core.

Family  
Prompt format

Qwen2.5  
USER: &lt;image&gt;{prompt} ASSISTANT:  
Qwen3  
<span style="color:#ff0000">[fill]</span>  
Llama-3.2  
<span style="color:#ff0000">[fill]</span>  
Gemma-2  
<span style="color:#ff0000">[fill]</span>

Table 10:<span style="color:#ff0000"> Prompt template used for each student family at</span>  
<span style="color:#ff0000">inference, matching the format used during training.</span>

D.2  
Export Pipeline  
Students with CLIP Encoder.  
Because the student mod-  
els are trained using LoRA (Hu et al. 2021), each trained  
checkpoint consists of a PEFT LoRA adapter together with  
a two-layer MLP vision-language projector. We export each  
CLIP-based student using the llama.cpp toolchain 4 in

4https://github.com/ggml-org/llama.cpp

four steps. First, the LoRA adapter is merged into the cor-  
responding base language model to obtain dense float16  
weights. Second, the merged language model is converted  
to GGUF format and quantized to Q8_0. Third, the CLIP vi-  
sion encoder and the student-specific two-layer MLP vision-  
language projector are packaged into a separate F16 multi-  
modal projector file. Because the projector is trained inde-  
pendently for each student, this file cannot be shared across  
models. Fourth, the chat template used during training is  
embedded into the GGUF file so that prompts are formatted  
consistently with the training procedure at inference time.  
The template used for each model family is summarized in  
Table 10. We found this step to be important in practice: a  
mismatched chat template does not necessarily cause an ex-  
plicit runtime error and can still produce fluent outputs, but  
substantially increases the false-alarm rate.  
The final deployed configuration therefore consists of a  
Q8_0-quantized language model together with an F16 multi-  
modal vision module. At inference time, the exported files are  
served using llama-server from llama.cpp, which  
loads the model and performs inference directly on the sen-  
sor CPU.

Students with MobileCLIP Encoder.  
The llama.cpp  
conversion pipeline supports the CLIP-based vision encoder  
used in our main configurations, but does not support the  
FastViT (Vasu et al. 2023) backbone used by MobileCLIP-  
S2. As a result, the MobileCLIP-S2 students, Qwen3-0.6B  
and Qwen2.5-3B, cannot be exported through the same  
GGUF pipeline. Instead, these models are executed directly  
on the sensor CPU using PyTorch in bfloat16 precision, with  
the LoRA adapter applied at load time and the vision encoder


---


Errors (counts)  
Rates (%)

False alarms  
Missed fires  
Recall  
Specificity

Model  
Vision encoder  
Count  
∆  
Count  
∆  
Rate  
∆  
Rate  
∆

Qwen2.5 family  
Qwen2.5-0.5B  
CLIP ViT-L/336  
5.7 ± 2.5  
+3.7  
1.3 ± 1.2  
−0.4  
99.83 ± 0.16  
+0.05  
99.18 ± 0.36  
−0.53  
Qwen2.5-1.5B  
CLIP ViT-L/336  
5.0 ± 1.7  
+0.3  
1.7 ± 1.2  
0.0  
99.77 ± 0.16  
0.00  
99.28 ± 0.24  
−0.04  
Qwen2.5-3B  
CLIP ViT-L/336  
27.7 ± 9.0  
+10.4  
3.3 ± 0.6  
−1.0  
99.56 ± 0.08  
+0.13  
96.03 ± 1.29  
−1.49  
Qwen2.5-3B  
MobileCLIP-S2  
4.7 ± 1.5  
+0.4  
39.3 ± 9.7  
−0.4  
94.79 ± 1.29  
+0.05  
99.33 ± 0.21  
−0.06

Qwen3 family  
Qwen3-0.6B  
CLIP ViT-L/336  
6.3 ± 2.1  
+2.6  
2.7 ± 0.6  
+1.7  
99.64 ± 0.08  
−0.23  
99.10 ± 0.30  
−0.37  
Qwen3-0.6B  
MobileCLIP-S2  
2.7 ± 2.1  
−0.3  
14.7 ± 2.1  
−0.6  
98.05 ± 0.28  
+0.08  
99.61 ± 0.30  
+0.04  
Qwen3-4B  
CLIP ViT-L/336  
9.0 ± 8.7  
+4.7  
1.3 ± 0.6  
0.0  
99.83 ± 0.08  
0.00  
98.71 ± 1.25  
−0.67

Llama-3 family  
Llama-3.2-1B  
CLIP ViT-L/336  
4.7 ± 0.6  
+4.0  
3.0 ± 1.0  
+0.3  
99.60 ± 0.13  
−0.04  
99.33 ± 0.08  
−0.57  
Llama-3.2-3B  
CLIP ViT-L/336  
16.0 ± 2.6  
+14.7  
0.3 ± 0.6  
−1.7  
99.96 ± 0.08  
+0.23  
97.71 ± 0.37  
−2.11

Gemma-2 family  
Gemma-2-2B  
CLIP ViT-L/336  
10.3 ± 2.5  
+5.3  
0.7 ± 0.6  
−0.6  
99.91 ± 0.08  
+0.08  
98.52 ± 0.36  
−0.76

Table 11: Safety-relevant detection errors of the exported student models on the Detectium sensor. ∆denotes the change relative  
to the corresponding student before export, as reported in Table 2.

loaded through open_clip_torch (Ilharco et al. 2021;  
Cherti et al. 2023). The same four CPU threads are used as  
in the GGUF experiments.  
Because the MobileCLIP-S2 models use a different run-  
time and numerical precision from the quantized GGUF  
models, their latency and accuracy results are not strictly  
directly comparable to those of the CLIP-based students. We  
therefore report them primarily to characterize the deploy-  
ment behavior of the lightweight vision encoder under the  
available runtime.

D.3  
Measurement

For evaluation, we use the same DetectiumFire evaluation set  
as in the main experiments. Each inference request contains  
a single previously unseen image together with the prompt  
corresponding to the fire-understanding task. We use deter-  
ministic decoding with temperature set to 0 and a maximum  
generation length of 64 new tokens. No image or prompt  
cache is reused across requests, so each request indepen-  
dently re-encodes the input image, matching the behavior of  
a sensor processing a live image stream.  
We report three deployment metrics. Time to first to-  
ken (TTFT) measures the interval from receiving an in-  
ference request to generating the first output token. For a  
vision-language model, TTFT primarily captures the prefill  
stage, including image encoding, projection of visual fea-  
tures into the language-model embedding space, and the ini-  
tial language-model forward pass. End-to-end latency (E2E)  
measures the total inference time, including both TTFT and  
autoregressive generation of the remaining output tokens.  
Peak resident set size (RSS) measures the maximum physical  
memory occupied by the inference process during execution.  
For each metric, we report the median over 20 randomly sam-  
pled fire-understanding requests from the validation set rather

than the mean, following standard benchmarking practice for  
latency measurements (Hoefler and Belli 2015).  
Qwen2.5-3B with MobileCLIP-S2 and Qwen3-4B require  
approximately 8.8 GB and 9.9 GB of memory, respectively,  
exceeding the memory capacity available on the target De-  
tectium sensor. We therefore report their measurements for  
reference only. Neither configuration is deployable under the  
target hardware constraints.

E  
Additional Deployment Results

Table 11 decomposes on-device fire/non-fire detection error  
into false alarms and missed fires, together with recall and  
specificity. The results reveal more clearly how quantization  
changes the prediction behavior of the distilled students. For  
every CLIP-based student, the number of false alarms in-  
creases after export and specificity consequently decreases.  
Thus, the effect of quantization is not simply a general loss  
of accuracy, but a directional shift toward predicting fire  
more frequently. In contrast, the two MobileCLIP-based stu-  
dents, which are executed without quantization, exhibit no  
comparable shift.  
Moreover, the magnitude of this shift increases with model  
size. Among the CLIP-based students, the four models with  
1.5B parameters or fewer produce, on average, 2.7 addi-  
tional false alarms after export, whereas the four models with  
2B parameters or more produce 8.8 additional false alarms.  
Llama-3.2-3B provides the clearest example, producing 14.7  
additional false alarms and losing 2.11 percentage points of  
specificity after export, compared with increases of 4.0 false  
alarms and a 0.57-point specificity drop for Llama-3.2-1B.  
These results suggest that larger students are more sensi-  
tive to quantization. Consequently, smaller students are not  
only faster and more memory-efficient, but also degrade less  
under the compression required for on-device deployment.


---


The MobileCLIP-based students exhibit a different failure  
mode. Although they produce only 4.7 and 2.7 false alarms,  
respectively, among the lowest values in Table 11, they miss  
39.3 and 14.7 fires. This failure mode is particularly con-  
cerning for real-world deployment. False alarms produced  
by the CLIP-based models can still be resolved through hu-  
man verification, whereas a missed fire may remain entirely  
undetected and therefore carries substantially greater safety  
consequences. Thus, despite their low false-alarm rates, the  
high number of missed fires makes the MobileCLIP-based  
configurations less suitable for practical fire detection.
