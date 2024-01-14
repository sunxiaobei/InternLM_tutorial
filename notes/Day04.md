#  Day04 XTuner 大模型单卡低成本微调实战

- 汪周谦：眼科人工智能系统的临床应用，使用大语言模型与CV模型构建多模态的诊疗体系的研究

## Finetune简介

- 增量训练和指令跟随
- 增量预训练：新知识，垂域领域的常识

![image-20240114114117550](assets/Day04/image-20240114114117550.png)

### 指令跟随微调

![image-20240114114256538](assets/Day04/image-20240114114256538.png)


![image-20240114114356360](assets/Day04/image-20240114114356360.png)

- 对话模板

![image-20240114114504664](assets/Day04/image-20240114114504664.png)

- 指令微调训练

![image-20240114114521690](assets/Day04/image-20240114114521690.png)

### 增量预训练微调

![image-20240114114636922](assets/Day04/image-20240114114636922.png)

### LoRA & QLoRA

- 在原始模型参数 , 增加旁路分支(组件)

![image-20240114114946333](assets/Day04/image-20240114114946333.png)

- 微调-LoRA-QLoRA

![image-20240114115036653](assets/Day04/image-20240114115036653.png)



## Xtuner介绍

![image-20240114115129733](assets/Day04/image-20240114115129733.png)



![image-20240114115058772](assets/Day04/image-20240114115058772.png)



### Xtuner 快速上手

![image-20240114115419587](assets/Day04/image-20240114115419587.png)

- 自定义微调

![image-20240114115522792](assets/Day04/image-20240114115522792.png)

- 对话

![image-20240114115549674](assets/Day04/image-20240114115549674.png)

- 更多方式

![image-20240114115610848](assets/Day04/image-20240114115610848.png)





课程资料：

- 课程视频：https://www.bilibili.com/video/BV1Rc411b7ns
- OpenXLab：https://studio.intern-ai.org.cn
- 学习手册：https://kvudif1helh.feishu.cn/docx/Xx8hdqGwmopi5NxWxNWc76AOnPf



- 基于大模型搭建金融场景智能问答系统：https://github.com/Tongyi-EconML/FinQwen
- 天池LLM大模型：https://tianchi.aliyun.com/competition/entrance/532172
- https://huggingface.co/datasets/arxiv_dataset
- MirrorZ Help 开源镜像: https://help.mirrors.cernet.edu.cn/