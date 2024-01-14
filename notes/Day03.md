#  Day03 基于 InternLM 和 LangChain 搭建知识库

## 大模型开发范式

### LLM 的局限性

- 知识时效性首先：训练数据集截止时间之前的知识，之后的未知
- 专业能力有限：通用大模型 vs 垂域大模型
- 定制化成本高

![image-20240114111711249](assets/Day03/image-20240114111711249.png)

### RAG vs Finetune

![image-20240114111913915](assets/Day03/image-20240114111913915.png)

### RAG 检索增强

![image-20240114112111996](assets/Day03/image-20240114112111996.png)

## LangChain 简介

![image-20240114112207620](assets/Day03/image-20240114112207620.png)

- 检索问答链：最基本的链

### 基于LangChain搭建RAG应用

![image-20240114112234763](assets/Day03/image-20240114112234763.png)

## 构建向量数据库

### 加载源文件-文档分块-文档向量化

![image-20240114112334978](assets/Day03/image-20240114112334978.png)

## 搭建知识库助手

- 将InternLM接入LangChain

  ![image-20240114112426504](assets/Day03/image-20240114112426504.png)

- 构建检索问答链

![image-20240114112526871](assets/Day03/image-20240114112526871.png)

- RAG方案优化建议

![image-20240114112558214](assets/Day03/image-20240114112558214.png)

## Web Demo 部署

- 支持简易Web部署的框架，如Gradio、Streamlit等

![image-20240114112631927](assets/Day03/image-20240114112631927.png)





课程资料：

- 课程视频：https://www.bilibili.com/video/BV1Rc411b7ns
- OpenXLab：https://studio.intern-ai.org.cn
- 学习手册：https://kvudif1helh.feishu.cn/docx/Xx8hdqGwmopi5NxWxNWc76AOnPf



- 基于大模型搭建金融场景智能问答系统：https://github.com/Tongyi-EconML/FinQwen
- 天池LLM大模型：https://tianchi.aliyun.com/competition/entrance/532172
- https://huggingface.co/datasets/arxiv_dataset
- MirrorZ Help 开源镜像: https://help.mirrors.cernet.edu.cn/