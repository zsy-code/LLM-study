# LLM-Study

### Introduction

This is a repository for documenting the learning process of LLM. It mainly consists of two parts: the fine-tuning section and the algorithms section. I will provide a comprehensive theoretical overview, formula derivations, and executable sample codes related to all algorithms, to assist all interested friends in reference and learning.

Welcome everyone's opinions and suggestions on any content in the document. I will listen to your feedback at any time and improve and revise the document's content accordingly. You can send detailed feedback to the email address **zdm.code@gmail.com**. Alternatively, you can also open an issue.

Notice: I will eventually provide documentation in both Chinese and English versions, but at the initial stage, only Chinese documentation will be available.

### Recent Updates
- 2024-08-08: Accomplished a more detailed explanation of the sub-layer principles and descriptions within the Transformer (PE, LN, etc.), introduced the structural differences between the encoder and decoder, and included some code implementations (in `code/transformer`).
- 2024-08-06: Completed the basic explanation of the Transformer model, including the fundamental architecture of transformers and the attention mechanism.
- 2024-08-03: The first version of the document is released. Some general content about the basic knowledge of large language models has been uploaded.

### TODO List

- [x] Organize the document structure and establish the repository.
- [ ] Complete the general knowledge collation for large language models.
- [ ] Complete the content for the model fine-tuning section.
- [ ] Complete the content for the model algorithms section.

### Section Structure

#### The Fine-Tuning Section
| Unit Name | Covered Content | Article Link
| --- | --- | --- |
| Basic Knowledge of LM | <ul><li>Definition and Importance of Large Models</li> <li>Development History and Key Milestones of Large Models</li><li>Basic Concepts of Pretraining and Fine-Tuning</li> <li>Role of Data Processing and Alignment</li> <li>Infrastructure and Resource Requirements for Training Large Models</li> <li>Challenges Faced by Large Models and Future Development Directions</li></ul>| [Link](general/unit_1_bacic_knowledge_of_LM_zh.md)
| Analysis of Transformer Model Principles(Part 1)| <ul><li>Basic Architecture of the Transformer Model</li> <li>Principles and Calculation Process of the Self-Attention Mechanism</li> <li>Design and Role of Multi-Head Attention</li> <li>Role and Advantages of Self-Attention in the Model</li> </ul>| [Link](general/unit_2_analysis_of_the_principles_behind_transformer_1.md)
| Anaysis of Transformer Model Principles(Part 2)| <ul><li>The concept and implementation method of Position Encoding</li> <li>The Feed Forward Network in the Transformer</li> <li>The principle and importance of Layer Normalization in the Transformer</li> <li>The structural differences between the encoder and decoder in the Transformer.</li></ul>| [Link](general/unit_3_analysis_of_the_principles_behind_transformer_2.md)