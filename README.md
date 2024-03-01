# Generative AI:

Generative AI or generative artificial intelligence refers to the use of AI to generate high-quality new content, like text, images, music, audio, and videos based on the data they were trained on, using the deep-learning models.

Generative AI encompasses unsupervised and semi-supervised machine learning techniques that empower computers to utilize pre-existing content such as text, audio, video files, images, and code to produce novel content. The primary objective is to generate entirely original artifacts that closely resemble the original contents.

Generative artificial intelligence  is artificial intelligence capable of generating text, images or other data using generative models, often in response to prompts. Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.

# The goal of Generative AI is to minimize the loss function, the statistical difference between the model’s outputs and the data it was trained on. 

# Four Broad Categories of Generative AI 
1) Generative Adversarial Networks are trained through a two-step process. The generator network learns how to create fake data from random noise. At the same time, the discriminator network learns the difference between real and fake data. The result is a generator network capable of creating high-quality, realistic data samples.

Generative Adversarial Networks or GANs — technologies that can create visual and multimedia artifacts from both imagery and textual input data.
Generative Adversarial Networks (GANs) are a class of deep learning models, consisting of two neural networks, the generator and the discriminator, which are trained simultaneously in a competitive setting.

GANs are a type of generative model where the generator network learns to generate realistic data samples (e.g., images, audio, text) that are indistinguishable from real data, while the discriminator network learns to differentiate between real and generated samples.

Examples of GANs:

--Deep Convolutional GANs (DCGANs): DCGANs utilize deep convolutional neural networks in both the generator and discriminator networks, enabling stable training and high-quality image generation.
--Conditional GANs (cGANs): cGANs introduce additional conditioning information, such as class labels or auxiliary data, to both the generator and discriminator networks, allowing for more controlled and targeted generation of data samples.

# 2) Variational Autoencoders (VAEs) [particularly well-suited for generating images]: 
Variational Autoencoders (VAEs) are also trained through a two-part process. The encoder network maps input data to a latent space, where it’s represented as a probability distribution. The decoder network then samples from this distribution to reconstruct the input data. During training, VAEs seek to minimize a loss function that includes two components: reconstruction and regularization. The balance between reconstruction and regularization allows VAEs to generate new data samples by sampling from the learned latent space.

# 3) Transformer-based models (e.g., LLMs) 

Transformer-based models (e.g., LLMs) — technologies such as Generative Pre-Trained (GPT) language models that can use information gathered on the Internet to create textual content from website articles to press releases to whitepapers.

Transformer-based models have revolutionized natural language processing (NLP) by introducing a new architecture that relies entirely on self-attention mechanisms and feed-forward neural networks, eliminating the need for recurrent or convolutional layers. 

Transformer Models are trained with a two-step process, as well. First, they are pre-trained on a large dataset. Then, they are fine-tuned with a smaller, task-specific dataset. The combination of pre-training and fine-tuning allows transformer models to use supervised, unsupervised, and semi-supervised learning, depending on the available data and the specific task. This flexibility enables the same transformer model to be used for different types of content.

Examples of Transformer-based models:

--A) Self-Attention Mechanism:

The self-attention mechanism allows Transformer models to weigh the importance of different tokens in a sequence when computing representations. 
It calculates a weighted sum of the input embeddings, where the weights are determined by the similarity between tokens.
Algorithms: Multi-head attention, Scaled Dot-Product Attention.

--B) Feed-Forward Neural Networks (FFNN):

Transformer models employ feed-forward neural networks in their encoder and decoder layers. 
These networks apply linear transformations followed by non-linear activation functions to the input representations.
Algorithms: Feed-forward neural networks in Transformer layers.

--C) Masked Self-Attention:

In some Transformer-based models like BERT (Bidirectional Encoder Representations from Transformers), a masking mechanism is employed during self-attention to prevent tokens from attending to future tokens during training tasks like masked language modeling.
Algorithms: Masked self-attention in BERT.

# Large Language Models (LLMs) fall under the category of Transformer-based models.

LLMs, such as OpenAI's GPT (Generative Pre-trained Transformer) series and BERT (Bidirectional Encoder Representations from Transformers), are based on the Transformer architecture. They utilize self-attention mechanisms and feed-forward neural networks to process and generate natural language text.

# 4) Hybrid Generative AI Models 
 Hybrid Generative AI Models are trained with a combination of techniques. The exact details for training a hybrid generative AI model will vary depending on the specific architecture, its objectives, and the data type involved. 























SOURCES:

Link 1: https://research.ibm.com/blog/what-is-generative-AI

Link 2: https://en.wikipedia.org/wiki/Generative_artificial_intelligence

Link 3: https://www.techopedia.com/definition/34633/generative-ai


