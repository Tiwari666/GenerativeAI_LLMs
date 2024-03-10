# Generative AI:

# Note:
 Not all generative AI methods are based solely on neural networks. While neural networks, particularly deep learning models, have gained significant attention and success in generative tasks due to their ability to learn complex patterns and structures from data. 

Generative AI or generative artificial intelligence refers to the use of AI to generate high-quality new content, like text, images, music, audio, and videos based on the data they were trained on, using the deep-learning models.

Generative AI encompasses unsupervised and semi-supervised machine learning techniques that empower computers to utilize pre-existing content such as text, audio, video files, images, and code to produce novel content. The primary objective is to generate entirely original artifacts that closely resemble the original contents.

Generative artificial intelligence  is artificial intelligence capable of generating text, images or other data using generative models, often in response to prompts. Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.

# The goal of Generative AI is to minimize the loss function, the statistical difference between the model’s outputs and the data it was trained on. 

# Four Broad Categories of Generative AI 
# 1) Generative Adversarial Networks (GANs)-- based on neural networks architecture--: consist of two neural networks: a generator and a discriminator.

Generator: The generator neural network takes random noise or a latent vector as input and generates data samples (e.g., images) that are intended to resemble samples from the real data distribution. It learns to map points from a latent space to the data space.

Discriminator: The discriminator neural network takes both real data samples and generated samples as input and learns to distinguish between them. It aims to differentiate between real and fake samples, essentially acting as a binary classifier.
The two networks play a guessing game in which the generator gives the discriminator a data sample, and the discriminator predicts whether the sample is real or something the generator made up. The process is repeated until the generator can fool the discriminator with an acceptable level of accuracy. 

Generative Adversarial Networks (GANs) are trained through a two-step process. The generator network learns how to create fake data from random noise. At the same time, the discriminator network learns the difference between real and fake data. The result is a generator network capable of creating high-quality, realistic data samples.

Generative Adversarial Networks or GANs — technologies that can create visual and multimedia artifacts from both imagery and textual input data.
Generative Adversarial Networks (GANs) are a class of deep learning models, consisting of two neural networks, the generator and the discriminator, which are trained simultaneously in a competitive setting.

GANs are a type of generative model where the generator network learns to generate realistic data samples (e.g., images, audio, text) that are indistinguishable from real data, while the discriminator network learns to differentiate between real and generated samples.

Examples of GANs:

--Deep Convolutional GANs (DCGANs): DCGANs utilize deep convolutional neural networks in both the generator and discriminator networks, enabling stable training and high-quality image generation.
--Conditional GANs (cGANs): cGANs introduce additional conditioning information, such as class labels or auxiliary data, to both the generator and discriminator networks, allowing for more controlled and targeted generation of data samples.

# Conclusion: 

Generative adversarial networks are a type of machine learning model that uses deep learning techniques to generate new data based on patterns learned from existing data.

GANs consist of two neural networks, generators, and discriminators, that work together to create the most effective generative models. 

The generator-- being a convolutional neural network-- and the discriminator-- being a deconvolutional neural network--are fighting against each other to get one step ahead of the other, hence the name 'adversarial'.

The discriminator network tries to determine whether the image it's given is real or fake. However, the generator tries to fool the discriminator.

Also, they are dependent on each other for efficient training. If one of them fails, the whole system fails. So you have to make sure they don’t explode.

# 2) Variational Autoencoders (VAEs)--based on the neural network architecture-- [particularly well-suited for generating images]: 

Variational Autoencoders (VAEs) are trained through a two-part process: an encoder and a decoder. The encoder takes input data and compresses it into a latent space representation as a probability distribution that preserves its most important features. The decoder then takes the latent space representation and generates new data that captures the most important features of the training data.  During training, VAEs seek to minimize a loss function (reconstruction error) that includes two components: reconstruction and regularization. The balance between reconstruction and regularization allows VAEs to generate new data samples by sampling from the learned latent space.



# 3) Transformer-based models--based on neural network architecture-- (e.g., LLMs) 
e.g.,  OpenAI's GPT (Generative Pre-trained Transformer) series, Google's BERT (Bidirectional Encoder Representations from Transformers)

Transformer architectures consist of multiple stacked layers, each containing its own self-attention mechanism and feed-forward network. The self-attention mechanism enables each element in a sequence to consider and weigh its relationship with all other elements, and the feed-forward network processes the output of the self-attention mechanism and performs additional transformations on the data. As the model processes an input sequence through the stacked layers, it learns to generate new sequences that capture the most important information for the task.

Generative Pre-trained Transformers (GPTs) are a specific implementation of the transformer architecture. This type of model is first pre-trained on vast amounts of text data to capture linguistic patterns and nuances. Once the foundation training has been completed, the model is then fine-tuned for a specific use. 

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

# LangChain 
LangChain is an intuitive open-source framework created to simplify the development of applications using large language models (LLMs), such as OpenAI or Hugging Face. This allows us to build dynamic, data-responsive applications that harness the most recent breakthroughs in natural language processing (NLP).

# 4) Hybrid Generative AI Models 
 Hybrid Generative AI Models are trained with a combination of techniques. The exact details for training a hybrid generative AI model will vary depending on the specific architecture, its objectives, and the data type involved. 

# 5) Markov Models:
Markov models are probabilistic models that use the Markov property, which states that the future state of a system depends only on its current state and not on its past states. Markov models can be used for generating sequences of data by modeling the transition probabilities between different states. E.G., Hidden Markov Models (HMMs) and Markov Chain Monte Carlo (MCMC).

# 6) RNN 
# 7) LSTM



















# SOURCES:

Link 1: https://research.ibm.com/blog/what-is-generative-AI

Link 2: https://en.wikipedia.org/wiki/Generative_artificial_intelligence

Link 3: https://www.techopedia.com/definition/34633/generative-ai

Link 4 (see) : https://blog.enterprisedna.co/what-is-langchain-a-beginners-guide-with-examples/


