# Generative AI:


Generative AI recognizes patterns and structures within the existing data to produce unique and original content that may include text,audio, visual, code, 3D models, or other types of data. For training, generative AI models can use a variety of learning strategies, such as unsupervised or semi-supervised learning. As a result, businesses are now able to quickly build foundational models using a vast amount of unlabeled data. 

# Note:
 Not all generative AI methods are based solely on neural networks. While neural networks, particularly deep learning models, have gained significant attention and success in generative tasks due to their ability to learn complex patterns and structures from data. 

Generative AI or generative artificial intelligence refers to the use of AI to generate high-quality new content, like text, images, music, audio, and videos based on the data they were trained on, using the deep-learning models.

Generative AI encompasses unsupervised and semi-supervised machine learning techniques that empower computers to utilize pre-existing content such as text, audio, video files, images, and code to produce novel content. The primary objective is to generate entirely original artifacts that closely resemble the original contents.

Generative artificial intelligence  is artificial intelligence capable of generating text, images or other data using generative models, often in response to prompts. Generative AI models learn the patterns and structure of their input training data and then generate new data that has similar characteristics.

# The goal of Generative AI is to minimize the loss function, the statistical difference between the model’s outputs and the data it was trained on. 

# Some Broad Categories of Generative AI 
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

Attention mechanism:  Attention refers to the ability of a transformer model--deep-learning architecture-- to attend to different parts of another sequence of inputs when making predictions.

The attention mechanism lets transformers encode the meaning of words based on importance of various pieces of input data (words or tokens).

This enables transformers to process all words or tokens in parallel for faster performance, helping drive the growth of increasingly bigger LLMs.

The Transformer model is a new kind of encoder-decoder (VAEs) model that uses self-attention to make sense of language sequences.

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

------------------------------------
# Key concepts of Generative AI
A) GAN - Generative Adversarial Network pits two neural networks against one another in the context of a zero-sum game. GANs are made to create new, synthetic data that closely resembles the distribution of current data. Here, the two neural networks are a generator and a discriminator. The Generator attempts to deceive the Discriminator by creating artificial samples of data (such as an image, audio, etc.). On the other hand, the Discriminator tries to tell the difference between genuine and fraudulent samples.

B) VAE - Variational Auto-Encoders is a subclass of deep generative networks that share the same encoder (inference) and decoder (generative) components as the traditional auto-encoder. It is an autoencoder whose distribution of encodings is regulated during training to guarantee that its latent space has favorable attributes allowing us to produce some new information.

C) LLM - Large Language Models are artificial intelligence systems that are based on transformer architecture; built to comprehend and produce human language. In order to grasp the patterns and laws of language, these models are trained to function by learning the statistical correlations between words and phrases in a huge corpus of text.

D) NLP - Natural Language Processing is a branch of artificial intelligence that integrates computational linguistics with statistical, machine learning, and deep learning models. With these technologies, computers can completely "understand" what is said or written, including the speaker's or writer's intents and feelings, and interpret human language in the form of text or audio data.

E) RNN - Recurrent neural networks are a particular type of artificial neural network that are mostly utilized in NLP and speech recognition. It operates on the idea of preserving a layer's output and using that information to anticipate that layer's output from the input. RNN's Hidden state, which retains some details about a sequence, is its primary and most significant characteristic.
Autoencoders – It is an unsupervised neural network that first learns how to minimize and encode information before teaching itself how to decode the compressed and encoded data and rebuild it into a form that is as close to the original input as practical. By developing the ability to disregard data noise, it minimizes the dimensions of the data.

F) Autoregressive models – In order to increase the likelihood of training data, autoregressive models provide an attainable explicit density model. This makes it simple to estimate the likelihood of data observation and to provide an evaluation measure for the generative model using these approaches. It uses a regression model to estimate the value of the following time step after learning from a large number of timed steps and measurements from earlier activities.

G) Diffusion models - The diffusion model is a type of generative model that learns to recover the data by reversing the noising process after first erasing the training data by adding Gaussian noise one at a time. The Diffusion Model may be used to produce data after training by simply subjecting randomly sampled noise to the mastered denoising procedure, yielding innovative and varied high-resolution pictures that are similar to the initial data.

H) Transformer models - Transformer models are a subset of deep learning models that are frequently used in NLP and other generative AI applications. They are trained to learn the connections between the words in a phrase or line of text. They accomplish this learning by employing a technique known as self-attention, which enables the model to evaluate the relative weights of various words in a sequence according to their context. These models provide the essential benefit of processing input sequences concurrently, outperforming RNNs for many NLP applications.

I) Data augmentation - Modifying or "augmenting" a dataset with new data is known as data augmentation. The incorporation of this supplementary data, which may be anything from photographs to text, helps machine learning algorithms perform better. By making modified replicas of an existing dataset using previously collected data, it artificially expands the training set. The dataset may be slightly modified, or new data points may be produced using deep learning.

J)  Flow-based models - Flow-based models define an invertible transformation between the input and output areas to directly represent the data distribution. They provide effective density estimates in addition to data production. To describe complicated data distributions, they use normalizing fluxes, a series of invertible transformations. These changes make it possible to compute likelihoods and sample data quickly.
DeepDream - With the help of a convolutional neural network, the potent computer vision system DeepDream can identify and improve certain patterns in photos. DeepDream over-interprets and intensifies the patterns it notices in a picture, much like a toddler watching clouds and attempting to make sense of random shapes and formations.

K) Transfer learning - Transfer learning is a machine learning (ML) technique that leverages a trained model created for one job to complete another that is similar but unrelated. Transfer learning accelerates training and lowers the cost of creating a new model from scratch. Computer vision, natural language processing, sentiment analysis, and natural language creation are just a few of the generative AI applications that might benefit from transfer learning.

L) GPT - Generative Pre-trained Transformers are a class of neural network models that make use of the transformer architecture. They are sometimes referred to as GPT. They are all-purpose language models capable of writing code, summarizing text, extracting information from documents, and producing original content. Applications with GPT models may produce text and material (including photos, music, and more) that is human-like and can converse with users.

M) Fine-tuning - A machine learning model is fine-tuned to improve its performance on a particular task or dataset by changing its hyperparameters or pre-trained weights. When compared to the first pre-training, fine-tuning takes far less data and computing effort. Larger models sometimes perform worse than well-tuned models do.
Zero-shot learning - A machine learning paradigm known as "zero-shot learning" uses semantic data or connections between known and new classes, which is frequently represented as attribute vectors or knowledge graphs, to categorize or recognize novel categories or instances without the need for training examples.

N) Hallucination - The primary reason for hallucination is that the LLM employs its internal "knowledge" (what it has been developed on), which is irrelevant to the user inquiry. This results in the LLM producing the incorrect output.

I) Prompt engineering - The secret to endless universes is AI prompt engineering, which employs prompts to acquire the desired outcome from an AI tool. For the text we want the model to produce, the prompt gives context. The prompts we design might be anything from straightforward instructions to intricate texts.

J) Foundation models - A foundation model is a system based on deep learning that has been trained using very sizable data sets downloaded from the internet. Because they contain hundreds of billions of hyperparameters that have been trained using hundreds of terabytes of data, these models can cost millions of dollars to develop.

K) BERT - Bidirectional Encoder Representations from Transformers is a deep learning algorithm created by Google AI Research that makes use of unsupervised learning to better comprehend natural language inquiries. The model learns bidirectional representations of text data using a transformer architecture, which enables it to comprehend the context of words inside a phrase or paragraph.

L) Perceptron - Perceptrons are a component of artificial neural networks that are utilized for a variety of classification applications. Input values (Input nodes), weights and bias, net sum, and an activation function are the four essential factors that make up this single-layer neural network.

















# SOURCES:

Link 1: https://research.ibm.com/blog/what-is-generative-AI

Link 2: https://en.wikipedia.org/wiki/Generative_artificial_intelligence

Link 3: https://www.techopedia.com/definition/34633/generative-ai

Link 4 (see) : https://blog.enterprisedna.co/what-is-langchain-a-beginners-guide-with-examples/

Link 5: https://www.kaggle.com/code/sanjushasuresh/generative-ai-creating-machines-more-human-like


