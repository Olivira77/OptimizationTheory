# Domain Adaptation

## Concepts

### Techniques

#### Discrepancy Based

Discrepancy-based methods in domain adaptation focus on minimizing the distributional differences (discrepancy) between the source and target domains. These methods aim to align the feature distributions of the source and target domains to improve the generalization of a model trained on the source domain to the target domain. Here are two prominent discrepancy-based methods:

##### Maximum Mean Discrepancy (MMD):

**Objective:**
- **Purpose:** MMD aims to minimize the statistical discrepancy between the source and target domains by measuring the difference in means of feature representations.
- **Mathematical Formulation:** It quantifies the dissimilarity between the source and target distributions by comparing the means of kernelized representations of the data.

**Steps:**
1. **Feature Extraction:** Extract features from both the source and target domains using a feature extractor (e.g., a neural network).
2. **Kernelized Representation:** Apply a kernel function to the extracted features to map them into a reproducing kernel Hilbert space.
3. **MMD Computation:** Calculate the Maximum Mean Discrepancy between the source and target distributions using the kernelized representations.
4. **Loss Minimization:** Minimize the MMD loss during training to align the distributions.

##### Correlation Alignment:

**Objective:**
- **Purpose:** Correlation Alignment minimizes the correlation distance between the source and target domains in feature space.
- **Mathematical Formulation:** It encourages the alignment of the cross-covariance matrices of source and target features.

**Steps:**
1. **Feature Extraction:** Extract features from both the source and target domains.
2. **Covariance Computation:** Calculate the cross-covariance matrices of the source and target features.
3. **Correlation Alignment Loss:** Minimize the difference between the cross-covariance matrices, encouraging the features to be aligned.

#### Adversarial Based (state-of-the-art)

Adversarial-based methods in domain adaptation leverage adversarial training to align the feature distributions between the source and target domains. These methods introduce an adversarial component, often referred to as a domain discriminator, to encourage the model to generate domain-invariant features. The primary idea is to make the learned representations insensitive to domain shifts, facilitating better generalization to the target domain. Here are two well-known adversarial-based methods:

##### Domain Adversarial Neural Network (DANN):

Domain Adversarial Neural Networks (DANN) is a type of neural network architecture used for domain adaptation, particularly in the context of unsupervised domain adaptation. The goal of domain adaptation is to train a model on a source domain (where labeled data is available) and apply it to a target domain (where labeled data is scarce or unavailable).

1. **Source and Target Domains:**
   - The source domain refers to the domain where labeled data is available for training the model.
   - The target domain is the domain where you want to apply the trained model, but labeled data is limited or non-existent.

2. **Domain Adversarial Training:**
   - DANN introduces a domain discriminator that is trained to distinguish between the source and target domains.
   - Simultaneously, the feature extractor (commonly a neural network) is trained to confuse the domain discriminator by extracting features that are domain-invariant.

3. **Adversarial Loss:**
   - DANN employs an adversarial loss that encourages the feature extractor to generate domain-invariant features.
   - The objective is to minimize the domain discriminator's ability to differentiate between the source and target features.

4. **Overall Loss Function:**
   - The overall loss function is a combination of the task-specific loss (e.g., classification loss for a classification task) and the adversarial loss.
   - Minimizing the task-specific loss ensures that the model performs well on the source domain, while minimizing the adversarial loss promotes domain-invariant features.

DANN aims to learn representations that are not sensitive to variations between the source and target domains, facilitating the model's adaptation to new, unseen data.

##### Adversarial Discriminative Domain Adaptation (ADDA):

**Objective:**
- **Purpose:** ADDA extends DANN by introducing an additional adversarial alignment module to align the source and target feature distributions at a deeper level.
- **Two-Stage Training:** The model is trained in two stages—first, on the source domain, and then, an additional adversarial alignment module is introduced for target domain adaptation.
- **Feature Extractor Transfer:** The learned feature extractor from the source domain is transferred to the target domain adaptation module.

#### Reconstruction Based

Reconstruction-based methods in domain adaptation focus on reconstructing the input data in a way that reduces the domain gap between the source and target domains. These methods leverage the idea that forcing the model to reconstruct the input data from a shared representation can encourage the learning of domain-invariant features. Here are two prominent reconstruction-based methods:

##### Reconstruction-Based Domain Adaptation (ReconDA):

**Objective:**
- **Purpose:** ReconDA introduces a reconstruction loss that encourages the model to reconstruct the input data from the shared features, making the learned representations domain-invariant.
- **Two Loss Components:** The model is trained with two main loss components—an original task-specific loss and a reconstruction loss.
- **Joint Training:** Both losses are combined during training to jointly optimize for the task and domain-invariant representations.

##### Cycle-Consistent Adversarial Networks (CyCADA):

**Objective:**
- **Purpose:** CyCADA combines adversarial training with cycle-consistency to adapt the model across domains.
- **Cycle-Consistency Loss:** It introduces a cycle-consistency loss that enforces the reconstruction of source features after domain adaptation to be close to the original source features.
- **Adversarial Alignment:** Adversarial alignment is used to align the source and target feature distributions.
- **Two-Stage Training:** The model is trained in two stages—first, on the source domain, and then, an additional adaptation stage is introduced.

##### Autoencoder

An autoencoder is a type of artificial neural network used in unsupervised learning. It is designed to learn efficient representations of data, typically for the purpose of dimensionality reduction or feature learning. The network consists of an encoder and a decoder.

1. **Encoder:**
   - The input data is passed through the encoder, which compresses the input into a lower-dimensional representation. This process is often referred to as encoding.
   - The encoder learns to capture the essential features of the input data in a reduced set of dimensions, effectively creating a compressed representation.

2. **Bottleneck Layer:**
   - The compressed representation is stored in a layer called the bottleneck layer. This layer contains fewer neurons or nodes than the input layer, forcing the network to capture the most important features of the input data.

3. **Decoder:**
   - The compressed representation is then passed through the decoder, which aims to reconstruct the original input data from the compressed representation.
   - The decoder's output should closely match the input data, and the network is trained to minimize the reconstruction error.

4. **Objective Function:**
   - The training objective is typically to minimize the difference between the input and the reconstructed output. Mean Squared Error (MSE) is a common choice for this purpose.

Autoencoders have various applications, including:

- **Data Compression:** Creating compact representations of data.
- **Feature Learning:** Discovering meaningful features in an unsupervised manner.
- **Denoising:** Training on noisy data to learn a denoising process.

Both denoising autoencoders and variational autoencoders (VAEs) are extensions of the basic autoencoder architecture, each serving different purposes in the realm of machine learning.

###### Denoising Autoencoder:

**Objective:**

- **Purpose:** Denoising autoencoders are designed to learn robust features by training on corrupted data and reconstructing the clean, original input.
- **Training Process:** The input data is intentionally corrupted by adding noise, and the autoencoder is trained to remove this noise during the reconstruction process.
- **Applications:** Denoising autoencoders are useful in scenarios where the input data is prone to noise or imperfections, such as image denoising or signal processing.

###### Variational Autoencoder (VAE):

**Objective:**
- **Purpose:** VAEs are generative models that learn a probabilistic mapping between the input data and a latent space.
- **Latent Space:** Unlike traditional autoencoders, VAEs learn a probabilistic distribution in the latent space, allowing for the generation of new, meaningful data points.
- **Applications:** VAEs are often used for generating new samples, such as in image generation or creative applications where novel data is desired.

**Key Concepts:**
- **Encoder Output:** Instead of directly outputting a latent representation, the encoder outputs the parameters of a probability distribution (usually Gaussian) in the latent space.
- **Reparameterization Trick:** To enable backpropagation through the sampling process, a reparameterization trick is used, involving sampling from the learned distribution.
- **Generative Nature:** VAEs can generate new samples by sampling from the learned latent space distribution and decoding these samples.

## Pipeline

- **NOTICE:** record both predictions and logits.

### Method1 (not implement)

1. Train a DANN.

### Method 2 (not implement)

1. Collect all train and test data.
2. Train a denoising/variational autoencoder to extract features that robust to domain shift.
3. Training classifiers with the encoding of the train data.
4. Predict on the test data.

