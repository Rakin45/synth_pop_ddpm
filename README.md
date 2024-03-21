# One Hot Encoded is the best performing model!
Synthetic Population Generation Using a Denoising Diffusion Probabilistic Model


The data is structured to represent activity sequences:

Batch Size (batch): The first dimension represents the batch size, allowing for batch processing of multiple sequences.

Sequence Length (sequence_length): The third dimension corresponds to the sequence length, similar to the width of an image. This dimension captures the temporal aspect of the sequence, with each "column" of this "image" representing a timestep in the sequence.

Embedding Dimension (embedding_dimension): The fourth and final dimension represents the embedding dimension, similar to the height of an image. Each point along this dimension is a feature within the embedding space of a particular timestep.

In the dataset there are 39850 individuals with 144 time steps of their day (1440 minutes in a day, each step is 10 minutes), each time step has an associated activity which has a discrete mapping from 0-7 inclusive (8 activities). The goal is to generate sequences of this data.

