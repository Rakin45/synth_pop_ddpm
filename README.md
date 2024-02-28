# synth_pop_ddpm
Synthetic Population Generation Using a Denoising Diffusion Probabilistic Model


The data is structured to represent activity sequences in a 4-dimensional format:

Batch Size (batch): The first dimension represents the batch size, allowing for batch processing of multiple sequences.

Channels (channel=1): The second dimension indicates the number of channels in the data. In this case, it's set to 1, similar to a grayscale image, meaning there's a single feature channel for each sequence. This is an adaptation to treat the sequence data like image data, where each "pixel" in the "image" represents an embedded feature of the sequence at a specific timestep.

Sequence Length (sequence_length): The third dimension corresponds to the sequence length, similar to the height of an image. This dimension captures the temporal aspect of the sequence, with each "row" of this "image" representing a timestep in the sequence.

Embedding Dimension (embedding_dimension): The fourth and final dimension represents the embedding dimension, similar to the width of an image. Each point along this dimension is a feature within the embedding space of a particular timestep.

