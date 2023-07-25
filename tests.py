import pytest
import torch
import numpy as np
import sys
print(sys.path)
from basic_functions import create_transformer, create_markov_chain  # replace 'your_module' with the name of your module

def test_create_transformer():
    num_layers = 4
    vocab_size = 100
    model = create_transformer(num_layers, vocab_size)

    # Check model is an instance of nn.Module (all PyTorch models should be)
    assert isinstance(model, torch.nn.Module)

    # Check model has correct number of layers
    assert len(model.transformer_encoder.layers) == num_layers

    # Check vocab size
    assert model.embedding.num_embeddings == vocab_size

def test_create_markov_chain():
    # Transition matrix for a 3-state Markov chain
    transition_matrix = [
        [0.1, 0.6, 0.3],
        [0.4, 0.2, 0.4],
        [0.5, 0.2, 0.3]
    ]

    # Create the Markov chain
    generate_sequence = create_markov_chain(transition_matrix)

    # Test generating sequences of different lengths
    for sequence_length in [1, 5, 10, 20]:
        sequence = generate_sequence(0, sequence_length)
        assert len(sequence) == sequence_length  # Check sequence length

        # Check all states are valid
        for state in sequence:
            assert 0 <= state < 3  # Check state is one of 0, 1, 2
