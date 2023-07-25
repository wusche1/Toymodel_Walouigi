import torch
import torch.nn.functional as F
from torch import nn
import math
import numpy as np
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import pandas as pd

def create_transformer(num_layers, vocab_size=100, max_seq_len=5000):
    d_model = 8   # Scale linearly with num_layers
    dim_feedforward = 16   # Scale linearly with num_layers
    nhead = 4   # Scale logarithmically with num_layers, between 1 and 8
    dropout = 0.1   # Keep dropout constant

    class Transformer(nn.Module):
        def __init__(self):
            super().__init__()

            self.embedding = nn.Embedding(vocab_size, d_model)
            self.dropout1 = nn.Dropout(dropout)  # Dropout after embedding

            self.pos_encoder = nn.Embedding(max_seq_len, d_model)  # Maximum sequence length is set by max_seq_len
            self.bn1 = nn.BatchNorm1d(d_model)  # Batch normalization after position encoding

            self.decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_feedforward, dropout)
            self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers)

            self.dropout2 = nn.Dropout(dropout)  # Dropout after transformer decoder
            self.bn2 = nn.BatchNorm1d(d_model)  # Batch normalization after transformer decoder

            self.output_layer = nn.Linear(d_model, vocab_size)

        def forward(self, src):
            # Determine the device
            device = next(self.parameters()).device

            # Apply the embeddings
            src = self.embedding(src) * torch.sqrt(torch.tensor(self.embedding.embedding_dim, device=device).float())
            src = self.dropout1(src)  # Apply dropout after embedding
            
            # Add position encoding
            positions = torch.arange(len(src), device=device).unsqueeze(1)
            src = src + self.pos_encoder(positions)
            src = self.bn1(src.permute(1, 2, 0)).permute(2, 0, 1)  # Applying BatchNorm1d requires a permute
            
            # Use a memory of zeros for the TransformerDecoder
            memory = torch.zeros((src.size(0), src.size(1), d_model), device=device)
            
            # Pass through the transformer decoder
            output = self.transformer_decoder(src, memory)
            output = self.dropout2(output)  # Apply dropout after transformer decoder
            output = self.bn2(output.permute(1, 2, 0)).permute(2, 0, 1)  # Applying BatchNorm1d requires a permute
            
            # Pass through the output layer
            output = self.output_layer(output)
            return output

    # Instantiate the model
    model = Transformer()
    return model



def create_markov_chain(transition_matrix):
    transition_matrix = np.array(transition_matrix)
    state_space = list(range(len(transition_matrix)))

    def generate_sequence(initial_state, sequence_length):
        sequence = [initial_state]
        for _ in range(sequence_length - 1):
            next_state = np.random.choice(state_space, p=transition_matrix[sequence[-1]])
            sequence.append(next_state)
        return sequence

    return generate_sequence

def create_markov_chain_list_epsilon(epsilon):
    transition_matrix_1 = [
        [0.9, 0.1, 0],
        [0.1, 0.9, 0],
        [0, 0, 1]
    ]

    transition_matrix_2 = [
        [0.1, 0.9, 0],
        [0.1, 0.9, 0],
        [0, 0, 1]
    ]

    transition_matrix_3 = [
        [0.9, 0.1, 0],
        [0.9, 0.1, 0],
        [0, 0, 1]
    ]

    transition_matrix_4 = [
        [0.1, 0.9, 0],
        [0.9, 0.1, 0],
        [0, 0, 1]
    ]

    transition_matrix_5 = [
        [0.1, 0.9 - epsilon, epsilon],
        [0.9, 0.1, 0],
        [0, epsilon, 1 - epsilon]
    ]

    transition_matrices = [transition_matrix_1, transition_matrix_2, transition_matrix_3, transition_matrix_4, transition_matrix_5]
    return transition_matrices
  

from torch.optim.lr_scheduler import StepLR

def train_model(model, train_loader, test_loader, num_epochs=100, lr_decay_step=10, lr_decay_gamma=0.95):
    loss_fn = CrossEntropyLoss()
    optimizer = Adam(model.parameters())
    scheduler = StepLR(optimizer, step_size=lr_decay_step, gamma=lr_decay_gamma)
    
    model.train()
    loss_df = pd.DataFrame(columns=['epoch', 'train_loss', 'test_loss'])

    for epoch in range(num_epochs):
        # Training Loop
        train_loss = 0.0
        for train_inputs, train_targets in train_loader:
            optimizer.zero_grad()
            train_output = model(train_inputs)
            train_output_reshaped = train_output.view(-1, train_output.shape[-1])
            train_target_reshaped = train_targets.contiguous().view(-1)
            loss = loss_fn(train_output_reshaped, train_target_reshaped)
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
        train_loss /= len(train_loader)  # Average over batches

        scheduler.step()  # Adjust the learning rate

        # Test Loop
        test_loss = 0.0
        with torch.no_grad():
            for test_inputs, test_targets in test_loader:
                test_output = model(test_inputs)
                test_output_reshaped = test_output.view(-1, test_output.shape[-1])
                test_target_reshaped = test_targets.contiguous().view(-1)
                loss = loss_fn(test_output_reshaped, test_target_reshaped)
                test_loss += loss.item()
        test_loss /= len(test_loader)  # Average over batches

        loss_df.loc[epoch] = [epoch, train_loss, test_loss]

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Train Loss: {train_loss}, Test Loss: {test_loss}')

    return model, loss_df

def letter_to_number(prompt):
    """
    Convert a string of uppercase letters to a list of numbers.
    """
    return [ord(c) - ord('A') for c in prompt]

def number_to_letter(numbers):
    """
    Convert a list of numbers to a string of uppercase letters.
    """
    return ''.join([chr(i + ord('A')) for i in numbers])

def generate_sequence(model, prompt_numbers, sequence_length=100, temperature=1.0):
    """
    Generate a sequence of numbers from a given prompt of numbers.
    """

    # Prepare the model for evaluation
    model.eval()

    # Create a list to hold the generated sequence
    generated_sequence = list(prompt_numbers)

    with torch.no_grad():  # No need to track the gradients
        for _ in range(sequence_length - len(prompt_numbers)):
            input_sequence = torch.tensor(generated_sequence, dtype=torch.long).unsqueeze(0).to("cuda") # Add batch dimension
            output = model(input_sequence)

            # Get the output for the last token in the sequence
            output_last_token = output[0, -1, :]
            
            # Apply the softmax function to convert the logits to probabilities
            probabilities = F.softmax(output_last_token / temperature, dim=-1)
            
            # Sample a token using the probabilities
            token = torch.multinomial(probabilities, 1).item()

            # Add the token to the generated sequence
            generated_sequence.append(token)

    return generated_sequence
def sequence_probability(sequence, transition_matrix):
    """
    Calculate the probability of a sequence under a Markov chain with a given transition matrix.
    
    Args:
    sequence (list of int): The sequence of states.
    transition_matrix (numpy.ndarray): The transition probability matrix.

    Returns:
    float: The probability of the sequence.
    """
    # Start with a probability of 1
    probability = 1.0

    # Go through the sequence
    for i in range(len(sequence) - 1):
        # Multiply the probability by the transition probability from the current state to the next state
        probability *= transition_matrix[sequence[i]][sequence[i+1]]

    return probability