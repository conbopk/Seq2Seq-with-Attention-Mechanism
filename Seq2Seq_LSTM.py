import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchsummary import  summary, torchsummary
from tqdm import tqdm
import datasets
import evaluate
import spacy
import random
import numpy as np
from collections import Counter
from itertools import chain



#Load Dataset
dataset = datasets.load_dataset('bentrevett/multi30k')
train_data, valid_data, test_data = dataset['train'], dataset['validation'], dataset['test']


#Tokenizers
en_nlp = spacy.load('en_core_web_sm')
de_nlp = spacy.load('de_core_news_sm')

# string = 'What a lovely day it is today!'
#
# tokens = [token.text for token in en_nlp.tokenizer(string)]

def tokenize_example(example, en_nlp, de_nlp, max_length, lower, sos_token, eos_token):
    en_tokens = [token.text for token in en_nlp.tokenizer(example['en'])][:max_length]
    de_tokens = [token.text for token in de_nlp.tokenizer(example['de'])][:max_length]
    if lower:
        en_tokens = [token.lower() for token in en_tokens]
        de_tokens = [token.lower() for token in de_tokens]
    en_tokens = [sos_token] + en_tokens + [eos_token]
    de_tokens = [sos_token] + de_tokens + [eos_token]
    return {'en_tokens': en_tokens, 'de_tokens': de_tokens}

max_length = 50
lower = True
sos_token = '<sos>'
eos_token = '<eos>'

fn_kwargs = {
    'en_nlp': en_nlp,
    'de_nlp': de_nlp,
    'max_length': max_length,
    'lower': lower,
    'sos_token': sos_token,
    'eos_token': eos_token
}

train_data = train_data.map(tokenize_example, fn_kwargs=fn_kwargs)
valid_data = valid_data.map(tokenize_example, fn_kwargs=fn_kwargs)
test_data = test_data.map(tokenize_example, fn_kwargs=fn_kwargs)

# print(train_data.shape)
# print(valid_data.shape)
# print(test_data.shape)
# print(train_data[0])


#Build Vocabularies
en_counter = Counter(chain.from_iterable(train_data['en_tokens']))
de_counter = Counter(chain.from_iterable(train_data['de_tokens']))

en_vocab = {word:idx+2 for idx, (word,_) in enumerate(en_counter.most_common())}
de_vocab = {word:idx+2 for idx, (word,_) in enumerate(de_counter.most_common())}
en_vocab['<unk>'], en_vocab['<pad>'] = 0, 1
de_vocab['<unk>'], de_vocab['<pad>'] = 0, 1


#Class Dataset for dataloader
class TranslationDataset(Dataset):
    def __init__(self, data, en_vocab, de_vocab, max_length):
        self.data = data
        self.en_vocab = en_vocab
        self.de_vocab = de_vocab
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        en_tokens = self.data[idx]['en_tokens']
        de_tokens = self.data[idx]['de_tokens']

        en_indices = [self.en_vocab.get(token, self.en_vocab['<unk>']) for token in en_tokens]
        de_indices = [self.de_vocab.get(token, self.de_vocab['<unk>']) for token in de_tokens]

        #Padding
        if len(en_indices) < self.max_length:
            en_indices += [self.en_vocab['<pad>']] * (self.max_length - len(en_indices))
        else:
            en_indices = en_indices[:max_length]

        if len(de_indices) < self.max_length:
            de_indices += [self.de_vocab['<pad>']] * (self.max_length - len(de_indices))
        else:
            de_indices = de_indices[:max_length]

        return torch.tensor(en_indices), torch.tensor(de_indices)

#Dataloader
batch_size = 64
max_length = 50

train_dataset = TranslationDataset(train_data, en_vocab, de_vocab, max_length)
valid_dataset = TranslationDataset(valid_data, en_vocab, de_vocab, max_length)
test_dataset = TranslationDataset(test_data, en_vocab, de_vocab, max_length)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


#Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, num_layers, p):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = nn.Dropout(p)

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, bidirectional=True, dropout=p)

        self.fc_hidden = nn.Linear(hidden_dim*2, hidden_dim)
        self.cell = nn.Linear(hidden_dim*2, hidden_dim)

    def forward(self, x):
        #x shape: (seq_length, batch_size)

        embedding = self.dropout(self.embedding(x))

        encoder_states, (hidden, cell) = self.rnn(embedding)
        #outputs shape(encoder_states): (seq_length, batch_size, hidden_dim*2)

        #hidden shape: (2, batch_size, hidden_dim)
        hidden = self.fc_hidden(torch.cat((hidden[0:1], hidden[1:2]), dim=2))
        cell = self.fc_hidden(torch.cat((cell[0:1], cell[1:2]), dim=2))

        return encoder_states, hidden, cell

#Decoder
class Decoder(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers, p):
        super().__init__()
        self.dropout = nn.Dropout(p)
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.embedding = nn.Embedding(input_dim, embedding_dim)
        self.rnn = nn.LSTM(hidden_dim*2+embedding_dim, hidden_dim, num_layers, dropout=p)

        self.energy = nn.Linear(hidden_dim*3, 1)
        self.softmax = nn.Softmax(0)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, encoder_states, hidden, cell):
        x = x.unsqueeze(0)

        embedding = self.dropout(self.embedding(x))

        sequence_length = encoder_states.shape[0]
        h_reshape = hidden.repeat(sequence_length,1,1)

        energy = self.relu(self.energy(torch.cat((h_reshape, encoder_states), dim=2)))
        attention = self.softmax(energy)
        #(seq_length, batch_size, 1)

        attention = attention.permute(1,2,0)
        #(batch_size, 1, seq_length)
        encoder_states = encoder_states.permute(1,0,2)
        # (batch_size, seq_length, hidden_dim*2)

        #(batch_size, 1, hidden_dim*2) --> (1, batch_size, hidden_dim*2)
        context_vector = torch.bmm(attention, encoder_states).permute(1,0,2)

        rnn_input = torch.cat((context_vector, embedding), dim=2)
        outputs, (hidden, cell) =self.rnn(rnn_input, (hidden, cell))
        #outputs shape: (1, batch_size, hidden_dim)

        predictions = self.fc(outputs)
        #predictions shape: (1, batch_size, length_target_vocab)--> (batch_size, length_target_vocab)
        predictions = predictions.squeeze(0)

        return predictions, hidden, cell


#Seq2Seq model
class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, source, target, teacher_force_ratio=0.5):
        batch_size = source.shape[1]
        target_len = target.shape[0]
        target_vocab_size = len(de_vocab)

        outputs = torch.zeros(target_len, batch_size, target_vocab_size).to(device)

        encoder_states, hidden, cell = self.encoder(source)

        x = target[0]  #take the first token <sos>

        for t in range(target_len):
            output, hidden, cell = self.decoder(x, encoder_states, hidden, cell)

            outputs[t] = output

            best_guess = output.argmax(1)

            x = target[t] if random.random() < teacher_force_ratio else best_guess

        return outputs


#Define Model
input_dim_encoder = len(en_vocab)
input_dim_decoder = len(de_vocab)
output_dim = len(de_vocab)
encoder_embedding_dim = 300
decoder_embedding_dim = 300
hidden_dim = 1024
dropout = 0.5

num_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


encoder = Encoder(input_dim=input_dim_encoder, embedding_dim=encoder_embedding_dim,hidden_dim=hidden_dim,num_layers=num_layers, p=dropout).to(device)
decoder = Decoder(input_dim=input_dim_decoder, embedding_dim=encoder_embedding_dim,hidden_dim=hidden_dim,output_dim=output_dim,num_layers=num_layers, p=dropout).to(device)
model = Seq2Seq(encoder=encoder, decoder=decoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss(ignore_index=de_vocab['<pad>'])

summary(model)

n_epochs = 100
clip = 1.0

#Training model
def train_and_evaluate(model, train_iterator, valid_iterator, optimizer, criterion, clip, n_epochs):
    for epoch in range(n_epochs):

        #Training
        model.train()
        train_loss = 0.0

        for src, tg in tqdm(train_iterator,desc=f'Epoch {epoch+1:02}/{n_epochs}',colour='blue'):
            src, tg = src.to(device), tg.to(device)

            optimizer.zero_grad()
            output = model(src,tg)

            #remove token <sos>
            output_dim = output.shape[-1]
            output = output[1:].view(-1, output_dim)
            tg = tg[1:].view(-1)

            loss = criterion(output, tg)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_iterator)
        print(f'Train Loss: {avg_train_loss:.4f}')

        #Validation model
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for src, tg in valid_iterator:
                src, tg = src.to(device), tg.to(device)
                output = model(src, tg, 0)  #tắt teacher forcing

                output_dim = output.shape[-1]
                output = output[1:].view(-1, output_dim)
                tg = tg[1:].view(-1)

                loss = criterion(output, tg)
                val_loss += loss.item()

                predictions = output.argmax(1)
                mask = tg != de_vocab['<pad>']

                correct += (predictions[mask] == tg[mask]).sum().item()
                total += mask.sum().item()

        avg_val_loss = val_loss/len(valid_iterator)
        accuracy = correct / total
        print(f'Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.4f}')



train_and_evaluate(model, train_loader, valid_loader, optimizer, criterion, clip, n_epochs)





