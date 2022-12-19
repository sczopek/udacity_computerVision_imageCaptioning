import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()

        # Try ensembling during training
        self.embed_size = embed_size
        self.drop_prob = 0.5
        self.n_layers = num_layers
        self.n_hidden = hidden_size
        self.lr = 0.001
        self.n_steps = 100
        self.vocab_size = vocab_size
        
        # # creating character dictionaries
        # self.chars = tokens
        # self.int2char = dict(enumerate(self.chars))
        # self.char2int = {ch: ii for ii, ch in self.int2char.items()}
        self.fcFeatureProcessing = nn.Linear(embed_size, embed_size)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        ## TODO: define the LSTM
        self.lstm = nn.LSTM(self.embed_size, self.n_hidden, self.n_layers, 
                            dropout=self.drop_prob, batch_first=True)
        
        ## TODO: define a dropout layer
        self.dropout = nn.Dropout(p=self.drop_prob)
        
        ## TODO: define the final, fully-connected output layer
        self.fc = nn.Linear(self.n_hidden, self.vocab_size)
        
        # initialize the weights
        #self.init_weights()
        pass
    
    def forward(self, features, captions):
        batch_size = features.size(0)
        seq_length = captions.size(1)

        # embeds = self.embedding(captions)
        # embeds = torch.cat( (features.unsqueeze(dim=1), embeds), dim=1)
        
        # exclude the <"end"> token
        # there are no more words to predict after the end token
        embeds = self.embedding(captions[:, :-1])
        embeds = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)
        
        out, hidden = self.lstm(embeds)
        out = self.dropout(out)
        out = self.fc(out)
        
        return out
        
        #for word in captions:
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            # out, hidden = self.lstm(word.view(1, 1, -1))
            #print(word)

        # # the first value returned by LSTM is all of the hidden states throughout
        # # the sequence. the second is just the most recent hidden state

        # # Add the extra 2nd dimension
        # inputs = torch.cat(inputs).view(len(inputs), 1, -1)
        # hidden = (torch.randn(1, 1, 3), torch.randn(1, 1, 3))  # clean out hidden state
        # out, hidden = lstm(inputs, hidden)
        #out, hidden =  self.lstm(features)
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "

        # exclude the <"end"> token
        # there are no more words to predict after the end token
        # embeds = self.embedding(captions[:, :-1])
        # embeds = torch.cat((features.unsqueeze(dim=1), embeds), dim=1)

        # embeds = self.embedding(["<start>"])
        # embeds = torch.cat( (inputs.unsqueeze(dim=1), embeds), dim=1)
        
        # as per message on knowledge help board
        # initialize the hidden state and send it to the same device as the inputs
        hidden = (torch.randn(self.n_layers, 1, self.n_hidden).to(inputs.device),
                  torch.randn(self.n_layers, 1, self.n_hidden).to(inputs.device))
        
        output = []
        
        for i in range(0, max_len):
            prelinary_output, hidden = self.lstm(inputs, hidden)
            prelinary_output = self.dropout(prelinary_output)
            prelinary_output = self.fc(prelinary_output)

            word = prelinary_output
            #print(prelinary_output.shape)

            word = word.squeeze(1)
            word  = word.argmax(dim=1)
            output.append(int(word))
            inputs = self.embedding(word.unsqueeze(0))
            
        return output

#        Before executing the next code cell, you must write the sample method in the DecoderRNN class in model.py. This method should accept as input a PyTorch tensor features containing the embedded input features corresponding to a single image.

#It should return as output a Python list output, indicating the predicted sentence. output[i] is a nonnegative integer that identifies the predicted i-th token in the sentence. The correspondence between integers and tokens can be explored by examining either data_loader.dataset.vocab.word2idx (or data_loader.dataset.vocab.idx2word).
