import torch
import torch.nn as nn


class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output


class AttnLinear(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.w_att = nn.Linear(input_size, hidden_size)
        self.cls = nn.Linear(hidden_size, output_size)

    def forward(self, features):
        f_att = torch.sigmoid(self.w_att(features))
        v_feature, l_feature = torch.chunk(features, 2, dim=-1)
        output = f_att * v_feature + (1 - f_att) * l_feature
        return self.cls(output)  # (N, T, C)
