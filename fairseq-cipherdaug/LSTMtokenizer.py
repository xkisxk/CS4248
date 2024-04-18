class LSTMTokenizer:
    def __init__(self, lstm_model):
        self.lstm_model = lstm_model

    def __call__(self, x):
        return self.encode(x)

    def encode(self, x):
        return self.lstm_model.encode(x)