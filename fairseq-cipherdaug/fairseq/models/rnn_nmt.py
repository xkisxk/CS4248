import torch
import torch.nn as nn
from fairseq.models import (
    FairseqEncoder,
    FairseqDecoder,
    FairseqIncrementalDecoder,
    register_model,
    register_model_architecture,
)
from fairseq import utils

class Encoder(FairseqEncoder):
    def __init__(self, args, dictionary, embed_dim=128, hidden_dim=128, dropout=0.1,):
        super().__init__(dictionary)
        self.args = args
        
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(p=dropout)
        self.rnn = nn.GRU(embed_dim, hidden_dim)
        
    def forward(self, src_tokens, src_lengths):
        if self.args.left_pad_source:
            # Convert left-padding to right-padding.
            src_tokens = utils.convert_padding_direction(
                src_tokens,
                padding_idx=self.dictionary.pad(),
                left_to_right=True
            )
        embedded = self.dropout(self.embed_tokens(src_tokens))
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, src_lengths.to('cpu'), enforce_sorted=False)
        _outputs, final_hidden = self.rnn(packed_embedded)
        return {
            # this will have shape `(bsz, hidden_dim)`
            'final_hidden': final_hidden.squeeze(0),
        }

    def reorder_encoder_out(self, encoder_out, new_order):
        final_hidden = encoder_out['final_hidden']
        return {
            'final_hidden': final_hidden.index_select(0, new_order),
        }

class Decoder(FairseqIncrementalDecoder):
    def __init__(self, dictionary, encoder_hidden_dim=128, embed_dim=128, hidden_dim=128,dropout=0.1,):
        super().__init__(dictionary)

        # Our decoder will embed the inputs before feeding them to the LSTM.
        self.embed_tokens = nn.Embedding(
            num_embeddings=len(dictionary),
            embedding_dim=embed_dim,
            padding_idx=dictionary.pad(),
        )
        self.dropout = nn.Dropout(dropout)
        self.rnn = nn.GRU(encoder_hidden_dim + embed_dim, hidden_dim)
        #Takes in both context vector and output for both GRU layer and linear layer to "alleviate" information compression
        self.output_projection = nn.Linear(hidden_dim, len(dictionary))
        
    def forward(self, prev_output_tokens, encoder_out, incremental_state=None):
        if incremental_state is not None:
            prev_output_tokens = prev_output_tokens[:, -1:]

        bsz, tgt_len = prev_output_tokens.size()
        final_encoder_hidden = encoder_out['final_hidden']
        x = self.embed_tokens(prev_output_tokens)
        x = self.dropout(x)
        x = torch.cat([x, final_encoder_hidden.unsqueeze(1).expand(bsz, tgt_len, -1)], dim=2,)
        
        initial_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        if initial_state is None:
            # first time initialization, same as the original version
            initial_state = (
                final_encoder_hidden.unsqueeze(0),  # hidden
                torch.zeros_like(final_encoder_hidden).unsqueeze(0),  # cell
            )

        output, _ = self.rnn(
            x.transpose(0, 1),  # convert to shape `(tgt_len, bsz, dim)`
            initial_state,
        )
        x = output.transpose(0, 1)  # convert to shape `(bsz, tgt_len, hidden)`
        x = self.output_projection(x)
        return x, None

    def reorder_incremental_state(self, incremental_state, new_order):
        # Load the cached state.
        prev_state = utils.get_incremental_state(
            self, incremental_state, 'prev_state',
        )

        # Reorder batches according to *new_order*.
        reordered_state = (
            prev_state[0].index_select(1, new_order),  # hidden
            prev_state[1].index_select(1, new_order),  # cell
        )

        # Update the cached state.
        utils.set_incremental_state(
            self, incremental_state, 'prev_state', reordered_state,
        )

from fairseq.models import BaseFairseqModel, register_model

# Note: the register_model "decorator" should immediately precede the
# definition of the Model class.

@register_model('rnn_nmt')
class RNN_NMT(BaseFairseqModel):
    @staticmethod
    def add_args(parser):
        # Models can override this method to add new command-line arguments.
        # Here we'll add some new command-line arguments to configure dropout
        # and the dimensionality of the embeddings and hidden states.
        parser.add_argument(
            '--encoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the encoder embeddings',
        )
        parser.add_argument(
            '--encoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the encoder hidden state',
        )
        parser.add_argument(
            '--encoder-dropout', type=float, default=0.1,
            help='encoder dropout probability',
        )
        parser.add_argument(
            '--decoder-embed-dim', type=int, metavar='N',
            help='dimensionality of the decoder embeddings',
        )
        parser.add_argument(
            '--decoder-hidden-dim', type=int, metavar='N',
            help='dimensionality of the decoder hidden state',
        )
        parser.add_argument(
            '--decoder-dropout', type=float, default=0.1,
            help='decoder dropout probability',
        )
        parser.add_argument('--share-all-embeddings', default=False, action='store_true',
            help='share encoder, decoder and output embeddings'
                    ' (requires shared dictionary and embed dim)')

    @classmethod
    def build_model(cls, args, task):
        encoder = Encoder(
            args=args,
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_dim=args.encoder_hidden_dim,
            dropout=args.encoder_dropout,
        )
        decoder = Decoder(
            dictionary=task.target_dictionary,
            encoder_hidden_dim=args.encoder_hidden_dim,
            embed_dim=args.decoder_embed_dim,
            hidden_dim=args.decoder_hidden_dim,
            dropout=args.decoder_dropout,
        )
        model = RNN_NMT(encoder, decoder)
        print(model)
        return model

@register_model_architecture('rnn_nmt', 'tutorial_rnn_nmt')
def tutorial_rnn_nmt(args):
    # We use ``getattr()`` to prioritize arguments that are explicitly given
    # on the command-line, so that the defaults defined below are only used
    # when no other value has been specified.
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 256)
    args.encoder_hidden_dim = getattr(args, 'encoder_hidden_dim', 256)
    args.decoder_embed_dim = getattr(args, 'decoder_embed_dim', 256)
    args.decoder_hidden_dim = getattr(args, 'decoder_hidden_dim', 256)
    # def __init__(self, rnn, input_vocab):
    #     super(RNN_NMT, self).__init__()

    #     self.rnn = rnn
    #     self.input_vocab = input_vocab

    #     # The RNN module in the tutorial expects one-hot inputs, so we can
    #     # precompute the identity matrix to help convert from indices to
    #     # one-hot vectors. We register it as a buffer so that it is moved to
    #     # the GPU when ``cuda()`` is called.
    #     self.register_buffer('one_hot_inputs', torch.eye(len(input_vocab)))

    # def forward(self, src_tokens, src_lengths):
    #     # The inputs to the ``forward()`` function are determined by the
    #     # Task, and in particular the ``'net_input'`` key in each
    #     # mini-batch. We'll define the Task in the next section, but for
    #     # now just know that *src_tokens* has shape `(batch, src_len)` and
    #     # *src_lengths* has shape `(batch)`.
    #     bsz, max_src_len = src_tokens.size()

    #     # Initialize the RNN hidden state. Compared to the original PyTorch
    #     # tutorial we'll also handle batched inputs and work on the GPU.
    #     hidden = self.rnn.initHidden()
    #     hidden = hidden.repeat(bsz, 1)  # expand for batched inputs
    #     hidden = hidden.to(src_tokens.device)  # move to GPU

    #     for i in range(max_src_len):
    #         # WARNING: The inputs have padding, so we should mask those
    #         # elements here so that padding doesn't affect the results.
    #         # This is left as an exercise for the reader. The padding symbol
    #         # is given by ``self.input_vocab.pad()`` and the unpadded length
    #         # of each input is given by *src_lengths*.

    #         # One-hot encode a batch of input characters.
    #         input = self.one_hot_inputs[src_tokens[:, i].long()]

    #         # Feed the input to our RNN.
    #         output, hidden = self.rnn(input, hidden)

    #     # Return the final output state for making a prediction
    #     return output
