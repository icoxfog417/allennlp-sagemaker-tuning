import argparse
import logging
import os
import sys


def set_pythonpath():
    python_version = "python" + str(sys.version_info.major) \
                     + "." + str(sys.version_info.minor)
    venv_dir = "../.venv/lib/{}/site-packages".format(python_version)
    lib_dir = os.path.join(os.path.dirname(__file__), venv_dir)
    project_dir = os.path.join(os.path.dirname(__file__), "../")
    sys.path.append(lib_dir)
    sys.path.append(project_dir)

set_pythonpath()
import torch
import torch.optim as optim
import numpy as np
from allennlp.common.file_utils import cached_path
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import PytorchSeq2SeqWrapper
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer
from allennlp.predictors import SentenceTaggerPredictor
from example.dataset_reader import PosDatasetReader
from example.model import LstmTagger



logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler(sys.stdout))



def train(args):
    _train_data_path = os.path.join(args.data_dir, args.train_file_name)
    _validation_data_path = os.path.join(args.data_dir, args.validation_file_name)
    print(_train_data_path)
    reader = PosDatasetReader()
    train_dataset = reader.read(_train_data_path)
    validation_dataset = reader.read(_validation_data_path)
    vocab = Vocabulary.from_instances(train_dataset + validation_dataset)

    token_embedding = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                embedding_dim=args.embedding_dim)
    word_embeddings = BasicTextFieldEmbedder({"tokens": token_embedding})
    lstm = PytorchSeq2SeqWrapper(torch.nn.LSTM(args.embedding_dim, args.hidden_dim, batch_first=True))
    model = LstmTagger(word_embeddings, lstm, vocab)

    if torch.cuda.is_available():
        cuda_device = 0
        model = model.cuda(cuda_device)
    else:
        cuda_device = -1

    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    iterator = BucketIterator(batch_size=args.batch_size,
                              sorting_keys=[("sentence", "num_tokens")])
    iterator.index_with(vocab)
    trainer = Trainer(model=model,
                      optimizer=optimizer,
                      iterator=iterator,
                      train_dataset=train_dataset,
                      validation_dataset=validation_dataset,
                      patience=10,
                      num_epochs=args.epochs,
                      cuda_device=cuda_device)
    trainer.train()

    predictor = SentenceTaggerPredictor(model, dataset_reader=reader)
    tag_logits = predictor.predict("The dog ate the apple")['tag_logits']
    tag_ids = np.argmax(tag_logits, axis=-1)
    print([model.vocab.get_token_from_index(i, 'labels') for i in tag_ids])

    # Here's how to save the model.
    model_path = os.path.join(args.model_dir, "model.th")
    vocab_path = os.path.join(args.model_dir, "vocabulary")
    with open(model_path, 'wb') as f:
        torch.save(model.state_dict(), f)
    vocab.save_to_files(vocab_path)

    # And here's how to reload the model.
    vocab2 = Vocabulary.from_files(vocab_path)
    model2 = LstmTagger(word_embeddings, lstm, vocab2)
    with open(model_path, 'rb') as f:
        model2.load_state_dict(torch.load(f))
    if cuda_device > -1:
        model2.cuda(cuda_device)

    predictor2 = SentenceTaggerPredictor(model2, dataset_reader=reader)
    tag_logits2 = predictor2.predict("The dog ate the apple")['tag_logits']
    np.testing.assert_array_almost_equal(tag_logits2, tag_logits)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Data and model checkpoints directories
    parser.add_argument("--batch-size", type=int, default=2, metavar="N",
                        help="input batch size for training (default: 2)")
    parser.add_argument("--epochs", type=int, default=1000, metavar="N",
                        help="number of epochs to train (default: 1000)")
    parser.add_argument("--lr", type=float, default=0.1, metavar="LR",
                        help="learning rate (default: 0.1)")
    parser.add_argument("--embedding-dim", type=int, default=6, metavar="N",
                        help="dimension of embedding vector (default: 6)")
    parser.add_argument("--hidden-dim", type=int, default=6, metavar="N",
                        help="dimension of hidden vector (default: 6)")
    parser.add_argument("--train-file-name", type=str, default="training.txt")
    parser.add_argument("--validation-file-name", type=str, default="validation.txt")

    # Container environment
    parser.add_argument("--model-dir", type=str, default=os.getenv("SM_MODEL_DIR", "/tmp"))
    parser.add_argument("--data-dir", type=str, default=os.getenv("SM_CHANNEL_TRAINING", "data"))

    train(parser.parse_args())
