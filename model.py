import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Dropout, GRU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from data import SessionDataset, SessionDataLoader


def create_model(args):
    inputs = Input(batch_shape=(args.batch_size, 1, args.num_items))
    gru, gru_states = GRU(args.hsz, stateful=True, return_state=True, name='GRU')(inputs)
    dropout = Dropout(args.drop_rate)(gru)
    predictions = Dense(args.num_items, activation='softmax')(dropout)
    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(loss=categorical_crossentropy, optimizer=Adam(args.lr))
    model.summary()
    return model


def train_model(model, args, train, valid):
    train_dataset = SessionDataset(train)
    train_loader = SessionDataLoader(train_dataset, batch_size=args.batch_size)

    for epoch in range(1, args.epochs + 1):
        tr_loader = tqdm(train_loader, total=len(train)//args.batch_size)
        for i, (feat, target, mask) in enumerate(tr_loader):
            reset_hidden_states(model, mask)

            input_ohe = to_categorical(feat, num_classes=train_loader.n_items)
            input_ohe = np.expand_dims(input_ohe, axis=1)
            target_ohe = to_categorical(target, num_classes=train_loader.n_items)

            tr_loss = model.train_on_batch(input_ohe, target_ohe)
            tr_loader.set_postfix(train_loss=tr_loss)

        val_recall, val_mrr = get_metrics(valid, model, args, args.k)

        print(f"\t - Recall@{args.k} epoch {epoch}: {val_recall:3f}")
        print(f"\t - MRR@{args.k}    epoch {epoch}: {val_mrr:3f}\n")


def test_model(model, args, test):
    test_recall, test_mrr = get_metrics(test, model, args, 20)
    print(f"\t - Recall@{args.k}: {test_recall:3f}")
    print(f"\t - MRR@{args.k}: {test_mrr:3f}\n")


def reset_hidden_states(model, mask):
    gru_layer = model.get_layer(name='GRU')
    hidden_states = gru_layer.states[0].numpy()
    for elt in mask:
        hidden_states[elt, :] = 0
    gru_layer.reset_states(states=hidden_states)


def get_metrics(data, model, args, k: int):
    dataset = SessionDataset(data)
    loader = SessionDataLoader(dataset, batch_size=args.batch_size)

    print("Evaluating model...")
    recall_list = []
    mrr_list = []

    for inputs, label, mask in tqdm(loader, total=len(data) // args.batch_size):

        input_ohe = to_categorical(inputs, num_classes=args.num_items)
        input_ohe = np.expand_dims(input_ohe, axis=1)

        pred = model.predict(input_ohe, batch_size=args.batch_size)

        for row_idx in range(inputs.shape[0]):
            pred_row = tf.argsort(pred[row_idx], direction='DESCENDING')
            label_row = label[row_idx]
            recall_list.append(recall_k(pred_row, label_row, k))
            mrr_list.append(mrr_k(pred_row, label_row, k))

    recall = np.mean(recall_list)
    mrr = np.mean(mrr_list)
    return recall, mrr


def mrr_k(pred, truth: int, k: int):
    indexing = np.where(pred[:k] == truth)[0]
    if len(indexing) > 0:
        return 1 / (indexing[0] + 1)
    else:
        return 0


def recall_k(pred, truth: int, k: int) -> int:
    answer = truth in pred[:k]
    return int(answer)
