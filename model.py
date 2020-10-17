import numpy as np
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


def train_model(model, args):
    for epoch in range(1, args.epochs + 1):
        with tqdm(total=args.train_samples_qty) as pbar:
            train_dataset = SessionDataset(args.train_data)
            loader = SessionDataLoader(train_dataset, batch_size=args.batch_size)
            for feat, target, mask in loader:
                reset_hidden_states(model, mask)

                input_ohe = to_categorical(feat, num_classes=loader.n_items)
                input_ohe = np.expand_dims(input_ohe, axis=1)
                target_ohe = to_categorical(target, num_classes=loader.n_items)

                tr_loss = model.train_on_batch(input_ohe, target_ohe)

                pbar.set_description(f'Epoch {epoch}. Loss: {tr_loss:.5f}')  # todo update this
                pbar.update(loader.done_sessions_counter)

        (recall, recall_k), (mrr, mrr_k) = get_metrics(model, args, train_dataset.itemmap)
        print(f"\t - Recall@{recall_k} epoch {epoch}: {recall:5f}")
        print(f"\t - MRR@{mrr_k}    epoch {epoch}: {mrr:5f}\n")


def reset_hidden_states(model, mask):
    gru_layer = model.get_layer(name='GRU')
    hidden_states = gru_layer.states[0].numpy()
    for elt in mask:
        hidden_states[elt, :] = 0
    gru_layer.reset_states(states=hidden_states)


def get_metrics(model, args, train_generator_map, recall_k=20, mrr_k=20):
    test_dataset = SessionDataset(args.test_data, itemmap=train_generator_map)
    test_generator = SessionDataLoader(test_dataset, batch_size=args.batch_size)

    n = 0
    rec_sum = 0
    mrr_sum = 0

    print("Evaluating model...")
    for feat, label, mask in test_generator:

        target_oh = to_categorical(label, num_classes=args.train_n_items)
        input_oh = to_categorical(feat, num_classes=args.train_n_items)
        input_oh = np.expand_dims(input_oh, axis=1)

        pred = model.predict(input_oh, batch_size=args.batch_size)

        for row_idx in range(feat.shape[0]):
            pred_row = pred[row_idx]
            label_row = target_oh[row_idx]

            rec_idx = pred_row.argsort()[-recall_k:][::-1]
            mrr_idx = pred_row.argsort()[-mrr_k:][::-1]
            tru_idx = label_row.argsort()[-1:][::-1]

            n += 1

            if tru_idx[0] in rec_idx:
                rec_sum += 1

            if tru_idx[0] in mrr_idx:
                mrr_sum += 1 / int((np.where(mrr_idx == tru_idx[0])[0] + 1))

    recall = rec_sum / n
    mrr = mrr_sum / n
    return (recall, recall_k), (mrr, mrr_k)
