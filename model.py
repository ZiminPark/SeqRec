import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Input, Dense, Dropout, GRU
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

from data import SessionDataset, SessionDataLoader


def create_model(args):
    hsz = args.hsz
    dropout = args.dropout

    inputs = Input(batch_shape=(args.batch_size, 1, args.train_n_items))
    gru, gru_states = GRU(hsz, stateful=True, return_state=True, name='GRU')(inputs)
    drop2 = Dropout(dropout)(gru)
    predictions = Dense(args.train_n_items, activation='softmax')(drop2)
    model = Model(inputs=inputs, outputs=[predictions])
    model.compile(loss=categorical_crossentropy, optimizer=Adam)
    model.summary()
    return model


def train_model(model, args):
    train_dataset = SessionDataset(args.train_data)
    model_to_train = model
    batch_size = args.batch_size

    for epoch in range(1, args.epochs):
        with tqdm(total=args.train_samples_qty) as pbar:
            loader = SessionDataLoader(train_dataset, batch_size=batch_size)
            for feat, target, mask in loader:

                gru_layer = model_to_train.get_layer(name="GRU")
                hidden_states = gru_layer.states[0].numpy()
                for elt in mask:
                    hidden_states[elt, :] = 0
                gru_layer.reset_states(states=hidden_states)

                input_oh = to_categorical(feat, num_classes=loader.n_items)
                input_oh = np.expand_dims(input_oh, axis=1)

                target_oh = to_categorical(target, num_classes=loader.n_items)

                tr_loss = model_to_train.train_on_batch(input_oh, target_oh)

                pbar.set_description("Epoch {0}. Loss: {1:.5f}".format(epoch, tr_loss))
                pbar.update(loader.done_sessions_counter)

        if args.save_weights:
            print("Saving weights...")
            model_to_train.save('./GRU4REC_{}.h5'.format(epoch))

        if args.eval_all_epochs:
            (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)
            print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
            print("\t - MRR@{}    epoch {}: {:5f}\n".format(mrr_k, epoch, mrr))

    if not args.eval_all_epochs:
        (rec, rec_k), (mrr, mrr_k) = get_metrics(model_to_train, args, train_dataset.itemmap)
        print("\t - Recall@{} epoch {}: {:5f}".format(rec_k, epoch, rec))
        print("\t - MRR@{}    epoch {}: {:5f}\n".format(mrr_k, epoch, mrr))


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
