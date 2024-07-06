from dataset import *
from models import *
from utils_train import *
from decoder import *
from model_inputs import data, create_model_inputs, create_dataset_inputs, all_constants

import pickle
from sklearn.model_selection import train_test_split

import torchtext
torchtext.disable_torchtext_deprecation_warning()
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR, CyclicLR
from torch.utils.data import DataLoader
torch.manual_seed(1608)

EPOCH_NUM = 30

tag2in = all_constants()[2]

train_val, test = train_test_split(data, test_size=0.1, random_state=16)
train, valid = train_test_split(train_val, test_size=0.1, random_state=16)

def categorical_accuracy(preds, y, tag2idx, is_crf=False):
    """
    To not calculate accuracy over the <pad>, <start>, <end> tokens
    :return accuracy per batch
    """

    max_preds = preds.argmax(dim = 1) if not is_crf else preds
    non_pad_elements = ~torch.isin(y, torch.tensor([tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>']]))
    correct = torch.sum(max_preds[non_pad_elements] == y[non_pad_elements])
    return correct / y[non_pad_elements].shape[0]

def train_model(model, dataloader, optimizer, criterion, tag2in, schedular):

    epoch_loss = 0
    epoch_acc = 0

    model.train()

    for batch in dataloader:

        optimizer.zero_grad()

        predictions, tags, l = model(*batch)
        # predictions = [batch size, sent len, output dim]
        # tags = [batch size, sent len]

        predictions, tags = predictions.view(-1, predictions.shape[-1]), tags.view(-1)

        # predictions = [sent len * batch size, output dim]
        # tags = [sent len * batch size]

        loss = criterion(predictions, tags)
        acc = categorical_accuracy(predictions, tags, tag2in)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        epoch_acc += acc.item()

    schedular.step()
    lr = schedular.get_last_lr()[0]

    print(f"\tTrain Loss:  {epoch_loss / len(dataloader):.3f}")
    print(f"\tTrain Accuracy:  {epoch_acc / len(dataloader):.3f}")
    print(f"\tLearning Rate: {lr}")
    print('^'*20)

    return epoch_acc / len(dataloader), epoch_loss / len(dataloader), lr

def evaluate_model(model, dataloader, criterion, tag2in):

    epoch_loss = 0
    epoch_acc = 0

    model.eval()

    with torch.no_grad():
        for batch in dataloader:

            predictions, tags, l = model(*batch)

            predictions, tags = predictions.view(-1, predictions.shape[-1]), tags.view(-1)

            loss = criterion(predictions, tags)
            acc = categorical_accuracy(predictions, tags, tag2in)

            epoch_loss += loss.item()
            epoch_acc += acc.item()

    print(f"\t Eval Loss: {epoch_loss / len(dataloader):.3f}")
    print(f"\t Eval Accuracy: {epoch_acc / len(dataloader):.3f}")

    return epoch_acc / len(dataloader), epoch_loss / len(dataloader)

def train_evaluate_models(train, valid, model_type, tag2in, epoch_num=EPOCH_NUM):

    dataset_inputs = create_dataset_inputs(model_type)

    train_data, valid_data = POSDataset(train, *dataset_inputs), POSDataset(valid, *dataset_inputs)

    train_loader, valid_loader = DataLoader(train_data, shuffle=True, batch_size=10), \
                                 DataLoader(valid_data, shuffle=True, batch_size=10)


    model_inputs = create_model_inputs(model_type)
    model = VanillaPOSTagger(*model_inputs) if model_type == 'vanilla' else DualPOSTagger(*model_inputs)
    optimizer = Adam(model.parameters())
    weight = torch.ones(len(tag2in))
    ignore_idx = torch.tensor([tag2in['<pad>'], tag2in['<start>'], tag2in['<end>']])
    weight[ignore_idx] = torch.zeros(3)
    criterion = nn.NLLLoss(weight) if model_type == 'vanilla' else nn.CrossEntropyLoss(weight)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, mode='triangular2')

    best_acc = 0.0
    train_acc, eval_acc = [], []
    train_loss, eval_loss = [], []
    lrs = []

    print(f'Training {model_type} model')

    for epoch in range(epoch_num):
        print(f'For epoch #{epoch + 1}')
        tr_acc, tr_loss, lr = train_model(model, train_loader, optimizer, criterion, tag2in, scheduler)
        ev_acc, ev_loss = evaluate_model(model, valid_loader, criterion, tag2in)

        train_acc.append(tr_acc)
        train_loss.append(tr_loss)
        eval_acc.append(ev_acc)
        eval_loss.append(ev_loss)
        lrs.append(lr)

        if ev_acc > best_acc:
            best_acc = ev_acc
            file_name = f"./saved_files/{model_type}-model.pt"
            torch.save(model.state_dict(), file_name)

    return train_acc, eval_acc, train_loss, eval_loss, lrs

# train_evaluate_models(train, valid, 'vanilla', tag2in)

def train_crf_model(model, dataloader, optimizer, criterion, decoder, tag2in, schedular):

    epoch_loss = 0
    epoch_acc = 0
    model.train()

    for batch in dataloader:

        optimizer.zero_grad()

        outputs = model(*batch)

        loss = criterion(*outputs)

        loss.backward()
        optimizer.step()

        decoded = decoder.decode(outputs[0], outputs[2])
        tags = outputs[3]

        acc = categorical_accuracy(decoded, tags, tag2in, is_crf=True)

        epoch_acc += acc.item()
        epoch_loss += loss.item()

    schedular.step()
    lr = schedular.get_last_lr()[0]

    print(f"\t Train Loss:  {epoch_loss / len(dataloader):.3f}")
    print(f"\t Train Accuracy: {epoch_acc / len(dataloader):.3f}")
    print('^' * 20)


    return epoch_acc / len(dataloader), epoch_loss / len(dataloader), lr

def evaluate_crf_model(model, dataloader, criterion, decoder, tag2in):

    epoch_loss = 0
    epoch_acc = 0
    model.eval()

    with torch.no_grad():
        for batch in dataloader:

            outputs = model(*batch)

            loss = criterion(*outputs)

            decoded = decoder.decode(outputs[0], outputs[2])
            tags = outputs[3]

            acc = categorical_accuracy(decoded, tags, tag2in, is_crf=True)

            epoch_acc += acc.item()
            epoch_loss += loss.item()


    print(f"\t Eval Loss:  {epoch_loss / len(dataloader):.3f}")
    print(f"\t Eval Accuracy: {epoch_acc / len(dataloader):.3f} \n")

    return epoch_acc / len(dataloader), epoch_loss / len(dataloader)

def train_evaluate_crf_model(train, valid, tag2in, model_type='crf', epoch_num=EPOCH_NUM):

    dataset_inputs = create_dataset_inputs(model_type)
    train_dataset, valid_dataset = POSDataset(train, *dataset_inputs), POSDataset(valid, *dataset_inputs)
    train_loader, valid_loader = DataLoader(train_dataset, shuffle=True, batch_size=10), \
                               DataLoader(valid_dataset, shuffle=True, batch_size=10)
    model_inputs = create_model_inputs(model_type)
    model = LSTMTagger(*model_inputs)
    criterion = ViterbiLoss(tag2in)
    optimizer = Adam(model.parameters())
    decoder = ViterbiDecoder(tag2in)
    scheduler = CyclicLR(optimizer, base_lr=0.0001, max_lr=0.001, mode='triangular2')

    best_acc = 0.0
    train_acc, eval_acc = [], []
    train_loss, eval_loss = [], []
    lrs = []

    for i in range(epoch_num):

        print(f'For epoch #{i + 1}')
        tr_acc, tr_loss, lr = train_crf_model(model, train_loader, optimizer, criterion, decoder, tag2in, scheduler)
        ev_acc, ev_loss = evaluate_crf_model(model, valid_loader, criterion, decoder, tag2in)

        train_acc.append(tr_acc)
        train_loss.append(tr_loss)
        eval_acc.append(ev_acc)
        eval_loss.append(ev_loss)
        lrs.append(lr)

        if ev_acc > best_acc:
            best_acc = ev_acc
            file_name = f"./saved_files/{model_type}-model.pt"
            torch.save(model.state_dict(), file_name)

    return train_acc, eval_acc, train_loss, eval_loss, lrs

# train_evaluate_crf_model(train, valid, tag2in)

def train_all_models(train, valid, tag2in, try_num=3):

    results = {'vanilla': None, 'dual': None, 'dual-highway': None, 'crf':None}
    for model_type in results:

        if model_type == 'crf':
            results[model_type] = train_evaluate_crf_model(train, valid, tag2in)
        else:
            results[model_type] = train_evaluate_models(train, valid, model_type, tag2in)

        print(f'Training of {model_type} is done')
        print('*'*50)

    path = f"./saved_files/results_dictionary-{try_num}.pkl"
    with open(path, 'wb') as f:
        pickle.dump(results, f)

    return results

results = train_all_models(train, valid, tag2in)



def show_results(model_type, tag2in):

    dataset_inputs = create_dataset_inputs(model_type)
    test_data = POSDataset(valid, *dataset_inputs)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=3)

    model_inputs = create_model_inputs(model_type)
    model = VanillaPOSTagger(*model_inputs) if model_type == 'vanilla' else DualPOSTagger(*model_inputs)
    file_name = f"./saved_files/{model_type}-model.pt"
    model.load_state_dict(torch.load(file_name))
    model.eval()

    with torch.no_grad():

        all_preds = {t: 0 for t in tag2in}
        correct_preds = {t: 0 for t in tag2in}
        incorrect_preds = {t: [] for t in tag2in}
        acc = 0

        for batch in test_loader:

            predictions, tags, lengths = model(*batch)
            preds = predictions.argmax(dim=2)

            batch_acc = categorical_accuracy(predictions.view(-1, predictions.shape[2]), tags.view(-1), tag2in)
            acc += batch_acc.item()

            for i in range(preds.shape[0]):

                original_sen = from_tensor_to_tag(tags[i, :lengths[i]], tag2in)
                pred_sen = from_tensor_to_tag(preds[i, :lengths[i]], tag2in)
                for j in range(len(pred_sen)):
                    all_preds[original_sen[j]] += 1
                    if original_sen[j] == pred_sen[j]:
                        correct_preds[original_sen[j]] += 1
                    else:
                        incorrect_preds[original_sen[j]].append(pred_sen[j])

    print(f'Accuracy {acc / len(test_loader):.3f}')

    incorrect_preds = {t: Counter(incorrect_preds[t]) for t in incorrect_preds}
    for tag in tag2in:
        print(f"Tag {tag} appeared {all_preds[tag]} times in test data")
        print(f"It is corectly recognized {correct_preds[tag]} times")
        for t, num in incorrect_preds[tag].items():
            print(f"Tag {tag} incorecltly recognized as {t} {num} times")
        print('*'*50)

    return all_preds, correct_preds, incorrect_preds

# show_results('vanilla', tag2in)

def show_crf_results(tag2in):

    dataset_inputs = create_dataset_inputs('crf')
    test_data = POSDataset(valid, *dataset_inputs)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=3)

    model_inputs = create_model_inputs('crf')
    model = LSTMTagger(*model_inputs)
    decoder = ViterbiDecoder(tag2in)
    file_name = f"./saved_files/crf-model.pt"
    model.load_state_dict(torch.load(file_name))
    model.eval()

    with torch.no_grad():

        all_preds = {t: 0 for t in tag2in}
        correct_preds = {t: 0 for t in tag2in}
        incorrect_preds = {t: [] for t in tag2in}
        acc = 0

        for batch in test_loader:

            predictions, _, lengths, tags = model(*batch)
            preds = decoder.decode(predictions, lengths).type(torch.int32)

            batch_acc = acc = categorical_accuracy(preds, tags, tag2in, is_crf=True)
            acc += batch_acc.item()

            for i in range(preds.shape[0]):

                original_sen = from_tensor_to_tag(tags[i, :lengths[i]], tag2in)
                pred_sen = from_tensor_to_tag(preds[i, :lengths[i]], tag2in)
                for j in range(len(pred_sen)):
                    all_preds[original_sen[j]] += 1
                    if original_sen[j] == pred_sen[j]:
                        correct_preds[original_sen[j]] += 1
                    else:
                        incorrect_preds[original_sen[j]].append(pred_sen[j])

    print(f'Accuracy {acc / len(test_loader):.3f}')

    incorrect_preds = {t: Counter(incorrect_preds[t]) for t in incorrect_preds}
    for tag in tag2in:
        print(f"Tag {tag} appeared {all_preds[tag]} times in test data")
        print(f"It is corectly recognized {correct_preds[tag]} times")
        for t, num in incorrect_preds[tag].items():
            print(f"Tag {tag} incorecltly recognized as {t} {num} times")
        print('*'*50)

    return all_preds, correct_preds, incorrect_preds

# show_crf_results(tag2in)