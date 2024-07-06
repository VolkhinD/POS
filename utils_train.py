# can be deleted
def categorical_accuracy(preds, y, tag2idx, is_crf=False):
    """
    To not calculate accuracy over the <pad>, <start>, <end> tokens
    :return accuracy per batch
    """

    max_preds = preds.argmax(dim = 1) if not is_crf else preds
    non_pad_elements = ~torch.isin(y, torch.tensor([tag2idx['<pad>'], tag2idx['<start>'], tag2idx['<end>']]))
    correct = torch.sum(max_preds[non_pad_elements] == y[non_pad_elements])
    return correct / y[non_pad_elements].shape[0]

def train_model(model, dataloader, optimizer, criterion, tag2in):

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

    print(f"\tTrain Loss:  {epoch_loss / len(dataloader):.3f}")
    print(f"\tTrain Accuracy:  {epoch_acc / len(dataloader):.3f}")
    print('^'*20)

    return epoch_acc / len(dataloader), epoch_loss / len(dataloader)

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

def train_crf_model(model, dataloader, optimizer, criterion, decoder, tag2in):

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


    print(f"\t Train Loss:  {epoch_loss / len(dataloader):.3f}")
    print(f"\t Train Accuracy: {epoch_acc / len(dataloader):.3f}")
    print('^' * 20)


    return epoch_acc / len(dataloader), epoch_loss / len(dataloader)

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

