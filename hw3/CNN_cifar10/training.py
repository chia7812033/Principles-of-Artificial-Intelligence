import torch
import sklearn.metrics


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))


def train(tr_set, val_set, model, config, device):

    n_epochs = config['n_epochs']

    optimizer = getattr(torch.optim, config['optimizer'])(
        model.parameters(), **config['optim_hparas'])

    loss_record = {'train': [], 'val': []}
    acc_record = {'train': [], 'val': []}
    epoch = 0
    min_loss = 999
    while epoch < n_epochs:

        early_stop = 0
        train_loss = 0.0
        val_loss = 0.0
        model.train()
        for x, y in tr_set:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            output = model(x)
            loss = model.cal_loss(output, y)
            loss.backward()
            optimizer.step()

            train_loss += loss
            loss_record['train'].append(loss.detach().cpu().item())

        val_loss, val_acc = val(val_set, model, device)
        print('[{:03d}/{:03d}] Loss: {:3.6f} Accuracy: {:3.6f}'.format(
            epoch + 1, n_epochs, val_loss,  val_acc))

        epoch += 1
        val_loss = val_loss.detach().cpu().item()
        loss_record['val'].append(val_loss)
        acc_record['val'].append(val_acc)

        if min_loss > val_loss:
            min_loss = val_loss
            early_stop = 0
        else:
            early_stop += 1
            if early_stop >= config['early_stop']:
                break

    print('Finished training after {} epochs'.format(epoch))
    return loss_record, acc_record


def val(val_set, model, device):
    losses = []
    accs = []
    model.eval()
    for x, y in val_set:
        x, y = x.to(device), y.to(device)
        with torch.no_grad():
            output = model(x)
            loss = model.cal_loss(output, y)
            acc = accuracy(output, y)
        losses.append(loss)
        accs.append(acc)
    epoch_loss = torch.stack(losses).mean()
    epoch_acc = torch.stack(accs).mean()

    return epoch_loss, epoch_acc


def test(tt_set, model, device):
    model.eval()
    preds = []
    ys = []
    for x, y in tt_set:
        x = x.to(device)
        with torch.no_grad():
            pred = model(x)
            pred = pred.round()
            preds.append(pred.detach().cpu())
            ys.append(y.detach().cpu())
    preds = torch.cat(preds, dim=0).numpy()
    ys = torch.cat(ys, dim=0).numpy()
    preds = preds.argmax(axis=1)
    acc = sklearn.metrics.accuracy_score(ys, preds)
    print(acc)
