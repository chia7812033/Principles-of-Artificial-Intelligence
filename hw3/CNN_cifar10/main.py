import settings
from dataset import prep_dataloader
from neural_net import CNN
from training import train, test
from plot_result import plot_losses, plot_accuracies

if __name__ == "__main__":

    device = settings.get_device()
    settings.some_settings()

    train_set = prep_dataloader('train', settings.config['batch_size'])
    val_set = prep_dataloader('val', settings.config['batch_size'])
    test_set = prep_dataloader('test', settings.config['batch_size'])

    model = CNN().to(device)

    loss_record, acc_record = train(
        train_set, val_set, model, settings.config, device)

    test(test_set, model, device)

    plot_losses(loss_record)
    plot_accuracies(acc_record)
