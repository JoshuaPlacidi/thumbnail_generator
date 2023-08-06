import torch
import wandb
import os
import argparse
from tqdm import tqdm
from data import ThumbnailScoreDataloaders
from model import ScoreModel


def train(
        model,
        train_dl,
        test_dl,
        num_epochs,
        lr,
        log,
    ):

    # loss optimisation
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    crit = torch.nn.MSELoss()

    # initialise evaluation variables
    best_test_loss = 1e9
    train_loss = 0

    if log:
        # initialise the weights and baises logger
        run = wandb.init(
            project='tray_health_classification',
            config={
                'initial_learning_rate': lr,
                'num_epochs': num_epochs,
            }
        )

        # create folder to store model outputs, name the folder after the wand run name
        folder_path = f'training_results/{run.name}'
        os.mkdir(folder_path)

    epoch_pb = tqdm(range(1, num_epochs+1), desc='Epoch')
    for epoch in epoch_pb:
        model = model.train()
        for iteration, batch in enumerate(train_dl):

            # forward
            thumbnail, score = batch
            pred = model(thumbnail)

            # backward
            optimiser.zero_grad()
            loss = crit(pred, score.float())
            print(pred, score, loss)
            loss.backward()
            optimiser.step()

            # tracking
            train_loss += loss.item()

        if log:
            test_loss, test_accuracy = test(model=model, dl=test_dl, crit=crit)

            wandb.log({
                'train_loss': train_loss,
                'test_loss': test_loss,
                'epoch': epoch
            })

            if test_loss < best_test_loss:
                best_test_loss = test_loss
                torch.save(model.state_dict(), f'{folder_path}/model.pt')

        train_loss = 0


def test(model, dl, crit):

    running_loss = 0
    model = model.eval()

    with torch.no_grad():
        for batch in tqdm(dl, desc='Test', position=2, leave=False):

            # forward
            _, video, label = batch
            pred = model(video.to(model.device))

            # loss tracking
            loss += crit(pred, label.to(model.device)).item()

    # calculate mean loss
    loss = running_loss / len(dl)

    return loss

    
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--csv_path', type=str, required=True)
    parser.add_argument('-f', '--images_path', type=str, required=True)
    parser.add_argument('-b','--batch_size', type=int, default=8,
                        help='Number of samples to batch together for training')
    parser.add_argument('-lr','--learning_rate', type=float, default=1e-4,
                            help='Initial learning rate to use during training')
    parser.add_argument('-e','--num_epochs', type=int, default=20,
                            help='Number of epochs to train for')
    parser.add_argument('-log' , action=argparse.BooleanOptionalAction,
                        help='If training data should be logged using weights and baiases')
    
    # extract arguments
    args = parser.parse_args()
    log = args.log == True

    # load data
    train_dl, test_dl = ThumbnailScoreDataloaders(args.csv_path, args.images_path)

    # load model
    model = ScoreModel()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model).to(device)
    model.device = device

    train(
        model,
        train_dl = train_dl,
        test_dl = test_dl,
        num_epochs = args.num_epochs,
        lr = args.learning_rate,
        log = log
    )