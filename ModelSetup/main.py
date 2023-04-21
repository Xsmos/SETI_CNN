import optuna
import torch.nn as nn
import torch.nn.functional as F
import data as dataCode, architecture
import torch
import torch.optim as optim

fin = '../seti-breakthrough-listen/train_preprocessed/0.h5'
   
def train(log_interval, model, train_loader, optimizer, epoch, device):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data.to(device))
        output = torch.reshape(output, (output.size()[0],))
        loss = F.binary_cross_entropy_with_logits(output, target.to(device))
        loss.backward()
        optimizer.step()
        if batch_idx%log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
    
    return
            
def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data.to(device))
            output=torch.reshape(output, (output.size()[0],))
            test_loss += F.binary_cross_entropy_with_logits(output, target.to(device), reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=0, keepdim=True)  # get the index of the max log-probability
            print(pred)# xia
            print(target.to(device).view_as(pred))# xia
            print(pred.eq(target.to(device).view_as(pred)).sum())# xia
            print(pred.eq(target.to(device).view_as(pred)).sum().item())#xia
            correct += pred.eq(target.to(device).view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    test_accuracy = 100. * correct / len(test_loader.dataset)
    
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    
    return test_accuracy   


def train_seti():

    cfg = { 'device' : "cuda" if torch.cuda.is_available() else "cpu",
          'trainbs' : 64,
          'testbs' : 1000,
          'n_epochs' : 1,
          'seed' : 12345,
          'log_interval' : 100,
          'save_model' : False,
          'lr' : 0.001,
          'momentum': 0.5,
          'optimizer': optim.SGD,
          'activation': F.leaky_relu,
          'dr1': 0.1,
          'dr2': 0.3
    }
    torch.manual_seed(cfg['seed'])
    device = cfg['device']
    criterion = nn.BCEWithLogitsLoss()

    print('preparing data')
    train_loader = dataCode.create_dataset('train', cfg['seed'], fin, batch_size=cfg['trainbs'], shuffle=True)
    # valid_loader = data.create_dataset('valid', cfg['seed'], fin, batch_size=cfg['trainbs'], shuffle=False)
    test_loader = dataCode.create_dataset('test', cfg['seed'], fin, batch_size=cfg['testbs'], shuffle=False)

    model = architecture.Net(cfg['activation'], cfg['dr1'], cfg['dr2']).to(device)
    optimizer = cfg['optimizer'](model.parameters(), lr=cfg['lr'])

    for epoch in range(1, cfg['n_epochs']+1):
        train(cfg['log_interval'], model, train_loader, optimizer, epoch, device)
        test_accuracy = test(model, test_loader, device)
    
    if cfg['save_model']:
        torch.save(model.state_dict(), 'set_cnn.pt')

    return test_accuracy

if __name__ == '__main__':
    train_seti()
