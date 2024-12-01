import torch
import torch.nn.functional as F
import torchvision
from model import CNN
from dataset import BuildingDataset
import os

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def test(net, loader, device):
    net.eval() 
    correct = 0

    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum().item()

    accuracy = 100.0 * correct / len(loader.dataset)
    print(f'Test Accuracy: {accuracy:.2f}%', flush=True)
    return accuracy


if __name__ == '__main__':
    IMAGE_HEIGHT = 189
    IMAGE_WIDTH = 252
    resize_factor = 1
    new_h = int(IMAGE_HEIGHT / resize_factor)
    new_w = int(IMAGE_WIDTH / resize_factor)

    normalize = torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resize = torchvision.transforms.Resize(size=(new_h, new_w))
    convert = torchvision.transforms.ConvertImageDtype(torch.float)

    test_transforms = torchvision.transforms.Compose([resize, convert, normalize])

    data_dir = '/gpfs/u/scratch/RNL2/shared/data'
    test_labels_dir = os.path.join(data_dir, 'test_labels.csv')

    test_dataset = BuildingDataset(test_labels_dir, data_dir, transform=test_transforms)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False)

    input_dim = (3, new_h, new_w)
    out_dim = 11
    model_path = 'cnn_buildings_98pct.pth'

    network = CNN(in_dim=input_dim, out_dim=out_dim).to(device)
    network.load_state_dict(torch.load(model_path))

    print("Evaluating on test set...", flush=True)
    test(network, test_loader, device)
