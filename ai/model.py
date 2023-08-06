import torch
from torchvision.models import resnet18, ResNet18_Weights

class ScoreModel(torch.nn.Module):
    def __init__(self):
        super(ScoreModel, self).__init__()

        # load a pretrained resnet model and
        # remove its last (classification) layer
        self.resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.resnet = torch.nn.Sequential(*list(self.resnet.children())[:-3])

        # create new regression last layer
        self.pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.linear = torch.nn.Linear(in_features=256, out_features=1, bias=False)


    def forward(self, x):

        x = self.resnet(x)
        x = self.pool(x)

        # remove the singleton dimensions
        x = x.squeeze(3)
        x = x.squeeze(2)

        x = self.linear(x)

        return x
    
if __name__ == "__main__":

    score_model = ScoreModel()
    print(score_model)

    ran_input = torch.randn((5,3,224,224))
    print(f'input shape: {ran_input.shape}')

    output = score_model(ran_input)
    print(f'output shape: {output.shape}')

