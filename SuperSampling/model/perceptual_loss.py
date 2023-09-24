import torch
import torch.nn as nn
import torchvision
    
class PerceptualLoss(nn.Module):
    def __init__(self, *loss_layers: tuple[str, str]):
        super().__init__()
        self.vgg = torchvision.models.vgg.vgg16(pretrained=True)

        if not loss_layers:
            # only relu2_2 was used in the paper
            # https://arxiv.org/pdf/1603.08155.pdf 
            loss_layers = [
                ("8", "relu2_2")
            ]

        self.loss_layers = { layer: 0 for layer, _ in loss_layers }
        for layer, _ in loss_layers: 
            self.loss_layers[layer] += 1 / len(loss_layers)

        self.eval()
        for param in self.parameters():
            param.requires_grad = False
    
    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        assert output.shape == target.shape
        B = output.shape[0]

        batch = torch.cat((output, target), dim=0)
        total_loss = 0
        layers_seen = 0
        for name, module in self.vgg.features._modules.items():
            batch = module(batch)

            if name in self.loss_layers:
                total_loss += self.loss_layers[name] * torch.mean((batch[:B] - batch[B:]) ** 2)
                layers_seen += 1

            if layers_seen == len(self.loss_layers):
                break

        # doesn't appear to be normalized by number of loss layers
        return total_loss