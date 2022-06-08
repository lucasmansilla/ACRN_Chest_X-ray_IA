import torch

from .networks import AutoEncoder, RegNet
from .modules import SpatialTransformer
from .losses import NormalizedCrossCorrelation, TotalVariation, L2Squared


class AENet:

    def __init__(self, input_size, num_labels, args, mode='train'):
        assert mode in ['train', 'test']

        self.network = AutoEncoder(input_size, num_labels).to(args.device)

        if mode == 'train':
            self.loss_fn = torch.nn.CrossEntropyLoss()

            self.optimizer = torch.optim.Adam(
                self.network.parameters(),
                lr=args.lr
            )

    def train(self, inputs, target):

        # Training mode
        self.network.train()

        output = self.network(inputs)
        loss = self.loss_fn(output, target)

        train_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_loss

    def predict(self, inputs):

        # Testing mode
        self.network.eval()

        with torch.no_grad():
            output = self.network(inputs)

        return output

    def load(self, path):
        self.network.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.network.state_dict(), path)


class ACRegNet:

    def __init__(self, input_size, num_labels, args, mode='train'):
        assert mode in ['train', 'test']

        self.regnet = RegNet(input_size).to(args.device)

        if mode == 'train':
            self.transformer = SpatialTransformer(input_size, mode='nearest').to(args.device)

            self.encoder = AutoEncoder(input_size, num_labels).to(args.device)
            self.encoder.load_state_dict(torch.load(args.autoencoder_file))
            self.encoder = self.encoder.eval().encode

            self.image_loss_fn = NormalizedCrossCorrelation()
            self.flow_loss_fn = TotalVariation()
            self.label_loss_fn = torch.nn.CrossEntropyLoss()
            self.shape_loss_fn = L2Squared()

            self.flow_weight = args.flow_weight
            self.label_weight = args.label_weight
            self.shape_weight = args.shape_weight

            self.optimizer = torch.optim.Adam(
                self.regnet.parameters(),
                lr=args.lr
            )

    def train(self, moving, fixed, moving_label, fixed_label):

        # Training mode
        self.regnet.train()

        out, flow = self.regnet(moving, fixed)
        out_label = self.transformer(moving_label, flow)

        out_label_enc = self.encoder(out_label)
        fixed_label_enc = self.encoder(fixed_label)

        loss = self.image_loss_fn(out, fixed)
        loss += self.flow_weight * self.flow_loss_fn(flow)
        loss += self.label_weight * self.label_loss_fn(out_label, fixed_label)
        loss += self.shape_weight * self.shape_loss_fn(out_label_enc, fixed_label_enc)

        train_loss = loss.item()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return train_loss

    def register(self, moving, fixed):

        # Testing mode
        self.regnet.eval()

        with torch.no_grad():
            output, flow = self.regnet(moving, fixed)

        return output, flow

    def load(self, path):
        self.regnet.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.regnet.state_dict(), path)
