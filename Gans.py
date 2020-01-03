import torch
class Generator1(torch.nn.Module):
    def __init__(self):
        super(Generator1, self).__init__()
        
        self.layer1And2 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(150, 1024, kernel_size=4, stride=4, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU()
        )

        self.layerAandB = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
        )

        self.lastLayers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(640, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 7, kernel_size=4, stride=2, padding=1),
            torch.nn.LogSoftmax()
        )

    def forward(self, z, downsampled, d):
        concatenated = torch.cat((z, d), 1)
        outfirst = self.layer1And2(concatenated)
        outsecond = self.layerAandB(downsampled)
        concFirstAndSecond = torch.cat((outfirst, outsecond), 1)
        outfinal = self.lastLayers(concFirstAndSecond)
        return outfinal
class Discriminator1(torch.nn.Module):
    def __init__(self):
        super(Discriminator1, self).__init__()
        
        self.layer1To4 = torch.nn.Sequential(
            torch.nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2)
        )

        self.layerAandB = torch.nn.Sequential(
            torch.nn.Conv2d(4, 64, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
        )

        self.layer5 = torch.nn.Sequential(
            torch.nn.Conv2d(640, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2)
        )
        
        self.lastLaysers = torch.nn.Sequential(
            torch.nn.Conv2d(1074, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(1024, 1, kernel_size=4, stride=4, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU(0.2)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, G1, downsampled, dx4):
        outfirst = self.layer1To4(G1)
        outsecond = self.layerAandB(downsampled)
        concFirstAndSecond = torch.cat((outfirst, outsecond), 1)
        outthird = self.layer5(concFirstAndSecond)
        concThirdAndd = torch.cat((outthird,dx4), 1)
        outFinal = self.lastLaysers(concThirdAndd)
        return self.sigmoid(outFinal)

class Generator2(torch.nn.Module):
    def __init__(self):
        super(Generator2, self).__init__()
        
        self.layer1To3 = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(150, 1024, kernel_size=4, stride=4, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU()
        )

        self.layerAToC = torch.nn.Sequential(
            torch.nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
        )

        self.lastLayers = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(512, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            torch.nn.Tanh()
        )

    def forward(self, z, segMap, d):
        concatenated = torch.cat((z, d), 1)
        outfirst = self.layer1To3(concatenated)
        outsecond = self.layerAToC(segMap)
        concFirstAndSecond = torch.cat((outfirst, outsecond), 1)
        outfinal = self.lastLayers(concFirstAndSecond)

        return outfinal
		
class Discriminator2(torch.nn.Module):
    def __init__(self):
        super(Discriminator2, self).__init__()
        
        self.layer1To3 = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2)
        )

        self.layerAToC = torch.nn.Sequential(
            torch.nn.Conv2d(7, 64, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
        )

        self.layer4And5 = torch.nn.Sequential(
            torch.nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(512),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(512, 1024, kernel_size=4, stride=2, padding=1),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2)
        )
        
        self.lastLaysers = torch.nn.Sequential(
            torch.nn.Conv2d(1074, 1024, kernel_size=1, stride=1, padding=0),
            torch.nn.BatchNorm2d(1024),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(1024, 1, kernel_size=4, stride=4, padding=0),
            torch.nn.BatchNorm2d(1),
            torch.nn.LeakyReLU(0.2)
        )
        
        self.sigmoid = torch.nn.Sigmoid()
        
    def forward(self, G2, segMap, d):
        outfirst = self.layer1To3(G2)
        outsecond = self.layerAToC(segMap)
        concFirstAndSecond = torch.cat((outfirst, outsecond), 1)
        outthird = self.layer4And5(concFirstAndSecond)
        concThirdAndd = torch.cat((outthird,d), 1)
        outFinal = self.lastLaysers(concThirdAndd)

        return self.sigmoid(outFinal)
		
		
def initGans(device):
    generator1 = Generator1().to(device)
    discriminator1 = Discriminator1().to(device)
    generator2 = Generator2().to(device)
    discriminator2 = Discriminator2().to(device)
    return generator1, discriminator1, generator2, discriminator2