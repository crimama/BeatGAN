import torch.nn as nn 
import torch 
from src.Network import Encoder,Decoder
from torch import optim 
import time 
import numpy as np 
from src.metric import evaluate

class Discriminator(nn.Module):

    def __init__(self, opt):
        super(Discriminator, self).__init__()
        model = Encoder(opt.ngpu,opt,1)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features

##
class Generator(nn.Module):

    def __init__(self, opt):
        super(Generator, self).__init__()
        self.encoder1 = Encoder(opt.ngpu,opt,opt.nz)
        self.decoder = Decoder(opt.ngpu,opt)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_x = self.decoder(latent_i)
        return gen_x, latent_i

def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param m:
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        # mod.weight.data.normal_(0.0, 0.02)
        nn.init.xavier_normal_(mod.weight.data)
        # nn.init.kaiming_uniform_(mod.weight.data)

    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)
    elif classname.find('Linear') !=-1 :
        torch.nn.init.xavier_uniform(mod.weight)
        mod.bias.data.fill_(0.01)


class BeatGAN:
    def __init__(self, opt, dataloader, device):
        super(BeatGAN, self).__init__()
        self.dataloader = dataloader
        self.device = device
        self.opt=opt
        self.outf = opt.outf

        self.batchsize = opt.batchsize
        self.nz = opt.nz
        self.niter = opt.niter

        self.G = Generator(opt).to(device)
        self.G.apply(weights_init)

        self.D = Discriminator(opt).to(device)
        self.D.apply(weights_init)


        self.bce_criterion = nn.BCELoss()
        self.mse_criterion=nn.MSELoss()


        self.optimizerD = optim.Adam(self.D.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.G.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


        self.total_steps = 0
        self.cur_epoch=0


        self.input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, self.opt.nc, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = 1
        self.fake_label= 0


        self.out_d_real = None
        self.feat_real = None

        self.fake = None
        self.latent_i = None
        self.out_d_fake = None
        self.feat_fake = None

        self.err_d_real = None
        self.err_d_fake = None
        self.err_d = None

        self.out_g = None
        self.err_g_adv = None
        self.err_g_rec = None
        self.err_g = None
        
    def set_input(self, input):
        self.input.data.resize_(input[0].size()).copy_(input[0])
        self.gt.data.resize_(input[1].size()).copy_(input[1])

        # fixed input for view
        if self.total_steps == self.opt.batchsize:
            self.fixed_input.data.resize_(input[0].size()).copy_(input[0])
        
    def train_epoch(self):

        epoch_start_time = time.time()
        self.G.train()
        self.D.train()
        epoch_iter = 0
        for data in self.dataloader["train"]:
            self.total_steps += self.opt.batchsize
            epoch_iter += 1

            self.set_input(data)
            self.optimize()

            errors = self.get_errors()

            self.train_hist['D_loss'].append(errors["err_d"])
            self.train_hist['G_loss'].append(errors["err_g"])

            if (epoch_iter  % self.opt.print_freq) == 0:

                print("Epoch: [%d] [%4d/%4d] D_loss(R/F): %.6f/%.6f, G_loss: %.6f" %
                      ((self.cur_epoch), (epoch_iter), self.dataloader["train"].dataset.__len__() // self.batchsize,
                       errors["err_d_real"], errors["err_d_fake"], errors["err_g"]))
                # print("err_adv:{}  ,err_rec:{}  ,err_enc:{}".format(errors["err_g_adv"],errors["err_g_rec"],errors["err_g_enc"]))


        self.train_hist['per_epoch_time'].append(time.time() - epoch_start_time)
        with torch.no_grad():
            real_input,fake_output = self.get_generated_x()

            self.visualize_pair_results(self.cur_epoch,
                                        real_input,
                                        fake_output,
                                        is_train=True)
        
    def train(self):
        self.train_hist = {}
        self.train_hist['D_loss'] = []
        self.train_hist['G_loss'] = []
        self.train_hist['per_epoch_time'] = []
        self.train_hist['total_time'] = []




        print("Train model.")
        start_time = time.time()
        best_auc=0
        best_auc_epoch=0
        
        for epoch in range(self.niter):
            self.cur_epoch+=1
            self.train_epoch()
            auc,th,f1=self.validate()
            if auc > best_auc:
                best_auc = auc
                best_auc_epoch=self.cur_epoch
                self.save_weight_GD()



        self.train_hist['total_time'].append(time.time() - start_time)
        print("Avg one epoch time: %.2f, total %d epochs time: %.2f" % (np.mean(self.train_hist['per_epoch_time']),
                                                                        self.niter,
                                                                        self.train_hist['total_time'][0]))

        self.save(self.train_hist)

        self.save_loss(self.train_hist)
        
    def optimize(self):

        self.update_netd()
        self.update_netg()

        # If D loss too low, then re-initialize netD
        if self.err_d.item() < 5e-6:
            self.reinitialize_netd()
            
    def update_netd(self):
        ##

        self.D.zero_grad()
        # --
        # Train with real
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.out_d_real, self.feat_real = self.D(self.input)
        # --
        # Train with fake
        self.label.data.resize_(self.opt.batchsize).fill_(self.fake_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_d_fake, self.feat_fake = self.D(self.fake)
        # --


        self.err_d_real = self.bce_criterion(self.out_d_real, torch.full((self.batchsize,), self.real_label, device=self.device))
        self.err_d_fake = self.bce_criterion(self.out_d_fake, torch.full((self.batchsize,), self.fake_label, device=self.device))


        self.err_d=self.err_d_real+self.err_d_fake
        self.err_d.backward()
        self.optimizerD.step()
        
    def reinitialize_netd(self):
        """ Initialize the weights of netD
        """
        self.D.apply(weights_init)
        print('Reloading d net')
        
    def update_netg(self):
        self.G.zero_grad()
        self.label.data.resize_(self.opt.batchsize).fill_(self.real_label)
        self.fake, self.latent_i = self.G(self.input)
        self.out_g, self.feat_fake = self.D(self.fake)
        _, self.feat_real = self.D(self.input)


        # self.err_g_adv = self.bce_criterion(self.out_g, self.label)   # loss for ce
        self.err_g_adv=self.mse_criterion(self.feat_fake,self.feat_real)  # loss for feature matching
        self.err_g_rec = self.mse_criterion(self.fake, self.input)  # constrain x' to look like x


        self.err_g =  self.err_g_rec + self.err_g_adv * self.opt.w_adv
        self.err_g.backward()
        self.optimizerG.step()
        
        
    def get_generated_x(self):
        fake = self.G(self.fixed_input)[0]

        return  self.fixed_input.cpu().data.numpy(),fake.cpu().data.numpy()
    
    def get_errors(self):

        errors = {'err_d':self.err_d.item(),
                    'err_g': self.err_g.item(),
                    'err_d_real': self.err_d_real.item(),
                    'err_d_fake': self.err_d_fake.item(),
                    'err_g_adv': self.err_g_adv.item(),
                    'err_g_rec': self.err_g_rec.item(),
                  }


        return errors
    
    def validate(self):
        '''
        validate by auc value
        :return: auc
        '''
        y_,y_pred=self.predict(self.dataloader["val"])
        rocprc,rocauc,best_th,best_f1=evaluate(y_,y_pred)
        return rocauc,best_th,best_f1
    
    def predict(self,dataloader_,scale=True):
        with torch.no_grad():

            self.an_scores = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(dataloader_.dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(dataloader_.dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.dis_feat = torch.zeros(size=(len(dataloader_.dataset), self.opt.ndf*16*10), dtype=torch.float32,
                                        device=self.device)


            for i, data in enumerate(dataloader_, 0):

                self.set_input(data)
                self.fake, latent_i = self.G(self.input)


                # error = torch.mean(torch.pow((d_feat.view(self.input.shape[0],-1)-d_gen_feat.view(self.input.shape[0],-1)), 2), dim=1)
                #
                error = torch.mean(
                    torch.pow((self.input.view(self.input.shape[0], -1) - self.fake.view(self.fake.shape[0], -1)), 2),
                    dim=1)

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)


            # Scale error vector between [0, 1]
            if scale:
                self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))

            y_=self.gt_labels.cpu().numpy()
            y_pred=self.an_scores.cpu().numpy()

            return y_,y_pred