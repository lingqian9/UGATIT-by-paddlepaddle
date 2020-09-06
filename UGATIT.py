import time, itertools
from dataset import ImageFolder
from dataset import custom_reader
import paddle.fluid as fluid
# from torchvision import transforms
# from torch.utils.data import DataLoader
from networks import *
from utils import *
from glob import glob

class UGATIT(object) :
    def __init__(self, args):
        self.light = args.light

        if self.light :
            self.model_name = 'UGATIT_light'
        else :
            self.model_name = 'UGATIT'

        self.result_dir = args.result_dir
        self.dataset = args.dataset

        self.iteration = args.iteration
        self.decay_flag = args.decay_flag

        self.batch_size = args.batch_size
        self.print_freq = args.print_freq
        self.save_freq = args.save_freq

        self.lr = args.lr
        self.weight_decay = args.weight_decay
        self.ch = args.ch

        """ Weight """
        self.adv_weight = args.adv_weight
        self.cycle_weight = args.cycle_weight
        self.identity_weight = args.identity_weight
        self.cam_weight = args.cam_weight

        """ Generator """
        self.n_res = args.n_res

        """ Discriminator """
        self.n_dis = args.n_dis

        self.img_size = args.img_size
        self.img_ch = args.img_ch

        self.device = args.device
        self.benchmark_flag = args.benchmark_flag
        self.resume = args.resume

        # if torch.backends.cudnn.enabled and self.benchmark_flag:
            # print('set benchmark !')
            # torch.backends.cudnn.benchmark = True

        print()

        print("##### Information #####")
        print("# light : ", self.light)
        print("# dataset : ", self.dataset)
        print("# batch_size : ", self.batch_size)
        print("# iteration per epoch : ", self.iteration)

        print()

        print("##### Generator #####")
        print("# residual blocks : ", self.n_res)

        print()

        print("##### Discriminator #####")
        print("# discriminator layer : ", self.n_dis)

        print()

        print("##### Weight #####")
        print("# adv_weight : ", self.adv_weight)
        print("# cycle_weight : ", self.cycle_weight)
        print("# identity_weight : ", self.identity_weight)
        print("# cam_weight : ", self.cam_weight)

    ##################################################################################
    # Model
    ##################################################################################

    def build_model(self):
        """ DataLoader """
        # train_transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.Resize((self.img_size + 30, self.img_size+30)),
            # transforms.RandomCrop(self.img_size),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        # test_transform = transforms.Compose([
            # transforms.Resize((self.img_size, self.img_size)),
            # transforms.ToTensor(),
            # transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])

        self.trainA = ImageFolder(os.path.join('dataset', self.dataset, 'trainA'))
        self.trainB = ImageFolder(os.path.join('dataset', self.dataset, 'trainB'))
        self.testA = ImageFolder(os.path.join('dataset', self.dataset, 'testA'))
        self.testB = ImageFolder(os.path.join('dataset', self.dataset, 'testB'))
        self.trainA_loader = self.trainA
        self.trainB_loader = self.trainB
        self.testA_loader = self.testA
        self.testB_loader = self.testB

        """ Define Generator, Discriminator """
        self.genA2B = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.genB2A = ResnetGenerator(input_nc=3, output_nc=3, ngf=self.ch, n_blocks=self.n_res, img_size=self.img_size, light=self.light)
        self.disGA = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disGB = Discriminator(input_nc=3, ndf=self.ch, n_layers=7)
        self.disLA = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)
        self.disLB = Discriminator(input_nc=3, ndf=self.ch, n_layers=5)

        """ Define Loss """
        self.L1_loss = fluid.dygraph.L1Loss()
        self.MSE_loss = fluid.dygraph.MSELoss()
        self.BCE_loss = fluid.dygraph.BCELoss()

        """ Trainer """
        self.G_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, parameter_list=self.genA2B.parameters() + self.genB2A.parameters(), beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay)) # weight_decay=self.weight_decay
        self.D_optim = fluid.optimizer.AdamOptimizer(learning_rate=self.lr, parameter_list=self.disGA.parameters() + self.disGB.parameters() + self.disLA.parameters() + self.disLB.parameters(), beta1=0.5, beta2=0.999, regularization=fluid.regularizer.L2Decay(regularization_coeff=self.weight_decay))# weight_decay=self.weight_decay

        """ Define Rho clipper to constraint the value of rho in AdaILN and ILN"""
        self.Rho_clipper = RhoClipper(0, 1)

    def train(self):
        self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

        start_iter = 1
        if self.resume:
            model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
            if not len(model_list) == 0:
                model_list.sort()
                start_iter = int(model_list[-1].split('_')[-1].split('.')[0])
                self.train_load(os.path.join(self.result_dir, self.dataset, 'model'), start_iter)
                print(" [*] Load SUCCESS")
                # if self.decay_flag and start_iter > (self.iteration // 2):
                    # self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)
                    # self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2)) * (start_iter - self.iteration // 2)

        # training loop
        print('training start !')
        start_time = time.time()
        for step in range(start_iter, self.iteration + 1):
            # if self.decay_flag and step > (self.iteration // 2):
                # self.G_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))
                # self.D_optim.param_groups[0]['lr'] -= (self.lr / (self.iteration // 2))

            try:
                real_A, _ = next(trainA_iter)
            except:
                trainA_iter = self.trainA_loader
                real_A, _ = next(trainA_iter)

            try:
                real_B, _ = next(trainB_iter)
            except:
                trainB_iter = self.trainB_loader
                real_B, _ = next(trainB_iter)

            real_A = fluid.dygraph.to_variable(real_A)
            real_B = fluid.dygraph.to_variable(real_B)

            # real_A, real_B = real_A.to(self.device), real_B.to(self.device)

            # Update D
            # self.D_optim.zero_grad()
            self.D_optim.clear_gradients()

            fake_A2B, _, _ = self.genA2B(real_A)
            fake_B2A, _, _ = self.genB2A(real_B)

            real_GA_logit, real_GA_cam_logit, _ = self.disGA(real_A)
            real_LA_logit, real_LA_cam_logit, _ = self.disLA(real_A)
            real_GB_logit, real_GB_cam_logit, _ = self.disGB(real_B)
            real_LB_logit, real_LB_cam_logit, _ = self.disLB(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            D_ad_loss_GA = self.MSE_loss(real_GA_logit, fluid.layers.ones_like(real_GA_logit)) + self.MSE_loss(fake_GA_logit, fluid.layers.zeros_like(fake_GA_logit))
            D_ad_cam_loss_GA = self.MSE_loss(real_GA_cam_logit, fluid.layers.ones_like(real_GA_cam_logit)) + self.MSE_loss(fake_GA_cam_logit, fluid.layers.zeros_like(fake_GA_cam_logit))
            D_ad_loss_LA = self.MSE_loss(real_LA_logit, fluid.layers.ones_like(real_LA_logit)) + self.MSE_loss(fake_LA_logit, fluid.layers.zeros_like(fake_LA_logit))
            D_ad_cam_loss_LA = self.MSE_loss(real_LA_cam_logit, fluid.layers.ones_like(real_LA_cam_logit)) + self.MSE_loss(fake_LA_cam_logit, fluid.layers.zeros_like(fake_LA_cam_logit))
            D_ad_loss_GB = self.MSE_loss(real_GB_logit, fluid.layers.ones_like(real_GB_logit)) + self.MSE_loss(fake_GB_logit, fluid.layers.zeros_like(fake_GB_logit))
            D_ad_cam_loss_GB = self.MSE_loss(real_GB_cam_logit, fluid.layers.ones_like(real_GB_cam_logit)) + self.MSE_loss(fake_GB_cam_logit, fluid.layers.zeros_like(fake_GB_cam_logit))
            D_ad_loss_LB = self.MSE_loss(real_LB_logit, fluid.layers.ones_like(real_LB_logit)) + self.MSE_loss(fake_LB_logit, fluid.layers.zeros_like(fake_LB_logit))
            D_ad_cam_loss_LB = self.MSE_loss(real_LB_cam_logit, fluid.layers.ones_like(real_LB_cam_logit)) + self.MSE_loss(fake_LB_cam_logit, fluid.layers.zeros_like(fake_LB_cam_logit))

            D_loss_A = self.adv_weight * (D_ad_loss_GA + D_ad_cam_loss_GA + D_ad_loss_LA + D_ad_cam_loss_LA)
            D_loss_B = self.adv_weight * (D_ad_loss_GB + D_ad_cam_loss_GB + D_ad_loss_LB + D_ad_cam_loss_LB)

            Discriminator_loss = D_loss_A + D_loss_B
            Discriminator_loss.backward()
            self.D_optim.minimize(Discriminator_loss)

            # Update G
            # self.G_optim.zero_grad()
            self.G_optim.clear_gradients()

            fake_A2B, fake_A2B_cam_logit, _ = self.genA2B(real_A)
            fake_B2A, fake_B2A_cam_logit, _ = self.genB2A(real_B)

            fake_A2B2A, _, _ = self.genB2A(fake_A2B)
            fake_B2A2B, _, _ = self.genA2B(fake_B2A)

            fake_A2A, fake_A2A_cam_logit, _ = self.genB2A(real_A)
            fake_B2B, fake_B2B_cam_logit, _ = self.genA2B(real_B)

            fake_GA_logit, fake_GA_cam_logit, _ = self.disGA(fake_B2A)
            fake_LA_logit, fake_LA_cam_logit, _ = self.disLA(fake_B2A)
            fake_GB_logit, fake_GB_cam_logit, _ = self.disGB(fake_A2B)
            fake_LB_logit, fake_LB_cam_logit, _ = self.disLB(fake_A2B)

            G_ad_loss_GA = self.MSE_loss(fake_GA_logit, fluid.layers.ones_like(fake_GA_logit))
            G_ad_cam_loss_GA = self.MSE_loss(fake_GA_cam_logit, fluid.layers.ones_like(fake_GA_cam_logit))
            G_ad_loss_LA = self.MSE_loss(fake_LA_logit, fluid.layers.ones_like(fake_LA_logit))
            G_ad_cam_loss_LA = self.MSE_loss(fake_LA_cam_logit, fluid.layers.ones_like(fake_LA_cam_logit))
            G_ad_loss_GB = self.MSE_loss(fake_GB_logit, fluid.layers.ones_like(fake_GB_logit))
            G_ad_cam_loss_GB = self.MSE_loss(fake_GB_cam_logit, fluid.layers.ones_like(fake_GB_cam_logit))
            G_ad_loss_LB = self.MSE_loss(fake_LB_logit, fluid.layers.ones_like(fake_LB_logit))
            G_ad_cam_loss_LB = self.MSE_loss(fake_LB_cam_logit, fluid.layers.ones_like(fake_LB_cam_logit))

            G_recon_loss_A = self.L1_loss(fake_A2B2A, real_A)
            G_recon_loss_B = self.L1_loss(fake_B2A2B, real_B)

            G_identity_loss_A = self.L1_loss(fake_A2A, real_A)
            G_identity_loss_B = self.L1_loss(fake_B2B, real_B)

            G_cam_loss_A = self.BCE_loss(fluid.layers.sigmoid(fake_B2A_cam_logit), fluid.layers.ones_like(fake_B2A_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(fake_A2A_cam_logit), fluid.layers.zeros_like(fake_A2A_cam_logit))
            G_cam_loss_B = self.BCE_loss(fluid.layers.sigmoid(fake_A2B_cam_logit), fluid.layers.ones_like(fake_A2B_cam_logit)) + self.BCE_loss(fluid.layers.sigmoid(fake_B2B_cam_logit), fluid.layers.zeros_like(fake_B2B_cam_logit))

            G_loss_A =  self.adv_weight * (G_ad_loss_GA + G_ad_cam_loss_GA + G_ad_loss_LA + G_ad_cam_loss_LA) + self.cycle_weight * G_recon_loss_A + self.identity_weight * G_identity_loss_A + self.cam_weight * G_cam_loss_A
            G_loss_B = self.adv_weight * (G_ad_loss_GB + G_ad_cam_loss_GB + G_ad_loss_LB + G_ad_cam_loss_LB) + self.cycle_weight * G_recon_loss_B + self.identity_weight * G_identity_loss_B + self.cam_weight * G_cam_loss_B

            Generator_loss = G_loss_A + G_loss_B
            Generator_loss.backward()
            # clip = fluid.clip.GradientClipByGlobalNorm(clip_norm=1.0)
            self.G_optim.minimize(Generator_loss)

            # clip parameter of AdaILN and ILN, applied after optimizer step
            clip_rho(self.genA2B)
            clip_rho(self.genB2A)
            # self.genA2B.apply(self.Rho_clipper)
            # self.genB2A.apply(self.Rho_clipper)

            print("[%5d/%5d] time: %4.4f d_loss: %.8f, g_loss: %.8f" % (step, self.iteration, time.time() - start_time, Discriminator_loss, Generator_loss))
            if step % self.print_freq == 0:
                train_sample_num = 5
                test_sample_num = 5
                A2B = np.zeros((self.img_size * 7, 0, 3))
                B2A = np.zeros((self.img_size * 7, 0, 3))

                self.genA2B.eval(), self.genB2A.eval(), self.disGA.eval(), self.disGB.eval(), self.disLA.eval(), self.disLB.eval()
                for _ in range(train_sample_num):
                    try:
                        real_A, _ = next(trainA_iter)
                    except:
                        trainA_iter = self.trainA_loader
                        real_A, _ = next(trainA_iter)

                    try:
                        real_B, _ = next(trainB_iter)
                    except:
                        trainB_iter = self.trainB_loader
                        real_B, _ = next(trainB_iter)
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    real_A = fluid.dygraph.to_variable(real_A)
                    real_B = fluid.dygraph.to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                for _ in range(test_sample_num):
                    try:
                        real_A, _ = next(testA_iter)
                    except:
                        testA_iter = self.testA_loader
                        real_A, _ = next(testA_iter)

                    try:
                        real_B, _ = next(testB_iter)
                    except:
                        testB_iter = self.testB_loader
                        real_B, _ = next(testB_iter)
                    # real_A, real_B = real_A.to(self.device), real_B.to(self.device)
                    real_A = fluid.dygraph.to_variable(real_A)
                    real_B = fluid.dygraph.to_variable(real_B)

                    fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)
                    fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

                    fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)
                    fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

                    fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)
                    fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

                    A2B = np.concatenate((A2B, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                                               cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                                               cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                                               cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)), 1)

                    B2A = np.concatenate((B2A, np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                                               cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                                               cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                                               cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                                               RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)), 1)

                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'A2B_%07d.png' % step), A2B * 255.0)
                cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'img', 'B2A_%07d.png' % step), B2A * 255.0)
                self.genA2B.train(), self.genB2A.train(), self.disGA.train(), self.disGB.train(), self.disLA.train(), self.disLB.train()

            if step % self.save_freq == 0:
                self.save(os.path.join(self.result_dir, self.dataset, 'model'), step)

            if step % 1000 == 0:
                # params = {}
                # params['genA2B'] = self.genA2B.state_dict()
                fluid.dygraph.save_dygraph(self.genA2B.state_dict(), os.path.join(self.result_dir, self.dataset + '_genA2B_params_latest.pt'))
                # params['genB2A'] = self.genB2A.state_dict()
                fluid.dygraph.save_dygraph(self.genB2A.state_dict(), os.path.join(self.result_dir, self.dataset + '_genB2A_params_latest.pt'))
                # fluid.dygraph.save_dygraph(self.G_optim.state_dict(),os.path.join(self.result_dir, self.dataset ,'genA2B_'+str(step)))
                # params['disGA'] = self.disGA.state_dict()
                fluid.dygraph.save_dygraph(self.disGA.state_dict(), os.path.join(self.result_dir, self.dataset + '_disGA_params_latest.pt'))
                # params['disGB'] = self.disGB.state_dict()
                fluid.dygraph.save_dygraph(self.disGB.state_dict(), os.path.join(self.result_dir, self.dataset + '_disGB_params_latest.pt'))
                # params['disLA'] = self.disLA.state_dict()
                fluid.dygraph.save_dygraph(self.disLA.state_dict(), os.path.join(self.result_dir, self.dataset + '_disLA_params_latest.pt'))
                # params['disLB'] = self.disLB.state_dict()
                fluid.dygraph.save_dygraph(self.disLB.state_dict(), os.path.join(self.result_dir, self.dataset + '_disLB_params_latest.pt'))
                # fluid.dygraph.save_dygraph(self.D_optim.state_dict(), os.path.join(self.result_dir, self.dataset , 'genB2A_'+str(step))) 

    def save(self, dir, step):
        # params = {}
        # params['genA2B'] = self.genA2B.state_dict()
        fluid.dygraph.save_dygraph(self.genA2B.state_dict(), os.path.join(dir, self.dataset + '_genA2B_%07d' % step))
        # params['genB2A'] = self.genB2A.state_dict()
        fluid.dygraph.save_dygraph(self.genB2A.state_dict(), os.path.join(dir, self.dataset + '_genB2A_%07d' % step))
        fluid.dygraph.save_dygraph(self.G_optim.state_dict(),os.path.join(dir, self.dataset + '_genA2B_%07d' % step))
        # params['disGA'] = self.disGA.state_dict()
        fluid.dygraph.save_dygraph(self.disGA.state_dict(), os.path.join(dir, self.dataset + '_disGA_%07d' % step))
        # params['disGB'] = self.disGB.state_dict()
        fluid.dygraph.save_dygraph(self.disGB.state_dict(), os.path.join(dir, self.dataset + '_disGB_%07d' % step))
        # params['disLA'] = self.disLA.state_dict()
        fluid.dygraph.save_dygraph(self.disLA.state_dict(), os.path.join(dir, self.dataset + '_disLA_%07d' % step))
        # params['disLB'] = self.disLB.state_dict()
        fluid.dygraph.save_dygraph(self.disLB.state_dict(), os.path.join(dir, self.dataset + '_disLB_%07d' % step))
        fluid.dygraph.save_dygraph(self.D_optim.state_dict(), os.path.join(dir, self.dataset + '_genB2A_%07d' % step)) 

    def train_load(self, dir, step):
        # params = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_params_%07d.pt' % step))
        # self.genA2B.load_state_dict(params['genA2B'])
        # self.genB2A.load_state_dict(params['genB2A'])
        # self.disGA.load_state_dict(params['disGA'])
        # self.disGB.load_state_dict(params['disGB'])
        # self.disLA.load_state_dict(params['disLA'])
        # self.disLB.load_state_dict(params['disLB'])
        genA2B, g_optim = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_genA2B_%07d' % step))
        self.genA2B.set_dict(genA2B)
        # self.G_optim.set_dict(g_optim)
        genB2A, d_optim = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_genB2A_%07d' % step))
        self.genB2A.load_dict(genB2A)
        disGA, d_optim = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disGA_%07d' % step))
        self.disGA.set_dict(disGA)
        # self.D_optim.set_dict(d_optim)
        disGB, _ = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disGB_%07d' % step))
        self.disGB.set_dict(disGB)
        disLA, _ = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disLA_%07d' % step))
        self.disLA.set_dict(disLA)
        disLB, _ = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_disLB_%07d' % step))
        self.disLB.set_dict(disLB)
    
    def load(self, dir, step):
        genA2B, g_optim = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_genA2B_%07d' % step))
        self.genA2B.set_dict(genA2B)
        genB2A, d_optim = fluid.dygraph.load_dygraph(os.path.join(dir, self.dataset + '_genB2A_%07d' % step))
        self.genB2A.load_dict(genB2A)

        
    def test_load(self):
        a, _ =fluid.load_dygraph('results/YOUR_DATASET_NAME_genA2B_params_latest.pt')
        
        b, _ =fluid.load_dygraph('results/YOUR_DATASET_NAME_genB2A_params_latest.pt')
         
        self.genA2B.load_dict(a)
        self.genB2A.load_dict(b)

    def test(self):
        model_list = glob(os.path.join(self.result_dir, self.dataset, 'model', '*.pdparams'))
        if not len(model_list) == 0:
            model_list.sort()
            iter = int(model_list[-1].split('_')[-1].split('.')[0])
            self.load(os.path.join(self.result_dir, self.dataset, 'model'), iter)
            print(" [*] Load SUCCESS")
        else:
            print(" [*] Load FAILURE")
            return
            
        # self.test_load()
        self.genA2B.eval(), self.genB2A.eval()
        # for n, (real_A, _) in enumerate(self.testA_loader):
        testA_list = os.listdir('dataset//YOUR_DATASET_NAME//testA')
        for n in range(len(testA_list)):
            try:
                real_A, _ = next(testA_iter)
            except:
                testA_iter = self.testA_loader
                real_A, _ = next(testA_iter)
            # real_A = real_A.to(self.device)
            real_A = fluid.dygraph.to_variable(real_A)

            fake_A2B, _, fake_A2B_heatmap = self.genA2B(real_A)

            fake_A2B2A, _, fake_A2B2A_heatmap = self.genB2A(fake_A2B)

            fake_A2A, _, fake_A2A_heatmap = self.genB2A(real_A)

            A2B = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_A[0]))),
                                    cam(tensor2numpy(fake_A2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2A[0]))),
                                    cam(tensor2numpy(fake_A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B[0]))),
                                    cam(tensor2numpy(fake_A2B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_A2B2A[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'A2B_%d.png' % (n + 1)), A2B * 255.0)

        #for n, (real_B, _) in enumerate(self.testB_loader):
        testB_list = os.listdir('dataset//YOUR_DATASET_NAME//testB')
        for n in range(len(testB_list)):
            try:
                real_B, _ = next(testB_iter)
            except:
                testB_iter = self.testB_loader
                real_B, _ = next(testB_iter)
            # real_B = real_B.to(self.device)
            real_B = fluid.dygraph.to_variable(real_B)

            fake_B2A, _, fake_B2A_heatmap = self.genB2A(real_B)

            fake_B2A2B, _, fake_B2A2B_heatmap = self.genA2B(fake_B2A)

            fake_B2B, _, fake_B2B_heatmap = self.genA2B(real_B)

            B2A = np.concatenate((RGB2BGR(tensor2numpy(denorm(real_B[0]))),
                                    cam(tensor2numpy(fake_B2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2B[0]))),
                                    cam(tensor2numpy(fake_B2A_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A[0]))),
                                    cam(tensor2numpy(fake_B2A2B_heatmap[0]), self.img_size),
                                    RGB2BGR(tensor2numpy(denorm(fake_B2A2B[0])))), 0)

            cv2.imwrite(os.path.join(self.result_dir, self.dataset, 'test', 'B2A_%d.png' % (n + 1)), B2A * 255.0)


def clip_rho(net, vmin=0, vmax=1):
    for name, param in net.named_parameters():
        if 'rho' in name:
            param.set_value(fluid.layers.clip(param, vmin, vmax))
