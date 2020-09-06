import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph import Conv2D, Linear, InstanceNorm
# from Paddle.python.paddle.tensor.math import sum




class ResnetGenerator(fluid.dygraph.Layer):
    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=6, img_size=256, light=False):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.n_blocks = n_blocks
        self.img_size = img_size
        self.light = light

        DownBlock = []
        DownBlock += [ReflectionPad2D(3),
                      Conv2D(input_nc, ngf, filter_size=7, stride=1, padding=0, bias_attr=False),
                      InstanceNorm(ngf),
                      ReLU(inplace=True)]

        # Down-Sampling
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            DownBlock += [ReflectionPad2D(1),
                          Conv2D(ngf * mult, ngf * mult * 2, filter_size=3, stride=2, padding=0, bias_attr=False),
                          InstanceNorm(ngf * mult * 2),
                          ReLU(inplace=True)]

        # Down-Sampling Bottleneck
        mult = 2**n_downsampling
        for i in range(n_blocks):
            DownBlock += [ResnetBlock(ngf * mult, use_bias=False)]

        # Class Activation Map
        self.gap_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.gmp_fc = Linear(ngf * mult, 1, bias_attr=False)
        self.conv1x1 =Conv2D(ngf * mult * 2, ngf * mult, filter_size=1, stride=1, bias_attr=True)
        self.relu = ReLU(True)

        # Gamma, Beta block
        if self.light:
            FC = [Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  ReLU(inplace=True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  ReLU(inplace=True)]
        else:
            FC = [Linear(img_size // mult * img_size // mult * ngf * mult, ngf * mult, bias_attr=False),
                  ReLU(inplace=True),
                  Linear(ngf * mult, ngf * mult, bias_attr=False, act='relu'),
                  ReLU(inplace=True)]
        self.gamma = Linear(ngf * mult, ngf * mult, bias_attr=False)
        self.beta = Linear(ngf * mult, ngf * mult, bias_attr=False)

        # Up-Sampling Bottleneck
        for i in range(n_blocks):
            setattr(self, 'UpBlock1_' + str(i+1), ResnetAdaILNBlock(ngf * mult, use_bias=False))

        # Up-Sampling
        UpBlock2 = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            UpBlock2 += [Upsample(scale=2),
                         ReflectionPad2D(1),
                         Conv2D(ngf * mult, int(ngf * mult / 2), filter_size=3, stride=1, padding=0, bias_attr=False),
                         ILN(int(ngf * mult / 2)),
                         ReLU(inplace=True)]

        UpBlock2 += [ReflectionPad2D(3),
                     Conv2D(ngf, output_nc, filter_size=7, stride=1, padding=0, bias_attr=False, act='tanh')]

        self.DownBlock = fluid.dygraph.Sequential(*DownBlock)
        self.FC = fluid.dygraph.Sequential(*FC)
        self.UpBlock2 = fluid.dygraph.Sequential(*UpBlock2)

    def forward(self, input):
        x = self.DownBlock(input)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap_reshape = fluid.layers.reshape(gap, (x.shape[0], -1))
        gap_logit = self.gap_fc(gap_reshape)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(gap_weight, (1, 256))
        # gap = x * gap_weight.unsqueeze(2).unsqueeze(3)
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3], name=None)
        gap = x * gap_weight

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp_logit_reshape = fluid.layers.reshape(gmp, (x.shape[0], -1))
        gmp_logit = self.gmp_fc(gmp_logit_reshape)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, (1, 256))
        # gmp = x * gmp_weight.unsqueeze(2).unsqueeze(3)
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3], name=None)
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        if self.light:
            x_ = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
            x_ = fluid.layers.reshape(x_, (x_.shape[0], -1))
            x_ = self.FC(x_)
        else:
            x_ = fluid.layers.reshape(x, (x.shape[0], -1))
            x_ = self.FC(x_)
        gamma, beta = self.gamma(x_), self.beta(x_)


        for i in range(self.n_blocks):
            x = getattr(self, 'UpBlock1_' + str(i+1))(x, gamma, beta)
        out = self.UpBlock2(x)

        return out, cam_logit, heatmap


class ResnetBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetBlock, self).__init__()
        conv_block = []
        conv_block += [ReflectionPad2D(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=True),
                       InstanceNorm(dim),
                       ReLU(inplace=True)]

        conv_block += [ReflectionPad2D(1),
                       Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=True),
                       InstanceNorm(dim)
                       ]

        self.conv_block = fluid.dygraph.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


class ResnetAdaILNBlock(fluid.dygraph.Layer):
    def __init__(self, dim, use_bias):
        super(ResnetAdaILNBlock, self).__init__()
        self.pad1 = ReflectionPad2D(1)
        self.conv1 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=True)
        self.norm1 = adaILN(dim)
        self.relu1 = ReLU(inplace=True)

        self.pad2 = ReflectionPad2D(1)
        self.conv2 = Conv2D(dim, dim, filter_size=3, stride=1, padding=0, bias_attr=True)
        self.norm2 = adaILN(dim)

    def forward(self, x, gamma, beta):
        out = self.pad1(x)
        out = self.conv1(out) # x, out
        out = self.norm1(out, gamma, beta)
        out = self.relu1(out)
        out = self.pad2(out)
        out = self.conv2(out)
        out = self.norm2(out, gamma, beta)

        return out + x


class adaILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(adaILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape =(1, num_features, 1, 1), dtype ='float32', default_initializer=fluid.initializer.Constant(0.9))
        # self.rho.data.fill_(0.9)

    def forward(self, input, gamma, beta):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1-self.rho) * out_ln
        gamma_unsqueeze = fluid.layers.unsqueeze(input=gamma, axes=[2, 3], name=None)
        beta_unsqueeze = fluid.layers.unsqueeze(input=beta, axes=[2, 3], name=None)
        out = out * gamma_unsqueeze + beta_unsqueeze

        return out


class ILN(fluid.dygraph.Layer):
    def __init__(self, num_features, eps=1e-5):
        super(ILN, self).__init__()
        self.eps = eps
        self.rho = fluid.layers.create_parameter(shape =(1, num_features, 1, 1), dtype ='float32', default_initializer=fluid.initializer.Constant(0.0))
        self.gamma = fluid.layers.create_parameter(shape =(1, num_features, 1, 1), dtype ='float32', default_initializer=fluid.initializer.Constant(1.0))
        self.beta = fluid.layers.create_parameter(shape =(1, num_features, 1, 1), dtype ='float32', default_initializer=fluid.initializer.Constant(0.0))
        # self.rho.data.fill_(0.0)
        # self.gamma.data.fill_(1.0)
        # self.beta.data.fill_(0.0)

    def forward(self, input):
        in_mean, in_var = fluid.layers.reduce_mean(input, dim=[2, 3], keep_dim=True), var(input, dim=[2, 3], keep_dim=True)
        out_in = (input - in_mean) / fluid.layers.sqrt(in_var + self.eps)
        ln_mean, ln_var = fluid.layers.reduce_mean(input, dim=[1, 2, 3], keep_dim=True), var(input, dim=[1, 2, 3], keep_dim=True)
        out_ln = (input - ln_mean) / fluid.layers.sqrt(ln_var + self.eps)
        out = self.rho * out_in + (1-self.rho) * out_ln
        out = out * self.gamma + self.beta

        return out


class Discriminator(fluid.dygraph.Layer):
    def __init__(self, input_nc, ndf=64, n_layers=5):
        super(Discriminator, self).__init__()
        model = [ReflectionPad2D(1),
                 Spectralnorm(
                 Conv2D(input_nc, ndf, filter_size=4, stride=2, padding=0, bias_attr=True)),
                 leaky_relu(0.2)]

        for i in range(1, n_layers - 2):
            mult = 2 ** (i - 1)
            model += [ReflectionPad2D(1),
                      Spectralnorm(
                      Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=2, padding=0, bias_attr=True)),
                      leaky_relu(0.2)]

        mult = 2 ** (n_layers - 2 - 1)
        model += [ReflectionPad2D(1),
                  Spectralnorm(
                  Conv2D(ndf * mult, ndf * mult * 2, filter_size=4, stride=1, padding=0, bias_attr=True)),
                  leaky_relu(0.2)]

        # Class Activation Map
        mult = 2 ** (n_layers - 2)
        self.gap_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.gmp_fc = Spectralnorm(Linear(ndf * mult, 1, bias_attr=False))
        self.conv1x1 = Conv2D(ndf * mult * 2, ndf * mult, filter_size=1, stride=1, bias_attr=True)
        self.leaky_relu = leaky_relu(0.2)

        self.pad = ReflectionPad2D(1)
        self.conv = Spectralnorm(
            Conv2D(ndf * mult, 1, filter_size=4, stride=1, padding=0, bias_attr=False))

        self.model = fluid.dygraph.Sequential(*model)

    def forward(self, input):
        x = self.model(input)

        gap = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='avg')
        gap_logit_reshape = fluid.layers.reshape(gap, (x.shape[0], -1))
        gap_logit = self.gap_fc(gap_logit_reshape)
        gap_weight = list(self.gap_fc.parameters())[0]
        gap_weight = fluid.layers.reshape(gap_weight, [-1, gap_weight.shape[0]])
        gap_weight = fluid.layers.unsqueeze(input=gap_weight, axes=[2, 3], name=None)
        gap = x * gap_weight

        gmp = fluid.layers.adaptive_pool2d(input=x, pool_size=1, pool_type='max')
        gmp_logit_reshape = fluid.layers.reshape(gmp, (x.shape[0], -1))
        gmp_logit = self.gmp_fc(gmp_logit_reshape)
        gmp_weight = list(self.gmp_fc.parameters())[0]
        gmp_weight = fluid.layers.reshape(gmp_weight, [-1, gmp_weight.shape[0]])
        gmp_weight = fluid.layers.unsqueeze(input=gmp_weight, axes=[2, 3], name=None)
        gmp = x * gmp_weight

        cam_logit = fluid.layers.concat([gap_logit, gmp_logit], 1)
        x = fluid.layers.concat([gap, gmp], 1)
        x = self.leaky_relu(self.conv1x1(x))

        heatmap = fluid.layers.reduce_sum(x, dim=1, keep_dim=True)

        x = self.pad(x)
        out = self.conv(x)

        return out, cam_logit, heatmap


class RhoClipper(object):

    def __init__(self, min, max):
        self.clip_min = min
        self.clip_max = max
        assert min < max

    def __call__(self, module):

        if hasattr(module, 'rho'):
            w = module.rho.data
            w = w.clamp(self.clip_min, self.clip_max)
            module.rho.data = w

class ReflectionPad2D(fluid.Layer):
    def __init__(self, padding):
        super().__init__()
        if isinstance(padding, int):
            self.padding = [padding] * 4
        else:
            self.padding = padding

    def forward(self, x):
        return fluid.layers.pad2d(x, self.padding, mode='reflect')

class Spectralnorm(fluid.dygraph.Layer):

    def __init__(self,
                 layer,
                 dim=0,
                 power_iters=1,
                 eps=1e-12,
                 dtype='float32'):
        super(Spectralnorm, self).__init__()
        self.spectral_norm = fluid.dygraph.SpectralNorm(layer.weight.shape, dim, power_iters, eps, dtype)
        self.dim = dim
        self.power_iters = power_iters
        self.eps = eps
        self.layer = layer
        weight = layer._parameters['weight']
        del layer._parameters['weight']
        self.weight_orig = self.create_parameter(weight.shape, dtype=weight.dtype)
        self.weight_orig.set_value(weight)

    def forward(self, x):
        weight = self.spectral_norm(self.weight_orig)
        self.layer.weight = weight
        out = self.layer(x)
        return out

class ReLU(fluid.dygraph.Layer):
    def __init__(self, inplace=False):
        super(ReLU, self).__init__()
        self._inplace = inplace

    def forward(self, x):
        y = fluid.layers.relu(x)
        return y


class leaky_relu(fluid.Layer):
    def __init__(self, alpha=0.2):
        super().__init__()
        self.leaky_relu = lambda x: fluid.layers.leaky_relu(x, alpha=alpha)

    def forward(self, x):
        return self.leaky_relu(x)


def var(input, dim=None, keep_dim=False, unbiased=True, name=None):
    rank = len(input.shape)
    dims = dim if dim != [] else range(rank)
    dims = [e if e >= 0 else e + rank for e in dims]
    inp_shape = input.shape
    mean = fluid.layers.reduce_mean(input, dim=dim, keep_dim=True, name=name)
    tmp = fluid.layers.reduce_mean((input - mean)**2, dim=dim, keep_dim= keep_dim, name=name)
    if unbiased:
        n = 1
        for i in dims:
            n *= inp_shape[i]
        factor = n / (n - 1.0) if n > 1.0 else 0.0
        tmp *= factor
    return tmp

# 定义上采样模块
class Upsample(fluid.dygraph.Layer):
    def __init__(self, scale=2):
        super(Upsample, self).__init__()
        self.scale = scale

    def forward(self, inputs):
        # get dynamic upsample output shape
        shape_nchw = fluid.layers.shape(inputs)
        shape_hw = fluid.layers.slice(shape_nchw, axes=[0], starts=[2], ends=[4])
        shape_hw.stop_gradient = True
        in_shape = fluid.layers.cast(shape_hw, dtype='int32')
        out_shape = in_shape * self.scale
        out_shape.stop_gradient = True

        # reisze by actual_shape
        out = fluid.layers.resize_nearest(
            input=inputs, scale=self.scale, actual_shape=out_shape)
        return out