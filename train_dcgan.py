import argparse
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

import chainer
from chainer import cuda
from chainer import functions as F
from chainer import serializers
from chainer import Variable
from chainer import serializers

from model.dcgan import Generator
from model.dcgan import Discriminator


parser = argparse.ArgumentParser(description='')
parser.add_argument('--iteration', '-i', default=1000, type=int)
parser.add_argument('--save_interval', '-si', default=10, type=int)
parser.add_argument('--batchsize', '-b', default=32, type=int)
parser.add_argument('--learning_rate', '-lr', default=0.001, type=float)
parser.add_argument('--image_dir', '-im', default='data/images', type=str)
parser.add_argument('--save_dir', '-s', default='out/save', type=str)
parser.add_argument('--gpu', '-g', default=-1, type=int)
parser.add_argument('--test', '-t', action='store_true')
parser.add_argument('--model_save', '-ms', action='store_true')
args = parser.parse_args()

nz = 100
image_size = (64, 64)
batchsize = args.batchsize

save_dir = Path(args.save_dir)
save_image_dir = save_dir / 'images'
save_model_dir = save_dir / 'model'
save_image_dir.mkdir(parents=True, exist_ok=True)
save_model_dir.mkdir(parents=True, exist_ok=True)

image_paths = list(Path(args.image_dir).glob('*'))
print('train data num : {}'.format(len(image_paths)))

images = np.zeros((19, 1, 64, 64)).astype(np.float32)

for i in range(len(image_paths) - 1):
    image = Image.open(str(image_paths[i])).resize(image_size)
    # image = np.asarray(image).transpose((2, 0, 1))
    image = np.asarray(image)[np.newaxis, :, :]
    images[i] = (image / 127.5) - 1

gen = Generator(nz)
dis = Discriminator(nz)

if args.gpu >= 0:
    cuda.get_device(args.gpu).use()
    gen.to_gpu()
    dis.to_gpu()

xp = cuda.cupy if args.gpu >= 0 else np

o_gen = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
o_dis = chainer.optimizers.Adam(alpha=0.0002, beta1=0.5)
o_gen.setup(gen)
o_dis.setup(dis)
o_gen.add_hook(chainer.optimizer.WeightDecay(0.00001))
o_dis.add_hook(chainer.optimizer.WeightDecay(0.00001))


# zvis = np.matrix(np.identity(nz)).astype(np.float32)
zvis = np.random.uniform(-1, 1, (10, nz)).astype(np.float32)
if args.gpu >= 0:
    zvis = cuda.to_gpu(zvis)

print('train_start')
for i in tqdm(range(args.iteration)):
    perm = np.random.randint(0, 19, batchsize)

    x2 = images[perm]
    if args.gpu >= 0:
        x2 = cuda.to_gpu(x2)

    # train from gen image
    z = Variable(xp.random.uniform(-1, 1, (batchsize, nz)).astype(np.float32))
    x = gen(z)
    yl = dis(x)
    L_gen = F.softmax_cross_entropy(yl, Variable(xp.zeros(batchsize, dtype=np.int32)))
    L_dis = F.softmax_cross_entropy(yl, Variable(xp.ones(batchsize, dtype=np.int32)))

    # train from true image
    x2 = Variable(x2)
    yl2 = dis(x2)
    L_dis += F.softmax_cross_entropy(yl2, Variable(xp.zeros(batchsize, dtype=np.int32)))

    o_gen.zero_grads()
    L_gen.backward()
    o_gen.update()

    o_dis.zero_grads()
    L_dis.backward()
    o_dis.update()

    gen_loss = L_gen.data
    dis_loss = L_dis.data

    if args.gpu >= 0:
        gen_loss = gen_loss.get()
        dis_loss = dis_loss.get()

    if i % args.save_interval == 0:
        plt.rcParams['figure.figsize'] = (16.0, 2.0)
        plt.clf()
        vissize = 100
        z = Variable(zvis)
        x = gen(z, test=True)
        x = x.data
        if args.gpu >= 0:
            x = x.get()
        for i_ in range(len(zvis)):
            tmp = ((x[i_,:,:,:]+ 1) / 2 ).transpose(1, 2, 0)
            tmp = tmp[:,:,0]
            plt.subplot(1, 10, i_+1)
            plt.imshow(tmp)
            plt.gray()
            plt.axis('off')
        plt.savefig(str(save_image_dir / 'vis_{}'.format(i)))

        print('gen_loss {} / dis_loss {}'.format(gen_loss, dis_loss))

        if args.model_save:
            serializers.save_hdf5(str(save_model_dir / 'dis_{}'.format(i)), dis)
            serializers.save_hdf5(str(save_model_dir / 'gen_{}'.format(i)), gen)
            serializers.save_hdf5(str(save_model_dir / 'o_dis_{}'.format(i)), o_dis)
            serializers.save_hdf5(str(save_model_dir / 'o_gen_{}'.format(i)), o_gen)

print('finish')
