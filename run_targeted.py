import os

batch_size = 1

mnist_args = {'dataset': 'mnist',
              'data_path': '/home/zengyuyuan/data',
              'N': 50,
              'Test_Image': 100,
              'eps': 0.2,
              'LR': 0.01,
              'Max_Iter': 900,
              'model': 'mlp',
              'APath': 'no',
              'NPath': './np_pretrain/mnist_128/multihead_model.pkl',
              'attack_model_path': './target_model/mnist_mlp/best_dict-30.pth',
              'root_path': './log_mnist_targeted',
              'type': 'R' # R, Z or RZ
              }

cifar_args = {'dataset': 'cifar-10',
              'data_path': '/home/zengyuyuan/data',
              'N': 50,
              'Test_Image': 100,
              'eps': 0.05,
              'LR': 0.01,
              'Max_Iter': 900,
              'APath':'no',
              'model': 'wrn',
              'NPath': './np_pretrain/cifar_128/multihead_model.pkl',
              'attack_model_path': './target_model/cifar_wrn/model.pt',
              'root_path': './log_cifar_targeted',
              'type': 'R' # R, Z or RZ
            }

args = cifar_args
# args = mnist_args
print (args)

gpu = 'CUDA_VISIBLE_DEVICES=2 '

cmd = gpu + 'python ./NPAttack_targeted.py --batch-size %d --dataset %s --data-path %s ' \
            '--N %d --A-path %s --NP-path %s --attack-path %s --I %d ' \
            '--E %.2f --images-num %d --model %s --LR %.3f --root_path %s --type %s' % \
      (batch_size, args['dataset'], args['data_path'], args['N'], args['APath'], args['NPath'],
       args['attack_model_path'], args['Max_Iter'], args['eps'], args['Test_Image'],
       args['model'], args['LR'], args['root_path'], args['type'])
os.system(cmd)

