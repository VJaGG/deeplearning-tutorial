'''
@File    : model.py
@Modify Time     @Author    @Version    @Desciption
------------     -------    --------    -----------
2021/8/6 9:36   WuZhiqiang     1.0        None 
'''
from model import *
from common import *
from dataset import *
from torchsummary import summary
import torch.cuda.amp as amp


class AmpNet(EfficientB4):
    @torch.cuda.amp.autocast()
    def forward(self,*args):
        return super(AmpNet, self).forward(*args)


def do_valid(net, valid_loader):
    valid_label = []
    valid_probability = []

    start_timer = timer()
    with torch.no_grad():
        net.eval()
        valid_num = 0
        for t, batch in enumerate(valid_loader):
            index = batch['index']
            label = batch['label'].cuda()
            image = batch['image'].cuda()
            logit = net(image)
            p = torch.softmax(logit, dim=1)
            valid_num += len(index)

            valid_label.append(label.data.cpu().numpy())
            valid_probability.append(p.data.cpu().numpy())

            print('\r %8d / %d  %s' % (valid_num,
                                       len(valid_loader.dataset),
                                       time_to_str(timer() - start_timer, 'sec')),
                                       end='',
                                       flush=True)
        assert (valid_num == len(valid_loader.dataset))
    
    label = np.concatenate(valid_label)
    probability = np.concatenate(valid_probability)
    predict = probability.argmax(-1)

    acc = np_metric_accuracy(predict, label)
    loss = np_loss_cross_entropy(probability, label)
    return [loss, acc]


def run_train(config):
    # config
    initial_checkpoint = config['initial_checkpoint']
    weight_decay = config['weight_decay']
    start_lr = config['start_lr']
    fold = config['fold']
    batch_size = config['batch_size']
    out_dir = config['out_dir'] + 'fold-%d' % fold

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for f in ['checkpoint', 'valid']:
        os.makedirs(out_dir + '/' + f, exist_ok=True)
    log = Logger()
    log.open(out_dir + '/log.train.txt', mode='a')
    log.write('\n--- [START %s] %s\n\n' % (IDENTIFIER, '-' * 64))
    log.write('\t%s\n' % COMMON_STRING)
    log.write('\t__file__ = %s\n' % __file__)
    log.write('\tout_dir  = %s\n' % out_dir)
    log.write('\n')
    # log.write(config)

    #-------------dataset-----------------------
    df_train, df_valid = make_fold('train-%d' % fold)
    train_dataset = Food101(df_train, augment=train_augment)
    valid_dataset = Food101(df_valid, )

    train_loader = data.DataLoader(
        train_dataset,
        sampler     = RandomSampler(train_dataset),
        batch_size  = batch_size,
        num_workers = 4,
        collate_fn  = collate_fn,
    )

    valid_loader = data.DataLoader(
        valid_dataset,
        shuffle     = False,
        batch_size  = batch_size,
        num_workers = 4,
        collate_fn  = collate_fn,
    )

    log.write('root = %s\n' % str(root))
    log.write('train_dataset : \n%s\n' % (train_dataset))
    log.write('valid_dataset : \n%s\n' % (valid_dataset))
    log.write('\n')


    ## net ----------------------------------------
    log.write('** net setting **\n')
    if config['is_mixed_precision']:
        scaler = amp.GradScaler()
        net = AmpNet()
    else:
        net = EfficientB4()
    net.to(device)


    if initial_checkpoint is not None:
        f = torch.load(initial_checkpoint, map_location=lambda storage, loc: storage)
        start_iteration = f['iteration']
        start_epoch = f['epoch']
        state_dict  = f['state_dict']
        net.load_state_dict(state_dict,strict=False)  #True
    else:
        start_iteration = 0
        start_epoch = 0

    log.write('net=%s\n'%(type(net)))
    log.write('\tinitial_checkpoint = %s\n' % initial_checkpoint)
    log.write('\n')

    if 1:  # freeze
        for p in net.backbone.parameters():
            p.requires_grad = False

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()),
                           lr=start_lr,
                           weight_decay=weight_decay)

    num_iteration = 80000 * 1000
    iter_log   = 500  # print log
    iter_valid = 500  # validation
    iter_save  = list(range(0, num_iteration, 2000))

    log.write('optimizer\n  %s\n' % (optimizer))
    log.write('\n')


    # -------------------start training here!----------------------
    log.write('** start training here **\n')
    log.write('   batch_size = %d\n' % (batch_size))
    log.write('   experiment = %s\n' % str(__file__.split('/')[-2:]))
    log.write('                      |----- VALID -----|----- TRAIN/BATCH -----\n')
    log.write('rate      iter  epoch | loss    acc     |  loss0  | time        \n')
    log.write('----------------------------------------------------------------\n')
            # 0.00000   0.00* 0.00  |  4.625   0.000  |  0.000  |  0 hr 00 min

    def message(mode='print'):
        if mode == ('print'):
            asterisk = ' '
            loss = batch_loss
        
        if mode == ('log'):
            asterisk = '*' if iteration in iter_save else ' '
            loss = train_loss
        
        text = \
            '%0.5f  %5.2f%s %4.2f  | ' % (rate, iteration / 10000, asterisk, epoch,) +\
            '%6.3f  %6.3f  | ' % (*valid_loss, ) + \
            '%6.3f  | ' % (*loss, ) + \
            '%s' % (time_to_str(timer() - start_timer, 'min'))
        return text 

    # ---------
    valid_loss = np.zeros(2, np.float32)
    train_loss = np.zeros(1, np.float32)
    batch_loss = np.zeros_like(train_loss)
    sum_train_loss = np.zeros_like(train_loss)
    sum_train = 0

    start_timer = timer()
    epoch = start_epoch
    iteration = start_iteration
    rate = 0

    criteria = nn.CrossEntropyLoss()

    while iteration < num_iteration:
        for t, batch in enumerate(train_loader):
            if iteration in iter_save:
                if iteration != start_iteration:
                    torch.save({
                        'state_dict': net.state_dict(),
                        'iteration': iteration,
                        'epoch': epoch,
                    }, out_dir + '/checkpoint/%08d_model.pth' % (iteration))
                    pass

            if (iteration % iter_valid == 0):
                #if iteration!=start_iteration:
                    valid_loss = do_valid(net, valid_loader)  #
                    pass

            if (iteration % iter_log == 0):
                print('\r', end='', flush=True)
                log.write(message(mode='log') + '\n')

            # learning rate schduler 
            rate = get_learning_rate(optimizer)

            # one iteration update
            batch_size = len(batch['index'])
            label = batch['label'].to(device)
            image = batch['image'].to(device)

            net.train()
            optimizer.zero_grad()
            if config['is_mixed_precision']:
                with amp.autocast():
                    logit = net(image)
                    loss0 = criteria(logit, label)
                scaler.scale(loss0).backward()
                scaler.unscale_(optimizer)
                #torch.nn.utils.clip_grad_norm_(net.parameters(), 2)
                scaler.step(optimizer)
                scaler.update()
            else:
                logit = net(image)
                loss0 = criteria(logit, label)
                (loss0).backward()
                optimizer.step()
            
            epoch += 1 / len(train_loader)
            iteration += 1
            batch_loss = np.array([loss0.item()])
            sum_train_loss += batch_loss
            sum_train += 1
            if iteration % 100 == 0:
                train_loss = sum_train_loss / (sum_train + 1e-12)
                sum_train_loss[...] = 0
                sum_train = 0
            print('\r', end='', flush=True)
            print(message(mode='print'), end='', flush=True)
    log.write('\n')


if __name__ == '__main__':
    with open('/home/data_normal/abiz/wuzhiqiang/wzq/food101/scripts/baseline.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    print(config)
    run_train(config)