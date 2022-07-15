import os
import torch
import shutil

def save_checkpoint(state, is_best, epoch, save_path='./'):
    print("=> saving checkpoint '{}'".format(epoch))
    torch.save(state, os.path.join(save_path, 'checkpoint.pth.tar'))
    if(epoch % 10 == 0):
        torch.save(state, os.path.join(save_path, 'checkpoint_%03d.pth.tar' % epoch))
    if is_best:
        if epoch >= 90:
            shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'),
                            os.path.join(save_path, 'model_best_out_090_epochs.pth.tar'))
        else:
            shutil.copyfile(os.path.join(save_path, 'checkpoint.pth.tar'),
                            os.path.join(save_path, 'model_best_in_090_epochs.pth.tar'))

def load_checkpoint(args, model, model_t, optimizer=None, verbose=True):

    # student model checkpoint loading
    checkpoint = torch.load(args.fine_tune)
    start_epoch = 0
    best_acc = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_acc" in checkpoint:
        best_acc = checkpoint['best_acc']

    dict_ = {}
    for k in list(checkpoint['state_dict'].keys()):
        if 'module' in k:
            dict_[k[7:]] = checkpoint['state_dict'][k]
        else:
            dict_[k] = checkpoint['state_dict'][k]

    model.load_state_dict(dict_, True)
    if args.scheme == 'fdg':
        model_t.load_state_dict(dict_, True)
    
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda(args.gpu)

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
                .format(args.fine_tune, start_epoch))
    if args.scheme == 'fdg':
        return model, model_t, optimizer, best_acc, start_epoch
    else:
        return model, optimizer, best_acc, start_epoch

def preprocess_teacher(model, teacher):
    for param_m, param_t in zip(model.parameters(), teacher.parameters()):
        param_t.data.copy_(param_m.data)  # initialize
        param_t.requires_grad = False  # not update by gradient
        
def check_local(args, model, model_t, optimizer=None):
    if args.fine_tune!=None:
        return load_checkpoint(args,model,model_t,optimizer)
    if args.pretrained != None:
        checkpoint = torch.load(args.pretrained)
        dict_ = {}
        for k in list(checkpoint['state_dict'].keys()):
            if 'module' in k:
                dict_[k[7:]] = checkpoint['state_dict'][k]
            else:
                dict_[k] = checkpoint['state_dict'][k]
        model.load_state_dict(dict_, True)
    if args.moco_pretrained != None:
        checkpoint = torch.load(args.moco_pretrained)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        model.load_state_dict(state_dict, False)
    if args.scheme == 'fdg':
        preprocess_teacher(model, model_teacher)
        return model,model_teacher,optimizer,0,0
    else:
        return model,optimizer,0,0