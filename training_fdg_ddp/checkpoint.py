import os
import torch
import shutil
from util import preprocess_teacher

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


def load_checkpoint_fdg(args, model, model_t, optimizer=None, verbose=True):

    # teacher model checkpoint loading
    student_path_split = args.resume.split('student')
    resume_path = student_path_split[0]+'teacher'+student_path_split[1]
    checkpoint = torch.load(resume_path)
    model_t.load_state_dict(checkpoint['state_dict'], False)

    # student model checkpoint loading
    checkpoint = torch.load(args.resume)
    start_epoch = 0
    best_acc = 0

    if "epoch" in checkpoint:
        start_epoch = checkpoint['epoch']

    if "best_acc" in checkpoint:
        best_acc = checkpoint['best_acc']

    model.load_state_dict(checkpoint['state_dict'], False)

    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])

        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()

    if verbose:
        print("=> loading checkpoint '{}' (epoch {})"
                .format(args.resume, start_epoch))

    return model, model_t, optimizer, best_acc, start_epoch

def check_local(args, model, model_teacher, optimizer=None):
    args.resume = None
    for epoch in range(args.epochs):
        file_path = os.path.join(args.ckpt, 'student/checkpoint_%03d.pth.tar' % epoch)
        if os.path.exists(file_path):
            args.resume = file_path
    if args.resume!= None:
        return load_checkpoint_fdg(args,model,model_teacher,optimizer)
    else:
        if args.pretrained != None:
            print ('load pretrained model')
            checkpoint = torch.load(args.pretrained)
            state_dict = checkpoint['state_dict']
            model.load_state_dict(state_dict, True)
        if args.moco_pretrained != None:
            print ('load pretrained model')
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
        # |-Teacher model Initializing
        preprocess_teacher(model, model_teacher)
        return model,model_teacher,optimizer,0,0
