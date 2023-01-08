class OPT:
    workers = 1
    isize = 320 
    nc = 1 
    nz = 50 
    ngf = 32
    ndf = 32
    nz = 50 
    nc = 51
    niter = 100
    device='cuda'
    gpu_ids = 0 
    ngpu = 1
    model = 'beatgan'
    outf = './save_models'
    
    
    batchsize = 64
    lr = 0.0001 
    beta1 = 0.5 
    
    niter = 100 
    w_adv = 1 
    folder = 0 
    n_aug = 0 
    thres = 0.1
    