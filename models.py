import torch
import model
import utils

def S(cfg, info):

    if cfg["ML"]["dataset"]["winter"]["inputs"] != "setup" or cfg["ML"]["dataset"]["winter"]["targets"] != "agg":
        raise NotImplementedError()

    typ = cfg["ML"]["model"]["type"]
    print(f"INFO: init {typ} model")
    if typ != "pm" and typ != "linear":
        print("INFO: cfg: ", cfg["ML"]["model"][typ])
    if typ == "pm":
        return model.pm(cfg, info)
    elif typ == "linear":
        return model.ffn(cfg, info, linear=True)
    elif typ == "ffn":
        return model.ffn(cfg, info)
    elif typ == "res":
        return model.res(cfg, info)
    else:
        raise NotImplementedError(typ)

def model_device(cfg, model):
    model = model.cuda()
    return model

def model_load(cfg, run):
    fname = "{}run{:03d}/model_ema.pt".format(cfg["ML"]["system"]["dir_analyze_local"], run)
    model = torch.load(fname)
    print(f'INFO: loaded {fname}')
    return model

def model_save(cfg, model, opt):
    fname_m = "{}model".format(cfg["ML"]["system"]["dir_check"])
    fname_o = "{}opt".format(cfg["ML"]["system"]["dir_check"])
    if os.path.exists(fname_m):
        os.remove(fname_m)
    if os.path.exists(fname_o):
        os.remove(fname_o)
    torch.save(model.state_dict(), fname_m)
    torch.save(opt.state_dict(), fname_o)
    os.chmod(fname_m, 0o777)
    os.chmod(fname_o, 0o777)
    print("INFO: saved {}".format(fname_m))
    print("INFO: saved {}".format(fname_o))

def model_info(cfg, model):
    print('INFO: model n_params: {}'.format(utils.ML.n_params(model)))

def mod_clip_grad(cfg, mod):
    if cfg["ML"]["training"]["grad_clip_type"] == "norm":
        torch.nn.utils.clip_grad_norm_(mod.parameters(), cfg["ML"]["training"]["grad_clip"], norm_type=2)
    elif cfg["ML"]["training"]["grad_clip_type"] == "value":
        torch.nn.utils.clip_grad_value_(mod.parameters(), cfg["ML"]["training"]["grad_clip"])
    elif cfg["ML"]["training"]["grad_clip_type"] == "none":
        pass
    else:
        raise NotImplementedError()

def init_model_train(cfg, info=True):
    model = {}
    model = S(cfg, info=info)
    model = model_device(cfg, model)
    model.train()
    model_info(cfg, model)

    model_ema = S(cfg, info=info)
    model_ema.requires_grad_(False)
    model_ema = model_device(cfg, model_ema)
    model_ema.eval()
    return model, model_ema

def init_model_analyze(cfg, info=False):
    model = {}
    model = S(cfg, info=info)
    model_ema = S(cfg, info=info)
    if cfg["ML"]["model"]["type"] != "pm":
        model = model_load(cfg, model, "analyze")
        model = model_device(cfg, model)
        model.eval()
        model_info(cfg, model)
    return model, model_ema

if __name__ == '__main__':
    print('Done.')