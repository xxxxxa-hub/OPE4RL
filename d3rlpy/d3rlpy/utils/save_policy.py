import pickle

__all__ = ["save_policy"]

def save_policy(model, path, epoch = None):
    saved_dict = {}
    fc0_weight = list(model._impl._modules.policy.parameters())[0].detach().cpu().numpy()
    saved_dict["fc0/weight"] = fc0_weight

    fc0_bias = list(model._impl._modules.policy.parameters())[1].detach().cpu().numpy()
    saved_dict["fc0/bias"] = fc0_bias

    fc1_weight = list(model._impl._modules.policy.parameters())[2].detach().cpu().numpy()
    saved_dict["fc1/weight"] = fc1_weight

    fc1_bias = list(model._impl._modules.policy.parameters())[3].detach().cpu().numpy()
    saved_dict["fc1/bias"] = fc1_bias

    fc2_weight = list(model._impl._modules.policy.parameters())[4].detach().cpu().numpy()
    saved_dict["fc2/weight"] = fc2_weight

    fc2_bias = list(model._impl._modules.policy.parameters())[5].detach().cpu().numpy()
    saved_dict["fc2/bias"] = fc2_bias

    last_fc_weight = list(model._impl._modules.policy.parameters())[6].detach().cpu().numpy()
    saved_dict["last_fc/weight"] = last_fc_weight

    last_fc_bias = list(model._impl._modules.policy.parameters())[7].detach().cpu().numpy()
    saved_dict["last_fc/bias"] = last_fc_bias

    last_fc_log_std_weight = list(model._impl._modules.policy.parameters())[8].detach().cpu().numpy()
    saved_dict["last_fc_log_std/weight"] = last_fc_log_std_weight

    last_fc_log_std_bias = list(model._impl._modules.policy.parameters())[9].detach().cpu().numpy()
    saved_dict["last_fc_log_std/bias"] = last_fc_log_std_bias

    if epoch is None:
        pickle.dump(saved_dict,open("{}/policy.pkl".format(path),"wb"))
    else:
        pickle.dump(saved_dict,open("{}/policy_{}.pkl".format(path, epoch),"wb"))