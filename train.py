import os
import yaml
import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from models import ModelCollection
from datasets import MelDataset
from utils import get_vocoder, get_param_num, log, synth_one_sample


def evaluate_step(model, dataloader):
    nll = 0.
    con_kl = 0.
    spk_kl = 0.
    pro_kl = 0.
    pro_reg = 0.
    pit_kl = 0.
    ene_kl = 0.
    sample_size = 0
    for _, mels, spk_mels, lf0_q, ene_q, lens in dataloader:
        mels = mels.to(model.device)
        spk_mels = spk_mels.to(model.device)
        lf0_q = lf0_q.to(model.device)
        ene_q = ene_q.to(model.device)
        lens = lens.to(model.device)
        outputs = model(mels, spk_mels, mels, lens)
        _nll, _con_kl, _spk_kl, _pro_kl, _pro_reg, _pit_kl, _ene_kl = model.loss_fn(outputs, mels, lf0_q, ene_q, lens)
        batch_size = mels.shape[0]
        sample_size += batch_size
        nll += _nll.item() * batch_size
        con_kl += _con_kl.item() * batch_size
        spk_kl += _spk_kl.item() * batch_size
        pro_kl += _pro_kl.item() * batch_size
        pro_reg += _pro_reg.item() * batch_size
        pit_kl += _pit_kl.item() * batch_size
        ene_kl += _ene_kl.item() * batch_size
    return (nll / sample_size, con_kl / sample_size, spk_kl / sample_size, pro_kl / sample_size,
            pro_reg / sample_size, pit_kl / sample_size, ene_kl / sample_size)


def main(args, configs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Start training ...")
    preprocess_config, model_config, train_config = configs
    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    trn_set = MelDataset(preprocess_config, subset='train')
    val_set = MelDataset(preprocess_config, subset='val')
    tst_set = MelDataset(preprocess_config, subset='test')
    batch_size = train_config["optimizer"]["batch_size"]
    trn_loader = DataLoader(
        trn_set, batch_size=batch_size, num_workers=8, shuffle=True,
        collate_fn=trn_set.collate_fn, pin_memory=True)
    val_loader = DataLoader(
        val_set, batch_size=batch_size, num_workers=8, shuffle=True,
        collate_fn=val_set.collate_fn, pin_memory=True)
    val_single_sampler = DataLoader(val_set, batch_size=1, shuffle=True)
    tst_single_sampler = DataLoader(tst_set, batch_size=1, shuffle=True)
    # Prepare model
    model = ModelCollection[args.mdl](preprocess_config, model_config, device).to(device)
    num_param = get_param_num(model)
    print("Number of Parameters:", num_param)

    # set optimizers
    learning_rate = train_config["optimizer"]["learning_rate"]
    betas = train_config["optimizer"]["betas"]
    eps = train_config["optimizer"]["eps"]
    optim = torch.optim.AdamW(model.parameters(), learning_rate, betas=betas, eps=eps)

    # Load vocoder
    vocoder = get_vocoder(model_config, device)

    # Training hyper-parameters
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    val_step = train_config["step"]["val_step"]
    con_gamma = train_config["optimizer"]["con_gamma"]
    spk_gamma = train_config["optimizer"]["spk_gamma"]
    pro_gamma = train_config["optimizer"]["pro_gamma"]
    pro_reg_w = train_config["optimizer"]["pro_reg_w"]
    con_mi = args.con_mi if args.con_mi is not None else train_config["optimizer"]["con_mi"]
    spk_mi = args.spk_mi if args.spk_mi is not None else train_config["optimizer"]["spk_mi"]
    pro_mi = train_config["optimizer"]["pro_mi"]
    stop_step = train_config["step"]["mi_stop"]

    # Experiment name
    exp_name = 'output-{}-c_{}_{}-s_{}_{}-p_{}_{}'.format(
        args.mdl, con_gamma, con_mi, spk_gamma, spk_mi, pro_gamma, pro_mi)

    # Load model checkpoint
    if args.restore_step:
        ckpt_path = os.path.join(
            exp_name, train_config["path"]["ckpt_path"], "{}.pth.tar".format(args.restore_step))
        ckpt = torch.load(ckpt_path)
        model.load_state_dict(ckpt["model"])

    # Init logger
    for p in train_config["path"].values():
        os.makedirs(os.path.join(exp_name, p), exist_ok=True)
    train_log_path = os.path.join(exp_name, train_config["path"]["log_path"], "train")
    val_log_path = os.path.join(exp_name, train_config["path"]["log_path"], "val")
    os.makedirs(train_log_path, exist_ok=True)
    os.makedirs(val_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
    val_logger = SummaryWriter(val_log_path)
    val_losses = []
    global_step = args.restore_step + 1
    while True:
        for fids, mels, spk_mels, lf0_q, ene_q, lens in trn_loader:
            mels = mels.to(device)
            spk_mels = spk_mels.to(device)
            lf0_q = lf0_q.to(device)
            ene_q = ene_q.to(device)
            lens = lens.to(device)
            model.zero_grad(set_to_none=True)
            outputs = model(mels, spk_mels, mels, lens)
            nll, con_kl, spk_kl, pro_kl, pro_reg, pit_kl, ene_kl = model.loss_fn(outputs, mels, lf0_q, ene_q, lens)
            con_c = np.clip(con_mi / stop_step * global_step, 0, con_mi)
            spk_c = np.clip(spk_mi / stop_step * global_step, 0, spk_mi)
            pro_c = np.clip(pro_mi / stop_step * global_step, 0, pro_mi)
            loss = (nll + con_gamma * (con_kl - con_c).abs()
                    + spk_gamma * (spk_kl - spk_c).abs()
                    + pro_gamma * pro_kl.abs() + pro_reg_w * (pro_reg - pro_c).abs())
            loss.backward()
            optim.step()

            if global_step % log_step == 0:
                losses = [nll.item(), con_kl.item(), spk_kl.item(), pro_kl.item(),
                          pro_reg.item(), pit_kl.item(), ene_kl.item()]
                message1 = "Step {}/{}, ".format(global_step, total_step)
                message2 = "NLL: {:.3f}, con-kl: {:.3f}, spk-kl: {:.3f}," \
                           " pro-kl: {:.3f}, pro-reg: {:.3f}, pit-kl: {:.3f}, ene-kl: {:.3f}".format(*losses)
                with open(os.path.join(train_log_path, "log.txt"), "a") as f:
                    f.write(message1 + message2 + "\n")
                print(message1 + message2)
                log(train_logger, global_step, losses=losses, model=args.mdl)

            if global_step % val_step == 0:
                model.eval()
                val_nll, val_con_kl, val_spk_kl, val_pro_kl, val_pro_reg, val_pit_kl, val_ene_kl = evaluate_step(model, val_loader)
                log(val_logger, step=global_step, model=args.mdl,
                    losses=[val_nll, val_con_kl, val_spk_kl, val_pro_kl,
                            val_pro_reg, val_pit_kl, val_ene_kl])
                message = "Val-NLL: {:.3f}, val-con-kl: {:.3f}, val-spk-kl: {:.3f}," \
                          " val-pro-kl: {:.3f}, val-pro-reg: {:.3f} val-pit-kl: {:.3f}, val-ene-kl: {:.3f}".format(
                    val_nll, val_con_kl, val_spk_kl, val_pro_kl, val_pro_reg, val_pit_kl, val_ene_kl)
                with open(os.path.join(val_log_path, "log.txt"), "a") as f:
                    f.write(message + "\n")
                print(message)
                val_losses.append(
                    [val_nll, val_con_kl, val_spk_kl, val_pro_kl, val_pro_reg, val_pit_kl, val_ene_kl])
                # reconstruction
                _, val_mels, val_spk_mels, _, _, val_lens = next(iter(val_single_sampler))
                _, tst_mels, tst_spk_mels, _, _, tst_lens = next(iter(tst_single_sampler))
                val_mels = val_mels.to(device)
                tst_mels = tst_mels.to(device)
                val_spk_mels = val_spk_mels.to(device)
                tst_spk_mels = tst_spk_mels.to(device)
                tst_pro_mels = tst_mels[:, :, :val_lens.detach().cpu().numpy()[0]]
                val_lens = val_lens.to(device)
                tst_lens = tst_lens.to(device)
                with torch.no_grad():
                    pro_cvt = model.pitch_shift(
                        val_mels, val_spk_mels, val_lens, shift=1)['x_hat']
                    spk_cvt = model(
                        val_mels, tst_spk_mels, val_mels, val_lens)['x_hat']
                src_fig, src_wav = synth_one_sample(
                    val_mels, val_lens, vocoder, model_config, preprocess_config)
                log(val_logger, step=global_step, fig=src_fig,
                    tag="Val/step-{}-0-src_mel".format(global_step))
                log(val_logger, step=global_step, audio=src_wav, sampling_rate=sampling_rate,
                    tag="Val/step-{}-0-src_wav".format(global_step))
                pro_cvt_fig, pro_cvt_wav = synth_one_sample(
                    pro_cvt, val_lens, vocoder, model_config, preprocess_config)
                log(val_logger, step=global_step, fig=pro_cvt_fig,
                    tag="Val/step-{}-1-pit-cvt_mel".format(global_step))
                log(val_logger, step=global_step, audio=pro_cvt_wav, sampling_rate=sampling_rate,
                    tag="Val/step-{}-1-pit-cvt_wav".format(global_step))
                spk_cvt_fig, spk_cvt_wav = synth_one_sample(
                    spk_cvt, val_lens, vocoder, model_config, preprocess_config)
                log(val_logger, step=global_step, fig=spk_cvt_fig,
                    tag="Val/step-{}-1-spk-cvt_mel".format(global_step))
                log(val_logger, step=global_step, audio=spk_cvt_wav, sampling_rate=sampling_rate,
                    tag="Val/step-{}-1-spk-cvt_wav".format(global_step))
                tgt_fig, tgt_wav = synth_one_sample(
                    tst_pro_mels, tst_lens, vocoder, model_config, preprocess_config)
                log(val_logger, step=global_step, fig=tgt_fig,
                    tag="Val/step-{}-2-tgt_mel".format(global_step))
                log(val_logger, step=global_step, audio=tgt_wav, sampling_rate=sampling_rate,
                    tag="Val/step-{}-2-tgt_wav".format(global_step))
                model.train()

            if global_step % save_step == 0:
                torch.save(
                    {"model": model.state_dict(), "optimizer": optim.state_dict()},
                    os.path.join(
                        exp_name, train_config["path"]["ckpt_path"], "{}.pth.tar".format(global_step)))
            global_step += 1
            if global_step > total_step:
                avg_val_losses = np.array(val_losses).mean(axis=0).tolist()
                print("Overall: Val-NLL: {:.3f}, val-con-kl: {:.3f},"
                      " val-spk-kl: {:.3f}, val-pro-kl: {:.3f},"
                      " val-pro-reg: {:.3f} val-pit-kl: {:.3f}, val-ene-kl: {:.3f}".format(*avg_val_losses))
                quit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--restore_step", type=int, default=0)
    parser.add_argument("--mdl", type=str, choices=['CCDPJ', 'DCDPJ'], help='model type')
    parser.add_argument(
        "-p", "--preprocess_config", type=str, required=True, help="path to preprocess.yaml")
    parser.add_argument(
        "-m", "--model_config", type=str, required=True, help="path to model.yaml")
    parser.add_argument(
        "-t", "--train_config", type=str, required=True, help="path to train.yaml")
    parser.add_argument("--con_mi", type=float, default=None)
    parser.add_argument("--spk_mi", type=float, default=None)
    args = parser.parse_args()

    # Read Config
    preprocess_config = yaml.load(open(args.preprocess_config, "r"), Loader=yaml.FullLoader)
    model_config = yaml.load(open(args.model_config, "r"), Loader=yaml.FullLoader)
    train_config = yaml.load(open(args.train_config, "r"), Loader=yaml.FullLoader)
    configs = (preprocess_config, model_config, train_config)
    main(args, configs)
