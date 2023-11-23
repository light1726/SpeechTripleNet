import torch
import numpy as np
import torch.nn as nn
import torch.distributions as D
import torch.nn.functional as F

from modules import TVConvSABNEncoder, TVConvSADiscBNEncoder, \
    TIVConvEncoder, ConvSADecoder, sequence_mask, TVConvDiscLNEncoderJoint


class BaseModel(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super().__init__()
        self.speaker_posterior = TIVConvEncoder(
            in_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'],
            h_channels=model_config['spk_encoder']['h_channels'],
            out_channels=model_config['prior']['spk_dim'],
            conv_kernels=model_config['spk_encoder']['conv_kernels'],
            paddings=model_config['spk_encoder']['paddings'],
            pooling_type=model_config['spk_encoder']['pooling_type'],
            pooling_kernels=model_config['spk_encoder']['pooling_kernels'],
            activation=model_config['spk_encoder']['activation'])

    @staticmethod
    def dec_normal_nll(x_hat, x_gt, lens=None):
        if lens is not None:
            mask = sequence_mask(lens).to(x_hat.dtype)
        else:
            B = x_hat.size(0)
            T = x_hat.size(2)
            mask = torch.ones([B, T], dtype=x_hat.dtype, device=x_hat.device)
        dist_normal = D.Normal(x_hat, torch.ones_like(x_hat, requires_grad=False))
        logp = dist_normal.log_prob(x_gt)
        nll = - torch.sum(torch.sum(logp, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        nll = torch.mean(nll)
        return nll

    @staticmethod
    def dec_laplace_nll(x_hat, x_gt, lens=None):
        if lens is not None:
            mask = sequence_mask(lens).to(x_hat.dtype)
        else:
            B = x_hat.size(0)
            T = x_hat.size(2)
            mask = torch.ones([B, T], dtype=x_hat.dtype, device=x_hat.device)
        dist_laplace = D.Laplace(x_hat, torch.ones_like(x_hat, requires_grad=False))
        logp = dist_laplace.log_prob(x_gt)
        nll = - torch.sum(torch.sum(logp, dim=1) * mask, dim=1) / torch.sum(mask, dim=1)
        nll = torch.mean(nll)
        return nll


class VAECCDPJ(BaseModel):
    def __init__(self, preprocess_config, model_config, device):
        super(VAECCDPJ, self).__init__(preprocess_config, model_config)
        self.device = device
        self.content_posterior = TVConvSABNEncoder(
            in_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'],
            h_channels=model_config['content_conv_sa_encoder']['h_channels'],
            out_channels=model_config['prior']['con_dim'],
            conv_kernels=model_config['content_conv_sa_encoder']['conv_kernels'],
            paddings=model_config['content_conv_sa_encoder']['paddings'],
            activation=model_config['content_conv_sa_encoder']['activation'],
            sa_hidden=model_config['content_conv_sa_encoder']['sa_hidden'],
            sa_layer=model_config['content_conv_sa_encoder']['sa_layer'],
            sa_head=model_config['content_conv_sa_encoder']['sa_head'],
            sa_filter_size=model_config['content_conv_sa_encoder']['sa_filter_size'],
            sa_kernel_size=model_config['content_conv_sa_encoder']['sa_kernel_size'],
            sa_dropout=model_config['content_conv_sa_encoder']['sa_dropout'],
            max_seq_len=model_config['transformer']['max_mel_len'], device=device)
        self.prosody_posterior = TVConvDiscLNEncoderJoint(
            in_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'],
            h_channels=model_config['prosody_encoder']['h_channels'],
            n_pitch_class=model_config['prosody_encoder']['n_class'],
            n_energy_class=model_config['prosody_encoder']['n_class'],
            conv_kernels=model_config['prosody_encoder']['conv_kernels'],
            paddings=model_config['prosody_encoder']['paddings'],
            activation=model_config['prosody_encoder']['activation'],
            temperature=model_config['prosody_encoder']['temperature'],
            pit_dim=model_config['prior']['pit_dim'],
            ene_dim=model_config['prior']['ene_dim'], device=device)
        self.decoder = ConvSADecoder(
            in_channels=model_config['prior']['con_dim'] + model_config['prior']['spk_dim']
                        + model_config['prior']['pit_dim'] + model_config['prior']['ene_dim'],
            h_channels=model_config['conv_sa_decoder']['h_channels'],
            out_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'],
            conv_kernels=model_config['conv_sa_decoder']['conv_kernels'],
            paddings=model_config['conv_sa_decoder']['paddings'],
            activation=model_config['conv_sa_decoder']['activation'],
            sa_hidden=model_config['conv_sa_decoder']['sa_hidden'],
            sa_layer=model_config['conv_sa_decoder']['sa_layer'],
            sa_head=model_config['conv_sa_decoder']['sa_head'],
            sa_filter_size=model_config['conv_sa_decoder']['sa_filter_size'],
            sa_kernel_size=model_config['conv_sa_decoder']['sa_kernel_size'],
            sa_dropout=model_config['conv_sa_decoder']['sa_dropout'],
            max_seq_len=model_config['transformer']['max_mel_len'])

    def forward(self, x, spk_ref, pro_ref, lens):
        con_mu, con_log_std = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, z_pit, z_ene = self.prosody_posterior(pro_ref)
        z_con = self.content_posterior.sample(con_mu, con_log_std)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'con_mu': con_mu, 'con_log_std': con_log_std,
                'z_con': z_con, 'spk_mu': spk_mu, 'spk_log_std': spk_log_std,
                'prosody_logits': pro_logits, 'pitch_embds': z_pit, 'energy_embds': z_ene}

    @staticmethod
    def find_non_zero_segments(array):
        start = -1
        end = -1
        result = []
        for i in range(len(array)):
            if array[i] != 0:
                if start == -1:
                    start = i
                else:
                    end = i
            elif start >= 0:
                result.append((start, end))
                start = -1
                end = -1
        if start >= 0:
            result.append((start, end))
        return result

    def random_rising_modify(self, p_idx):
        """
        :param p_idx: [B, T]
        :return: modified p_idx: [B, T]
        """
        p_idx_np = p_idx.detach().cpu().numpy()
        for i in range(p_idx_np.shape[0]):
            seg_inds = self.find_non_zero_segments(p_idx_np[i])
            for j, inds in enumerate(seg_inds):
                if np.random.uniform(0., 1., size=1)[0] > 0.5:
                    p_idx_np[i][inds[0]: inds[1]] = self.prosody_posterior.n_pitch_class - 1
                else:
                    p_idx_np[i][inds[0]: inds[1]] = 2
        p_idx = torch.from_numpy(p_idx_np).to(self.device)
        return p_idx

    def pitch_shift(self, x, spk_ref, lens, shift):
        con_mu, con_log_std = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, _, z_ene = self.prosody_posterior(x)
        pro_probs = F.softmax(pro_logits, dim=1).permute(0, 2, 1).reshape(
            x.size(0), x.size(2), self.prosody_posterior.n_pitch_class,
            self.prosody_posterior.n_energy_class)
        pit_probs = torch.sum(pro_probs, dim=3)
        pit_idx = torch.argmax(pit_probs, dim=2)  # [B, T]
        pit_idx = self.random_rising_modify(pit_idx)
        z_pit = self.prosody_posterior.pitch_embed(pit_idx)
        z_con = self.content_posterior.sample(con_mu, con_log_std)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'pit_idx': z_con}

    def energy_shift(self, x, spk_ref, lens, shift):
        con_mu, con_log_std = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, z_pit, _ = self.prosody_posterior(x)
        pro_probs = F.softmax(pro_logits, dim=1).permute(0, 2, 1).reshape(
            x.size(0), x.size(2), self.prosody_posterior.n_pitch_class,
            self.prosody_posterior.n_energy_class)
        ene_probs = torch.sum(pro_probs, dim=2)
        ene_idx = torch.argmax(ene_probs, dim=2)  # [B, T]
        ene_idx = self.random_rising_modify(ene_idx)
        z_ene = self.prosody_posterior.energy_embed(ene_idx)
        z_con = self.content_posterior.sample(con_mu, con_log_std)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'ene_idx': ene_idx}

    def loss_fn(self, outputs, x_gt, pitch_prior, energy_prior, lens=None):
        x_hat = outputs['x_hat']
        con_mu = outputs['con_mu']
        con_log_std = outputs['con_log_std']
        spk_mu = outputs['spk_mu']
        spk_log_std = outputs['spk_log_std']
        prosody_logits = outputs['prosody_logits']
        con_kl = self.content_posterior.kl_divergence(con_mu, con_log_std, lens)
        spk_kl = self.speaker_posterior.kl_divergence(spk_mu, spk_log_std)
        jnt_kl, pit_kl, ene_kl = self.prosody_posterior.kl_divergence_with_prior(
            prosody_logits, pitch_prior, energy_prior, lens)
        nll = 0.5 * self.dec_laplace_nll(x_hat, x_gt, lens) + 0.5 * self.dec_normal_nll(x_hat, x_gt, lens)
        pro_reg = self.prosody_posterior.kl_divergence_with_uniform(prosody_logits, lens)
        return nll, con_kl, spk_kl, jnt_kl, pro_reg, pit_kl, ene_kl


class VAEDCDPJ(VAECCDPJ):
    def __init__(self, preprocess_config, model_config, device):
        super(VAEDCDPJ, self).__init__(preprocess_config, model_config, device)
        self.device = device
        self.content_posterior = TVConvSADiscBNEncoder(
            in_channels=preprocess_config['preprocessing']['mel']['n_mel_channels'],
            h_channels=model_config['content_conv_sa_encoder']['h_channels'],
            conv_kernels=model_config['content_conv_sa_encoder']['conv_kernels'],
            paddings=model_config['content_conv_sa_encoder']['paddings'],
            activation=model_config['content_conv_sa_encoder']['activation'],
            sa_hidden=model_config['content_conv_sa_encoder']['sa_hidden'],
            sa_layer=model_config['content_conv_sa_encoder']['sa_layer'],
            sa_head=model_config['content_conv_sa_encoder']['sa_head'],
            sa_filter_size=model_config['content_conv_sa_encoder']['sa_filter_size'],
            sa_kernel_size=model_config['content_conv_sa_encoder']['sa_kernel_size'],
            sa_dropout=model_config['content_conv_sa_encoder']['sa_dropout'],
            max_seq_len=model_config['transformer']['max_mel_len'],
            n_class=model_config['content_conv_sa_encoder']['n_class'],
            temperature=model_config['content_conv_sa_encoder']['temperature'],
            embd_dim=model_config['prior']['con_dim'], device=device)

    def forward(self, x, spk_ref, pro_ref, lens):
        con_logits, z_con = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, z_pit, z_ene = self.prosody_posterior(pro_ref)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'con_logits': con_logits,
                'z_con': z_con, 'spk_mu': spk_mu, 'spk_log_std': spk_log_std,
                'prosody_logits': pro_logits, 'pitch_embds': z_pit, 'energy_embds': z_ene}

    def pitch_shift(self, x, spk_ref, lens, shift):
        con_logits, z_con = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, _, z_ene = self.prosody_posterior(x)
        pro_probs = F.softmax(pro_logits, dim=1).permute(0, 2, 1).reshape(
            x.size(0), x.size(2), self.prosody_posterior.n_pitch_class,
            self.prosody_posterior.n_energy_class)
        pit_probs = torch.sum(pro_probs, dim=3)
        pit_idx = torch.argmax(pit_probs, dim=2)  # [B, T]
        pit_idx = torch.where(
            torch.eq(pit_idx, 0), torch.zeros_like(pit_idx),
            torch.clamp(pit_idx + shift, 1, self.prosody_posterior.n_pitch_class - 1))
        z_pit = self.prosody_posterior.pitch_embed(pit_idx)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'pit_idx': pit_idx}

    def energy_shift(self, x, spk_ref, lens, shift):
        con_logits, z_con = self.content_posterior(x, lens)
        spk_mu, spk_log_std = self.speaker_posterior(spk_ref)
        pro_logits, z_pit, _ = self.prosody_posterior(x)
        pro_probs = F.softmax(pro_logits, dim=1).permute(0, 2, 1).reshape(
            x.size(0), x.size(2), self.prosody_posterior.n_pitch_class,
            self.prosody_posterior.n_energy_class)
        ene_probs = torch.sum(pro_probs, dim=2)
        ene_idx = torch.argmax(ene_probs, dim=2)  # [B, T]
        ene_idx = torch.where(
            torch.eq(ene_idx, 0), torch.zeros_like(ene_idx),
            torch.clamp(ene_idx + shift, 1, self.prosody_posterior.n_energy_class - 1))
        z_ene = self.prosody_posterior.energy_embed(ene_idx)
        z_spk = self.speaker_posterior.sample(spk_mu, spk_log_std)
        time_len = z_con.size(2)
        z_spk = z_spk.unsqueeze(2).expand(-1, -1, time_len)
        dec_in = torch.cat([z_con, z_spk, z_pit, z_ene], dim=1)
        x_hat = self.decoder(dec_in, lens)
        return {'x_hat': x_hat, 'ene_idx': ene_idx}

    def loss_fn(self, outputs, x_gt, pitch_prior, energy_prior, lens=None):
        x_hat = outputs['x_hat']
        con_logits = outputs['con_logits']
        spk_mu = outputs['spk_mu']
        spk_log_std = outputs['spk_log_std']
        prosody_logits = outputs['prosody_logits']
        con_kl = self.content_posterior.kl_divergence(con_logits, lens)
        spk_kl = self.speaker_posterior.kl_divergence(spk_mu, spk_log_std)
        jnt_kl, pit_kl, ene_kl = self.prosody_posterior.kl_divergence_with_prior(
            prosody_logits, pitch_prior, energy_prior, lens)
        nll = 0.5 * self.dec_laplace_nll(x_hat, x_gt, lens) + 0.5 * self.dec_normal_nll(x_hat, x_gt, lens)
        pro_reg = self.prosody_posterior.kl_divergence_with_uniform(prosody_logits, lens)
        return nll, con_kl, spk_kl, jnt_kl, pro_reg, pit_kl, ene_kl
