import matplotlib
from matplotlib import pyplot as plt
matplotlib.use("Agg")


def log(
    logger, step=None, losses=None, fig=None, audio=None, sampling_rate=22050, tag="", model=None):
    if losses is not None:
        logger.add_scalar("Loss/NLL", losses[0], step)
        logger.add_scalar("Loss/CON-KL", losses[1], step)
        logger.add_scalar("Loss/SPK-KL", losses[2], step)
        logger.add_scalar("Loss/PIT-KL", losses[3], step)
        logger.add_scalar("Loss/ENE-KL", losses[4], step)
        logger.add_scalar("Loss/PIT-Reg", losses[5], step)
        logger.add_scalar("Loss/ENE-Reg", losses[6], step)

    if fig is not None:
        logger.add_figure(tag, fig)

    if audio is not None:
        logger.add_audio(tag, audio / max(abs(audio)), sample_rate=sampling_rate)


def synth_one_sample(predictions, lens, vocoder, model_config, preprocess_config):
    if lens is not None:
        mel_prediction = predictions[0].detach()[:, : lens[0]]
    else:
        mel_prediction = predictions[0].detach()
    fig = plot_mel([(mel_prediction.cpu().numpy())], ["Generated-Spectrogram"])

    if vocoder is not None:
        from .model import vocoder_infer
        wav_prediction = vocoder_infer(mel_prediction.unsqueeze(0), vocoder,
                                       model_config, preprocess_config)[0]
    else:
        wav_prediction = None

    return fig, wav_prediction


def plot_mel(data, titles):
    fig, axes = plt.subplots(len(data), 1, squeeze=False)
    if titles is None:
        titles = [str(i) for i in range(len(data))]

    for i in range(len(data)):
        mel = data[i]
        axes[i][0].imshow(mel, origin="lower")
        axes[i][0].set_aspect(2.5, adjustable="box")
        axes[i][0].set_ylim(0, mel.shape[0])
        axes[i][0].set_title(titles[i], fontsize="medium")
        axes[i][0].tick_params(labelsize="x-small", left=False, labelleft=False)
        axes[i][0].set_anchor("W")
    return fig
