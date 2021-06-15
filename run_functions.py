import argparse
import numpy as np
import os
import soundfile as sf
import torch
import torch.optim as optim
from tqdm import tqdm

# From custom modules
from configs import configs
from losses import CustomLoss
from models import SoundStyleTransfer
from utils import GetSpectrogram

##### Function to get model and loss
def get_style_model_and_losses(model, style_tensor, content_tensor, configs):

    ##### Define some variables
    N_FILTERS=configs.N_FILTERS
    N_SAMPLES=configs.N_SAMPLES
    l_type=configs.loss_type

    ##### add content loss:
    target = model(content_tensor).detach()
    content_loss = CustomLoss(target, l_type)

    ##### add style loss:
    target_feature = model(style_tensor).detach()
    
    # Get style gram
    style_features = target_feature.permute(0,2,3,1).view(-1, N_FILTERS)
    style_gram = torch.mm(style_features.t(), style_features)/N_SAMPLES
    style_loss = CustomLoss(style_gram, l_type)
    
    return model, style_loss, content_loss


##### Function to get optimizer
def get_input_optimizer(input_tensor):
    optimizer = optim.LBFGS([input_tensor.requires_grad_()])
    return optimizer


##### Function to run style transfer
def run_style_transfer(model_init, 
                       content_tensor, style_tensor, input_tensor, 
                       configs,
                       style_weight=1, content_weight=1e-2):
    """
    Run the style transfer.
    """
    # Get some variables
    N_SAMPLES=configs.N_SAMPLES
    num_steps=configs.N_STEP

    # Build model
    print('Building the style transfer model..')
    model, style_loss, content_loss = get_style_model_and_losses(model_init, style_tensor, content_tensor, configs)
    optimizer = get_input_optimizer(input_tensor)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():

            optimizer.zero_grad()
            output_tensor = model(input_tensor)
            
            # Get gram
            output_permute = output_tensor.permute(0,2,3,1)
            _, height, width, number = list(output_permute.shape)
            size = height * width * number
            feats = output_permute.view(-1, number)
            gram = torch.mm(feats.t(), feats)/N_SAMPLES
            
            # update loss
            style_loss(gram)
            content_loss(output_tensor)
            
            style_score = style_loss.loss*style_weight
            content_score = content_loss.loss*content_weight 
            
            loss = style_score + content_score
            loss.backward()
            
            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    return input_tensor


##### Define main functions
if __name__ == "__main__":
    '''
    This is the case when there are more than 1 style and content files exists.
    If there are just one content and one style files, you can just see Jupyter Notebook file.
    '''
    ##### Define argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', 
                        '--phase_used', 
                        type=str, 
                        default='content',
                        help="To indicate which phase will be used for audio reconstruction")

    parser.add_argument('-s', 
                        '--save_tag', 
                        type=str, 
                        default='outputs',
                        help="define save directory")

    args = parser.parse_args()
    configs.phase_used = args.phase_used
    configs.save_directory = args.save_tag + "_content{}_style{}".format(configs.content_weight, configs.style_weight)

    ##### Make saving directory
    # Raise ValueError if phase is not either "content" nor "style"
    if configs.phase_used not in ["content", "style"]:
        raise ValueError("phase_used is defined as '{}', which does not match to pre-defined phases(['content', 'style']).".format(configs.phase_used))

    # Create directory with defined phase_used argument
    save_directory = configs.save_directory
    if not os.path.exists(save_directory):
        os.mkdir(save_directory)

    ##### Define get_spectrogram class
    get_spectrogram = GetSpectrogram(configs)

    ##### Define some variables
    N_FFT = configs.N_FFT
    content_directory = configs.content_directory
    style_directory = configs.style_directory
    content_filenames = os.listdir("./inputs")
    style_filenames = os.listdir("./style")

    # Create content and style name pair
    c_s_list = []
    for c in content_filenames:
        for s in style_filenames:
            c_s_list.append([c,s])

    for i, (c, s) in tqdm(enumerate(c_s_list)):

        ##### Reads wav file and produces spectrum
        '''
        With stft, the height becomes frequency (Hz).
        Frequency means the value that FFT is applied with given sample rate.
        '''

        m_content, p_content = get_spectrogram(os.path.join(content_directory, c))
        m_style_original, p_style_original = get_spectrogram(os.path.join(style_directory, s))

        # Define # of samples and # of samples, and put in configs
        N_SAMPLES = m_content.shape[2]
        N_CHANNELS = m_content.shape[1]
        configs.N_SAMPLES = N_SAMPLES
        configs.N_CHANNLES = N_CHANNELS

        # magnitude of style
        m_style = m_style_original.clone()

        while m_style.shape[2] < N_SAMPLES:
            m_style = torch.cat((m_style, m_style_original), dim = 2)

        m_style = m_style[:,:N_CHANNELS, :N_SAMPLES]

        # phase of style
        p_style = p_style_original.clone()

        while p_style.shape[2] < N_SAMPLES:
            p_style = torch.cat((p_style, p_style_original), dim = 2)

        p_style = p_style[:,:N_CHANNELS, :N_SAMPLES]


        ##### Fix shape and tensorize(?)
        N_FILTERS = configs.N_FILTERS

        m_content_tf = torch.unsqueeze(m_content[0].t(), 0)
        m_content_tf = torch.unsqueeze(m_content_tf, 0)

        m_style_tf = torch.unsqueeze(m_style[0].t(), 0)
        m_style_tf = torch.unsqueeze(m_style_tf, 0)

        m_content_tensor = torch.tensor(m_content_tf, dtype=torch.float32).permute(0,3,1,2)
        m_style_tensor = torch.tensor(m_style_tf, dtype=torch.float32).permute(0,3,1,2)

        # filter shape is "[filter_height, filter_width, in_channels, out_channels]"
        std = np.sqrt(2) * np.sqrt(2.0 / ((N_CHANNELS + N_FILTERS) * 11))
        kernel = np.random.randn(1, 11, N_CHANNELS, N_FILTERS)*std

        ##### Run optimization
        model_init = SoundStyleTransfer(N_CHANNELS,N_FILTERS)
        input_tensor = torch.tensor(np.random.randn(1,N_CHANNELS, 1,N_SAMPLES).astype(np.float32)*1e-3, 
                                    dtype=torch.float32,
                                    requires_grad=True)

        output = run_style_transfer(model_init,
                                    m_content_tensor, m_style_tensor, input_tensor,
                                    configs,
                                    style_weight=configs.style_weight, content_weight=configs.content_weight)

        ##### Get and save result
        result = output.permute(0,2,3,1)[0,:,:]

        # Reconstruct
        if configs.phase_used=="content":
            audio_reconstruct = get_spectrogram.stft.inverse(result.permute(0,2,1), p_content)
        elif configs.phase_used=="style":
            audio_reconstruct = get_spectrogram.stft.inverse(result.permute(0,2,1), p_style)

        # Save
        filename = "{}_to_{}.wav".format(c.split(".")[0], s.split(".")[0])
        sf.write(os.path.join(save_directory, filename),
                audio_reconstruct[0][0].detach().cpu().numpy(),
                configs.sampling_rate,
                'PCM_24')
