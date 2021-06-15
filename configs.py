class configs:

    ##### Content and style filenames
    content_directory="./inputs"
    style_directory="./style"

    ##### For training
    loss_type = "MSE"
    N_STEP=300
    content_weight=1e-2
    style_weight=1

    ##### Parameters
    N_FFT=2048
    N_FILTERS=4096

    ##### Parameters related to audio processing
    filter_length=1024
    hop_length=256
    win_length=1024
    max_wav_value=32768.0
    sampling_rate=22050

    ##### Which phase going to use (either content or style)
    phase_used="content"
