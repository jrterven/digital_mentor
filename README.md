# Digital Mentor
This project creates an avatar that can be used as a mentor. We provide a Jupyter notebook to run the demo.

In order to run, you need two API keys:
- OpenAI API key to run the underlining gpt model.
- Eleven labs API key for the text to voice model.

The resulting animation is based on [Wav2lip](https://github.com/Mozer/wav2lip)  
For this, you need to download two models and save them in the following directories:
- Put [wav2lip_gan.pth](https://iiitaphyd-my.sharepoint.com/:u:/g/personal/radrabha_m_research_iiit_ac_in/EdjI7bZlgApMqsVoEUUXpLsBxqXbn5z8VTmoxp55YNDcIA?e=n9ljGW) inside checkpoints directory.
- Put [s3fd-619a316812.pth](https://www.dropbox.com/scl/fi/5r5tem8lm9r9j220wqbhk/s3fd-619a316812.pth?rlkey=t6kxmzim1rmiqb529rstn147t&dl=0) inside face_detection/detection.

