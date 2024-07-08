# 24Spr_CMagee_LanguageDectection

In our interconnected world, effective but accessible modes of communication are vital for making lasting connections, spreading information, and fostering understanding. Our everyday communities consist of individuals who speak numerous languages, exhibit various dialects, and possess different accents. Furthermore, as our world relies more and more on virtual communication it's important to address language barriers in virtual spaces. 

Our project utilizes deep learning techniques to construct a fine-tuned audio classification and recognition model. More specifically, our models have the ability to identify and transcribe speech recordings in over twenty languages from only fifteen seconds of audio.  The success achieved in this project contributes to effective and accessible communication, but also raises cultural awareness to various dialects and languages. In addition, the outcomes of the project can enhance user interactions in virtual settings and improve their experiences with online content, ultimately contributing to a more immersive digital experience. By addressing language barriers through projects like this one, we are able to promote inclusivity, accessibility, and interconnectedness among individuals in our society.

Note: Files for final model selection are wav2vec2-jack.py, wav2vec2-final-predict.py, and whisper.py

DIRECTORY:

CapstoneTemplate - Existing files from template provided by Prof. Lo

Code - All code used in development of the model
* DEMO FILES - Files to create UI and launch Demo used during our Capstone Presentation
* Confusion Matrix - Folder with images of confusion matrices from final Wav2Vec 2.0 model
* test_data_subset - Limited selection of audio files
* code_jack.py - Inital EDA and data configurations from beginning of project
* model_mlp.py - Binary multilayer perceptron. Built a small classifier with pytorch to get a sense of how to handle the data.
* model_mlp_load_data.py - Same as model_mlp.py but with different model size
* subseting_data.py - Loading data from 23 different languages to subset into final training dataset. This had to be done because the Common Voice dataset was far too large to use with our computing capacity, so we had to narrow it down with these methods.
* uploading_model_to_hf.py - Script to upload models and test access to hugging face repos.
* wav2vec2-final-predict.py - Model to get metrics from model predictions in wav2vec2-jack.py. Contains functions to obtain confusion matricies, accuracy, f1, etc.
* wav2vec2-jack.py - FINAL model used to train wav2vec 2.0. The results from this model are what are reported in the paper.
* wav2vec2-pt4.py - Training script for wav2vec 2.0.
* wav2vec2-transcription.py - Training script for wav2vec 2.0 for a potential transcription model. Was not selected and whisper used instead for transcription.
* wav2vec2_example_predict.py - Predicting model with data directly from hugging face.
* wav2vec2_inference.py - Predicting model with data from local device.
* wav2vec2_predict.py - Predicting model with data from local device.
* vav2vec2_manual_predict - Predicting model with data from hugging face. Hugging face data was loading locally, and then predictions made from those loaded files.
* whisper-small-fine-tune.py - - Pre-training Whisper model for transcription task with whisper-tiny. 
* whisper.py - - Pre-training Whisper model for transcription task with whisper-tiny with different hyperparameters. 

Proposals - Output from our proposal generators to devilver inital proposals at the beginning of the semester

requirements.txt - Package versions and dependencies
