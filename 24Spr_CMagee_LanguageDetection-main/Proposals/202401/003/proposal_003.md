
# Capstone Proposal
## Language Detection Using Audio Data
### Proposed by: Carrie Magee
#### Email: mageec@gwu.edu
#### Advisor: Edwin Lo
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
 
             The objective of this project is to design a spoken language detection program using human speech recordings provided by
             the Common Voices dataset. The diverse range of accents, dialects, and languages that make up the dataset will allow us 
             to develop a comprehensive language detection program that can accurately identify and transcribe speech in various languages, 
             regardless of their size or prevalance. We will employ deep learning techniques to train our model and optimize the performance
             of our language detection system."
            

# ![Figure 1: Example figure](202401_003.png)
# *Figure 1: Caption*

## 2 Dataset:  

            As mentioned above, the dataset for our project is called the Common Voice Corpus 16.1 created by Mozilla. 
            The data includes short audio recordings of human speech. The data is collected on a voluntary basis and contains 
            over 3,000 hours of audio recorded in 100+ languages. The audio is stored in MP3 format and 
            accompanied by a corresponding text file containing the spoken content. An updated dataset is released every few months, 
            and we will use the latest version that was released on 1/4/2024. This version of the dataset contains 3,438 recorded hours 
            and 2,856 validated hours, which are the hours of audio that have been validated by other individuals proficient in the 
            language used in the given recording. The Common Voices 16.1 dataset has also been uploaded to the Hugging Face Datasets library 
            which is how we will be accessing the data for our project.   

            Common Voice dataset: https://commonvoice.mozilla.org/en/datasets

            Hugging Face datasource: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1

            

## 3 Rationale:  

            In our interconnected world, effective but accessible modes of communication are vital for making lasting connections, 
            spreading information, and fostering understanding. In our capstone class, we have students who speak numerous languages, 
            exhibit various dialects, and possess different accents. Furthermore, as our world relies more and more on virtual 
            communication it's important to address language barriers in virtual spaces. 
            
            If these virtual communication barriers are not properly addressed they can have real implications in various domains, 
            including in the classroom setting, where students from diverse linguistic backgrounds may face challenges comprehending 
            course materials and actively participating. In the global business world, clear and seamless communication is essential 
            for building strong professional relationships and achieving professional success. Similarly, in healthcare, providers 
            often assist patients who speak different languages or lack the vocabulary required to advocate for their health needs. 
            Language detection systems, which we will be focusing on in our capstone project, can improve things like
            patient-provider communication. This can lead to more accurate diagnoses and treatment, streamline patients' 
            experiences, and potentially reduce interpretation costs in the medical field. 
            
            Our capstone project aims to utilize deep learning techniques to create a spoken language detection model. More specifically, 
            our aim is to identify and transcribe human speech in various languages. 
            
            Success in this project will contribute to effective, accessible communication but also raise cultural awareness to various 
            dialects and languages in our communities. In addition, the outcomes of this project can enhance user interactions in virtual 
            settings and improve their experiences with online content, ultimately contributing to a more immersive digital experience.
            Most importantly, the results of this project can greatly benefit individuals with hearing-related impairments by improving 
            their ability to interact with spoken language content. In conclusion, by addressing language barriers through projects like ours, 
            we are able to promote inclusivity, accessibility, and interconnectedness within our society.
            

## 4 Approach:  

            Our approach will rely on using deep learning techniques, particularly neural networks, to construct our model. More specifically,
            convolutional neural networks (CNNs), which are often used for image-related data, can be adapted for our language identification system. 
            Through the transformation of audio data into either audio waveforms or spectrograms, visual representations of the audio, we can 
            harness them as inputs for convolutional neural networks. Waveforms are useful for analyzing features such as duration, pauses in speech, and
            speech rate. Spectograms are useful for collecting information about different phonemes (distinct units of sound) and their frequencies.
            Another potential model architecture we can explore is transformers. Transformers can be useful for speech recognition (input as audio output as text) or 
            audio classification (input is audio output is probability of specific language). As we delve further into the project, we will gain a clearer 
            understanding of our specific objectives. This will enable us to select the most suitable model architecture for achieving our desired outcomes.
            

## 5 Timeline:  

            Jan 29: Proposal
            Feb 5: Elevator pitch
            Feb 12: EDA and research on NN with audio data
            Feb 19: Preliminary network design
            Feb 26: Continue with network design
            March 4: Fine tuning network
            April 15: Fine tuning network
            April 22: Fine tuning network
            April 29: Poster session
            May 3: Video Presentation
            May 6: Final report due

            

## 6 Expected Number Students:  

            There will be two students on this project: Carrie Magee and Jack McMorrow 
            

## 7 Possible Issues:  

            Considering that the data is provided at a voluntary basis, this could prove to have some issues regarding representation 
            of various language groups. Therefore, the program may not perform equally on every individual voice. Also, there is an imbalance
            between languages in the dataset in terms of recorded hours. Consequently, we will need to determine which languages 
            have sufficient audio data to be effective for training the model. 
            


## Contact
- Author: Carrie Magee
- Email: [mageec@gwu.edu](Eamil)
- GitHub: [https://github.com/mageeca/24Spr_CMagee_LanguageDetection ](Git Hub rep)
