
# Capstone Proposal
## Language Detection Using Audio Data
### Proposed by: Jack McMorrow
#### Email: jmcmorrow@gwu.edu
#### Advisor: Edwin Lo
#### The George Washington University, Washington DC  
#### Data Science Program


## 1 Objective:  
 
            The goal of this project is to use deep learning neural networks to develop a language detection program of audio 
            data. Using short audio clips from different languages, accents, and dialects, we believe it will be possible to 
            develop a model that will be able to distinguish the different languages in an efficient and precise manner. 
            

# ![Figure 1: Example figure](202401_002.png)
# *Figure 1: Caption*

## 2 Dataset:  

            The dataset that we are using is called the Common Voice Corpus 16.1. This contains thousands of short clips of people 
            speaking in their respective languages on a voluntary basis. It contains over 3,438 hours of audio in 120 different 
            languages. Conveniently, the data is already located in Hugging Face which makes it much more accessible to us.

            The link to the original database can be found here: https://commonvoice.mozilla.org/en/datasets

            And the Hugging Face datasource can be found here: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1

            

## 3 Rationale:  

            Language barriers are often a common problem in communication. Whether it be for educational, business, travel, or 
            conversational reasons, not understanding each other's language can make it impossible to communicate effectively. 
            Technology can aid in this communication in various ways. The first step to facilitate effective communication across 
            languages is to identify what language is being spoken. This can be difficult considering different dialects and accents 
            across different languages. Having an effective program that will be able to understand the language through just a 
            short clip can initiate communication between people using different languages.
            

## 4 Approach:  

            We will begin this problem by accessing the dataset and doing some EDA. We found that many languages had little audio 
            data, so we decided to limit to languages with over 100 hours of data, which comes down to 43 languages. After the EDA 
            and preparing the data, we will begin with experimenting with different model architectures. Convolutional neural 
            networks are effective with image recognition problems, and we think they would be promising for a project like this 
            one, too. After selecting a model architecture we will be doing model tuning to perfect its performance with language 
            detection.
            

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
            of various language groups. As we have already seen, some languages do not have a lot of samples, which we may decide to 
            leave out of the model. Additionally, the full dataset is very large, at around 80 gigabytes. Therefore, management and 
            storage of the data itself could prove to be a challenge.
            


## Contact
- Author: Edwin Lo
- Email: [edwinlo@gwu.edu](Eamil)
- GitHub: [https://github.com/amir-jafari/Capstone](Git Hub rep)
