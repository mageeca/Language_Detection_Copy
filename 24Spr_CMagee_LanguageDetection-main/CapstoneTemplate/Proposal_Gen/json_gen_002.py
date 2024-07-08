#%%
import json
import os
import shutil

#%%

def save_to_json(data, output_file_path):
    with open(output_file_path, 'w') as output_file:
        json.dump(data, output_file, indent=2)

semester2code = { "sp":"01", "spr":"01", "spring":"01", "su":"02", "sum":"02", "summer":"02", "fa":"03", "fall":"03"}
thisfilename = os.path.basename(__file__) # should match _ver for version, ideally 3-digit string starting as "000", up to "999"

data_to_save = \
    {
        # -----------------------------------------------------------------------------------------------------------------------
        "Version":
            """002""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Year":
            """2024""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Semester":
            """Spring""",
        # -----------------------------------------------------------------------------------------------------------------------
        "project_name":
            """Language Detection Using Audio Data""",
        # -----------------------------------------------------------------------------------------------------------------------
        "Objective":
            """ 
            The goal of this project is to use deep learning neural networks to develop a language detection program of audio 
            data. Using short audio clips from different languages, accents, and dialects, we believe it will be possible to 
            develop a model that will be able to distinguish the different languages in an efficient and precise manner. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            The dataset that we are using is called the Common Voice Corpus 16.1. This contains thousands of short clips of people 
            speaking in their respective languages on a voluntary basis. It contains over 3,438 hours of audio in 120 different 
            languages. Conveniently, the data is already located in Hugging Face which makes it much more accessible to us.

            The link to the original database can be found here: https://commonvoice.mozilla.org/en/datasets

            And the Hugging Face datasource can be found here: https://huggingface.co/datasets/mozilla-foundation/common_voice_16_1

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            Language barriers are often a common problem in communication. Whether it be for educational, business, travel, or 
            conversational reasons, not understanding each other's language can make it impossible to communicate effectively. 
            Technology can aid in this communication in various ways. The first step to facilitate effective communication across 
            languages is to identify what language is being spoken. This can be difficult considering different dialects and accents 
            across different languages. Having an effective program that will be able to understand the language through just a 
            short clip can initiate communication between people using different languages.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            We will begin this problem by accessing the dataset and doing some EDA. We found that many languages had little audio 
            data, so we decided to limit to languages with over 100 hours of data, which comes down to 43 languages. After the EDA 
            and preparing the data, we will begin with experimenting with different model architectures. Convolutional neural 
            networks are effective with image recognition problems, and we think they would be promising for a project like this 
            one, too. After selecting a model architecture we will be doing model tuning to perfect its performance with language 
            detection.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Timeline":
            """
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

            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            There will be two students on this project: Carrie Magee and Jack McMorrow 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Considering that the data is provided at a voluntary basis, this could prove to have some issues regarding representation 
            of various language groups. As we have already seen, some languages do not have a lot of samples, which we may decide to 
            leave out of the model. Additionally, the full dataset is very large, at around 80 gigabytes. Therefore, management and 
            storage of the data itself could prove to be a challenge.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Jack McMorrow",
        "Proposed by email": "jmcmorrow@gwu.edu",
        "instructor": "Edwin Lo",
        "instructor_email": "edwinlo@gwu.edu",
        "github_repo": "https://github.com/amir-jafari/Capstone",
        # -----------------------------------------------------------------------------------------------------------------------
    }
os.makedirs(
    os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}',
    exist_ok=True)
output_file_path = os.getcwd() + f'{os.sep}Proposals{os.sep}{data_to_save["Year"]}{semester2code[data_to_save["Semester"].lower()]}{os.sep}{data_to_save["Version"]}{os.sep}'
save_to_json(data_to_save, output_file_path + "input.json")
shutil.copy(thisfilename, output_file_path)
print(f"Data saved to {output_file_path}")
