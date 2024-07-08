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
            """001""",
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
            The objective of the project is to develop a language detection program using the Common Voice dataset. This dataset 
            contains a collection of speech recordings covering a range of languages, accents, and dialects. Using this extensive 
            dataset, our project aims to create a program that can accurately identify which language a human is speaking based on 
            various types of speech recordings. This type of project is crucial for facilitating communication across different 
            language barriers by being able to detect and transcribe various languages.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Dataset":
            """
            As mentioned above, we will be using data from the Common Voice dataset created by Mozilla. The dataset is publicly 
            available and contains audio recordings of various human voices. Recordings are done by volunteers and the database 
            contains recordings of over 50 languages. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Rationale":
            """
            By developing a language detection program, we hope to tackle linguistic barriers and promote effective communication 
            within our interconnected global communities. The first step to breaking this barrier is to develop a program that has 
            the ability to detect a language by speech alone. 
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Approach":
            """
            To approach a problem like this, we will have to use deep learning techniques such as neural networks to develop an 
            effective model. Convolutional neural networks, which are effective for image recognition problems, can also be very 
            effective when working with audio data. We believe that these types of networks will work well when classifying audio 
            data. We are also open to exploring different network structures. 
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
            April 15: Mock Presentation
            April 22: Mock Presentation
            April 29: Poster session
            May 3: Video Presentation
            May 6: Final report due
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Expected Number Students":
            """
            There will be 2 students working on this project: Carrie Magee and Jack McMorrow.  
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Possible Issues":
            """
            Considering the recordings are conducted on a voluntary basis, the dataset lacks representation of various language 
            groups. Therefore, the program may not perform equally on every individual voice. Other technical issues could include 
            the management and storage of the dataset.
            """,
        # -----------------------------------------------------------------------------------------------------------------------
        "Proposed by": "Jack McMorrow and Carrie Magee",
        "Proposed by email": "jmcmorrow@gwu.edu, mageec@gwu.edu",
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
