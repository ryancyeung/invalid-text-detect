# **InvalidTextDetect**

InvalidTextDetect is a Python script for detecting invalid text responses to open-ended or free text items. 

In brief, this a supervised machine learning approach that (a) trains, validates, and tests on a subset of texts manually coded as valid or invalid, (b) calculates performance metrics to help select the best model, and (c) predicts whether uncoded texts are valid or invalid based on the text alone. 

## **Gettting Started**

[Fork](https://docs.github.com/en/get-started/quickstart/fork-a-repo) and [clone](https://docs.github.com/en/get-started/quickstart/fork-a-repo#cloning-your-forked-repository) this repository to get a copy of this project locally on your machine.

## **Installation**

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install necessary libraries. For example:

```bash
pip install pandas
```


## **Usage**

### **Input Data**

Input data should be formatted into a single comma-separated values (CSV) file and placed into the input folder. Three columns are necessary in this file: 
1. unique identifiers for each text response
2. the original text responses
3. manually coded labels for the labelled subset (0 = *valid*, 1 = *invalid*, blank = *unlabelled*)

Headings for these columns (i.e., the first row) should be (1) `doc_id`, (2) `text`, and (3) `human_labelled`, respectively. As long as the three required columns are present (unique identifier, original text, manually coded labels), any number of additional columns can be included; extra columns will have no impact on running the code. As an example, see Table A1. In this mock dataset, text responses 1 and 2 were manually coded as valid or invalid, forming the labelled subset. Text responses 3 and 4 were not manually coded, forming the unlabelled subset.

**Table A1.** Example structure of the input data file.

| doc_id | text                                                                                                                        | human_labelled |
|--------|-----------------------------------------------------------------------------------------------------------------------------|----------------|
| 1      | The quick brown fox jumps over the lazy dog.                                                                                | 0              |
| 2      | Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. | 1              |
| 3      | Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat.                 |                |
| 4      | The five boxing wizards jump quickly.                                                                                       |                |


### **Running the Python Code**

[![Open in Visual Studio Code](https://open.vscode.dev/badges/open-in-vscode.svg)](https://open.vscode.dev/ryancyeung/invalid-text-detect)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ryancyeung/invalid-text-detect)

Our Python code is accessible as either a Python script (.py) or a Jupyter notebook file (.ipynb), both of which can be run locally or using services like Google Colab (with Python v3+). Based on one’s results from the model validation section of the code, the user should edit the model evaluation section appropriately. In other words, users should inspect the performance metrics from the model validation section (e.g., MCC, macro F1, precision, recall, accuracy) to decide which model performed best (e.g., highest mean MCC; [Chicco & Jurman, 2020](https://doi.org/10.1186/s12864-019-6413-7); [Luque et al., 2019](https://doi.org/10.1016/j.patcog.2019.02.023)) and therefore should be selected, though “best” is always dependent on the user’s own context. Once this best model is selected, users can edit the code to supply the name of the best model as an argument for the model evaluation and final predictions sections.

The only difference between the Python script (.py) and the Jupyter notebook (.ipynb) is that the Python script saves plots as image files (.png) by default. In the Jupyter notebook, plots can also be saved as such files by uncommenting the chunks containing the same code.


### **Output Data**

Output data are created in the output folder as the Python code runs. No files are necessary in the output folder to run the code; only the input data file in the input folder (e.g., `text_data.csv`) is needed to run the code. Our Python code is written to include the date and time that each output file was created in the output file names. In addition to helping with organization and version control, this helps prevent any accidental overwriting of output files.


## **Contributing**

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.


## **Authors and Acknowledgment**

Author: [Ryan Yeung](https://ryancyeung.github.io)

Adapted from [Make a README](https://www.makeareadme.com/) template and [Best-README-Template](https://github.com/othneildrew/Best-README-Template).


## **License**

Distributed under the [GNU GPLv3](https://www.gnu.org/licenses/gpl-3.0.en.html) license. See `LICENSE` for more information.
