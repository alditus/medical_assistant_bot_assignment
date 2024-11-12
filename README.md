# Medical Question Answering System
This project involves the development of a medical question-answering system, implemented using two main approaches: Extractive Question Answering (QA) and Retrieval-Augmented Generation (RAG). The system is designed to answer user queries about medical conditions based on a given dataset of medical information.  Is far from being perfect and it needs further revision.

## Approach

### Data Preprocessing 
1. **Initial Data Cleaning**:[EDA](https://github.com/alditus/medical_assistant_bot_assignment/blob/main/notebooks/data_exploration_eda/exploratory_data_analysis.ipynb)  
   - Removed duplicate question-answer pairs.
   - Filtered out answers that were longer than the average or that didn’t fit the dataset distribution.
   - Removed question-answer pairs where the question was equal or very similar to the answer. For similarity score I used edit_distance
   - For a lot of examples, the question appeared at the beginning of the provided answer. I remove those questions from the answer.
2. **Keyword Identification and Clustering**:[EDA](https://github.com/alditus/medical_assistant_bot_assignment/blob/main/notebooks/data_exploration_eda/exploratory_data_analysis.ipynb)  
   - Extracted keywords from the dataset to perform topic clustering using KMeans and Agglomerative Clustering.
   - The clustering process helped group related questions by topic, allowing for topic-based separation of question-answer pairs.

### Model 1: Extractive Question Answering
1. **Challenges and Setup**:
   - Extractive QA requires context and the position of the answer within the context, which was not provided in the dataset.
   - Solution: Treated the original answers as context and used OpenAI's assistant to verify if the answer was contained within the context. Non-matching cases were marked as "no-answer."
2. **Dataset Splitting**:
   - The dataset was split into training, validation, and test sets based on clusters. Each split had distinct clusters (topics) to assess the model’s ability to generalize.
3. **Model Training**:
   - Used Hugging Face’s `distilbert/distilbert-base-uncased` model, I trained it for extractive QA task.
   - I ran the fine-tuning process for just 2-3 epochs due to time constrains and also I was coming back and forth to solve some issues.  
4. **Evaluation**:
   - Evaluated using the SQuAD metric, a standard for extractive QA tasks.
   - Results: The model achieved ~70% F1 score and ~40% exact match, providing a solid baseline for further improvements.
5. **Limitations**:
   - **Runtime Dependency**: This approach requires a context at runtime, which could either be retrieved from a database or provided by the user.

### Model 2: Retrieval-Augmented Generation (RAG) Question Answering
1. **Document Creation and System Setup**:
   - Each cluster identified during preprocessing was treated as a separate document in the knowledge base for retrieval.
2. **System Components**:
   - **Embedding Model**: Used `abhinand/MedEmbed-small-v0.1`, a medical embedding model from Hugging Face, optimized for the medical domain.
   - **Retrieval System**: Implemented with FAISS, a high-performance vector store, to manage document retrieval efficiently.
   - **Language Model (LLM)**: Integrated Amazon Bedrock’s Anthropic Foundational Model for answer generation.
3. **Evaluation**:
   - Due to time constraints, separate evaluations of the retrieval and generation components were not conducted. However, I planned to use the RAGAs (Retrieval-Augmented Generation Assessment) framework for future evaluations.

### Model Training and Testing Environment
All experiments were run locally using available resources and following guidelines in the Hugging Face documentation.

## Example Interactions
- **BERT model**
- **RAG system**


## Assumptions and Decisions
- **Data Assumptions**: The provided dataset answers were assumed to be correct, though some required interpretation to fit the extractive QA approach.
- **Dataset Adaptation**: As the dataset lacked contextual positioning, answers were repurposed as context, though this may not align perfectly with extractive QA requirements.
- **Embedding Choice**: Selected a medical embedding model (`abhinand/MedEmbed-small-v0.1`) for better alignment with the medical domain, improving retrieval relevance.

## Model Performance
### Extractive Question Answering
- **Strengths**: Established an initial extractive QA environment with reasonable performance (F1: 70%, EM: 40%) as a baseline.
- **Weaknesses**: Dependent on context availability at runtime, which may require an additional retrieval system.

### RAG Question Answering
- **Strengths**: Effective integration of a medical embedding model with FAISS and an LLM for answer generation.
- **Limitations**: Lack of retrieval and generation evaluations due to time constraints.

## Potential Improvements
1. **Enhanced Dataset Review**:
   - A more thorough examination of the dataset could improve answer accuracy, ensuring better alignment with questions.
2. **Augmented Medical Data**:
   - Adding more diverse and relevant medical data could enhance both the extractive QA and RAG approaches.
3. **Further Model Tuning**:
   - Experimenting with other pre-trained models and embeddings to further optimize performance for the medical QA task.
4. **Comprehensive RAG Evaluation**:
   - Utilize the RAGAs framework to assess retrieval and generation components separately for targeted improvements.
