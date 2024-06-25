
# GUVI GPT Model Deployment using Hugging Face
# Hugging face LLM space link:[madmi/GVUI_final_project](https://huggingface.co/spaces/madmi/GVUI_final_project)
## Project Overview
This project involves deploying a fine-tuned GPT model, trained on GUVI’s company data, using Hugging Face services. A scalable and secure web application is created using Streamlit or Gradio to make the model accessible to users over the internet. The deployment leverages Hugging Face Spaces resources and uses a database to store usernames and login times.

## Skills Takeaway
- Deep Learning
- Transformers
- Hugging Face Models
- Large Language Models (LLM)
- Streamlit or Gradio
- AIOps

## Objective
Deploy a pre-trained or fine-tuned GPT model using Hugging Face Spaces, making it accessible through a web application built with Streamlit or Gradio.

### Steps to Fine-Tune the Model

#### 1. Data Collection or Extraction
- Gather text data from various sources within GUVI, such as website content, user queries, social media, blog posts, and training materials.

#### 2. Data Preparation
- Clean and preprocess the text data, ensuring it is in a format suitable for training (e.g., removing special characters, normalizing text).

#### 3. Tokenization
- Use the GPT-2 tokenizer to convert the text data into tokens. Ensure the data is tokenized consistently to match the pre-trained model’s requirements.

#### 4. Fine-Tuning
- Use the Hugging Face Transformers library or similar tools to fine-tune the GPT-2 model on the prepared dataset.
- Monitor the training process to prevent overfitting and ensure the model generalizes well to new data.

## Results
- **Functional Web Application**: A fully functional web application that users can access to interact with the pre-trained GPT model.
- **Scalable Deployment**: A scalable deployment framework using Hugging Face services.
- **Documentation**: Comprehensive documentation outlining setup, deployment, and usage instructions.

## Detailed Steps and Commands

### Data Preparation

1. **Store `app.py` in S3**:
   - Upload the `app.py` file and other necessary files to an Amazon S3 bucket.

### Infrastructure Setup

2. **Launch EC2 Instance**:
   - Go to the AWS Management Console.
   - Launch a new EC2 instance with appropriate IAM roles and security groups.
   - Ensure the instance has internet access via an Internet Gateway.

### Environment Configuration

3. **Install Required Packages**:
   - SSH into your EC2 instance.
   - Install required packages:
     ```sh
     sudo apt update
     sudo apt install python3-pip
     pip3 install streamlit boto3 transformers torch
     ```

4. **Download `app.py` from S3**:
   - Use AWS CLI to download files from S3:
     ```sh
     aws s3 cp s3://your-bucket-name/app.py .
     ```

### Application Deployment

5. **Run the Streamlit Application**:
   - Execute the Streamlit app:
     ```sh
     streamlit run app.py
     ```

6. **Temporary Public Access with ngrok** (optional):
   - Install ngrok:
     ```sh
     wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
     unzip ngrok-stable-linux-amd64.zip
     ```
   - Run ngrok to create a public URL:
     ```sh
     ./ngrok http 8501
     ```

### Security Configuration

7. **Configure Security Group**:
   - Ensure the security group associated with your EC2 instance allows inbound traffic on port 8501.

### Fine-Tuning the Model

1. **Gather Data**:
   - Collect text data from GUVI’s resources.

2. **Clean and Preprocess Data**:
   - Clean the text data:
     ```python
     import re
     def preprocess(text):
         text = text.lower()
         text = re.sub(r'\W', ' ', text)
         text = re.sub(r'\s+', ' ', text)
         return text
     ```

3. **Tokenization**:
   - Tokenize the data using GPT-2 tokenizer:
     ```python
     from transformers import GPT2Tokenizer
     tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
     tokens = tokenizer.batch_encode_plus(data, return_tensors='pt', padding=True, truncation=True)
     ```

4. **Fine-Tuning**:
   - Fine-tune the GPT-2 model:
     ```python
     from transformers import GPT2LMHeadModel, Trainer, TrainingArguments
     model = GPT2LMHeadModel.from_pretrained('gpt2')

     training_args = TrainingArguments(
         output_dir='./results',
         num_train_epochs=3,
         per_device_train_batch_size=4,
         save_steps=10_000,
         save_total_limit=2,
     )

     trainer = Trainer(
         model=model,
         args=training_args,
         train_dataset=tokens,
     )

     trainer.train()
     ```

## Documentation
Provide comprehensive documentation outlining setup, deployment, and usage instructions. Include details on:
- How to set up and configure the AWS environment.
- Steps to deploy and run the Streamlit application.
- Instructions for fine-tuning the model and preparing the data.
- Usage instructions for the deployed web application.

## Conclusion
By following the above steps, you will successfully deploy a fine-tuned GPT model using Hugging Face services and make it accessible through a secure and scalable web application built with Streamlit or Gradio. This project will enhance your skills in deep learning, transformers, LLMs, and web application deployment in the AIOps domain.
