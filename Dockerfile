FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

RUN mkdir -p /demo
RUN apt-get update && apt-get install -y wget
WORKDIR /demo

# Copy and install requirements
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && pip install --no-cache-dir -r requirements.txt
RUN pip install jupyter transformers

# Copy notebooks over
COPY . .

# Cache Llama-3.1-8B-Instruct
RUN --mount=type=secret,id=hub_token HUGGING_FACE_HUB_TOKEN=$(cat /run/secrets/hub_token) && huggingface-cli login --token $HUGGING_FACE_HUB_TOKEN
RUN python -c "from transformers import AutoModelForCausalLM, AutoTokenizer; AutoModelForCausalLM.from_pretrained('meta-llama/Llama-3.1-8B-Instruct'); AutoTokenizer.from_pretrained('meta-llama/Llama-3.1-8B-Instruct');"

# Download snowflake detector
WORKDIR ./notebooks
RUN mkdir -p models
WORKDIR /demo/notebooks/models
RUN wget https://huggingface.co/nvidia/NemoGuard-JailbreakDetect/blob/main/snowflake.pkl

WORKDIR /demo/notebooks

# Remove cached HF token
RUN rm /root/.cache/huggingface/token
RUN rm /root/.cache/huggingface/stored_tokens

EXPOSE 8888

CMD ["jupyter", "lab", "--NotebookApp.token", "''", "--NotebookApp.password", "'argon2:$argon2id$v=19$m=10240,t=10,p=8$dutey6IvrJac7c5LVbphwg$xU8zQJDzeePxdXPF2/XW+eADQWw25uruV6FlTFJC6KY'", "--allow-root", "--no-browser"]
