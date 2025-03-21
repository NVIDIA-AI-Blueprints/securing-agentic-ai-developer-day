![NVIDIA Header](assets/header.png)
<h1><img align="center" src="https://github.com/user-attachments/assets/cbe0d62f-c856-4e0b-b3ee-6184b7c4d96f">Securing Agentic AI Developer Day</h1>

_This Developer Day was given in March 2025 during NVIDIA GTC_. 

AI Adoption is transforming industries, organizations and daily operations – learn how to bring security fundamentals to the next generation of agentic AI application and deploy with confidence.

In this developer day, learn how to break down an agentic AI workflow, the [AI Virtual Assistant NVIDIA Blueprint](https://build.nvidia.com/nvidia/ai-virtual-assistant-for-customer-service) into its core components and: 
- Analyze the blueprint for sample attacks and threats
- Identify general security mitigations
- Discover weakness in the LLM model with `garak`
- Apply guardrails to mitigate LLM-specific weaknesses with NeMo Guardrails

Access the notebooks through NVIDIA Brev: 

[![ Click here to deploy.](https://brev-assets.s3.us-west-1.amazonaws.com/nv-lb-dark.svg)](https://brev.nvidia.com/launchable/deploy?launchableID=env-2u8ZUGPOQ7zUX1t4z9pkxhlxoAY)

## Structure

### Notebooks
The developer day contains several Jupyter notebooks that demostrate the usage of `garak` and NeMo Guardrails to discover model weakness and apply mitigations. 

Follow this order for notebooks: 
- `setup.ipynb`: Initial setup and configuration of API tokens 
- `garak_demo.ipynb`: Demonstration of Garak for LLM security testing
- `guardrails_demo.ipynb`: Demonstration of NeMo Guardrails implementation
- `guardrails_garak_demo.ipynb`: Combined demo of Guardrails and Garak

## References
- [Securing Agentic AI Developer Day Slides](assets/slides.pdf)
- [GTC 2025 Recording](https://www.nvidia.com/gtc/session-catalog/?deeplink=audience-recommend--2&tab.catalogallsessionstab=16566177511100015Kus&search.pcybersecurityp=1736267392004001Amu5&search.pcybersecurityp=1699468149882001CDqA&search.pcybersecurityp=1699468149882002CSFV&search.suggestedaudiencelevel=1732117107498003nOoA#/session/1728679235868001mijf)

## Technologies Used 

### NVIDIA Blueprints
NVIDIA AI Blueprints are reference examples that illustrate how NVIDIA NIM and NVIDIA AI Enterprise software can be leveraged to build innovative solutions 
- [Try Blueprints Today](https://build.nvidia.com/blueprints)
 - [Build with Blueprints](https://github.com/NVIDIA-AI-Blueprints)

### NVIDIA NIM 
NVIDIA NIM is a set of accelerated inference microservices that allow organizations to run AI models on NVIDIA GPUs anywhere.
- [Try NIMs Today](https://catalog.ngc.nvidia.com/?filters=nvidia_nim|NVIDIA%20NIM|nimmcro_nvidia_nim,resourceType|Container|container)
- [Build with NIMs](https://build.nvidia.com/)

### NVIDIA Brev 
NVIDIA Brev provides streamlined access to NVIDIA GPU instances on popular cloud platforms, automatic environment setup, and flexible deployment options, enabling developers to start experimenting instantly.
- [Try Brev](https://developer.nvidia.com/brev)

### `garak` 
`garak` helps developers discover weaknesses and unwanted behaviors in anything using language model technology.
- [Try `garak`](https://github.com/NVIDIA/garak?tab=readme-ov-file)

### NeMo Guardrails
NeMo Guardrails is an open-source toolkit for easily adding programmable guardrails to LLM-based conversational applications.
- [Try Guardrails](https://github.com/NVIDIA/NeMo-Guardrails)
