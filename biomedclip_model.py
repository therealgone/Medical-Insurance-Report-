import torch
import open_clip
from PIL import Image

# Load BiomedCLIP model
model, _, preprocess = open_clip.create_model_and_transforms(
    'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
)
tokenizer = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Medical condition prompts
medical_prompts = [
    "chest x-ray",
    "brain CT scan",
    "abdominal CT scan",
    "lung CT scan",
    "spine CT scan",
    "pelvis CT scan",
    "cardiac CT scan",
    "COVID-19 CT scan",
    "tumor in CT scan",
    "brain tumor",
    "brain hemorrhage",
    "stroke in brain",
    "pneumonia in chest x-ray",
    "lung cancer",
    "tuberculosis",
    "fracture in spine",
    "kidney stone in CT scan",
    "liver lesion",
    "aortic aneurysm",
    "pleural effusion",
    "fluid in lungs",
    "blood clot in brain",
    "cyst in brain",
    "multiple sclerosis",
    "abdominal infection"
]

def analyze_image_and_send_to_gemini(image: Image.Image, model_gemini) -> str:
    # Step 1: Run BiomedCLIP
    inputs = preprocess(image).unsqueeze(0).to(device)
    texts = tokenizer(medical_prompts).to(device)

    with torch.no_grad():
        image_features = model.encode_image(inputs)
        text_features = model.encode_text(texts)

        image_features /= image_features.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        similarity_scores = (image_features @ text_features.T).squeeze()

    # Step 2: Get top result
    top_index = similarity_scores.argmax().item()
    top_label = medical_prompts[top_index]
    top_score = similarity_scores[top_index].item()

    # Step 3: If the confidence is low, return "Unknown"
    if top_score < 0.3:
        top_label = "Unknown scan type"
        top_score = 0.0  # Optional: Set score to 0 to show no clear match

    # Step 4: Send to Gemini for detailed analysis
    prompt = f'''
You are a smart AI working for a health insurance company in India.

A medical scan was identified by a BiomedCLIP AI model as:

**{top_label.upper()}**
Confidence score: **{top_score:.2f}**

Tasks:
1. Explain what this scan is and what it diagnoses.
2. Mention common conditions or abnormalities related to it.
3. Identify insurance implications: hospitalization, pre-existing, high risk?
4. Suggest next medical steps or tests.
5. Check if conditions could be family-linked or hereditary.
6. Red flag check: Is this a serious or suspicious finding?

Format it in a clear, structured, professional insurance-style report.
'''

    gemini_response = model_gemini.generate_content(prompt)
    return gemini_response.text
