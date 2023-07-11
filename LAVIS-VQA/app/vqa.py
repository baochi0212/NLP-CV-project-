"""
 # Copyright (c) 2022, salesforce.com, inc.
 # All rights reserved.
 # SPDX-License-Identifier: BSD-3-Clause
 # For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import streamlit as st
from app import load_demo_image, device
from app.utils import load_model_cache
from lavis.processors import load_processor
from PIL import Image
from transformers import AutoProcessor, AutoModelForCausalLM
import torch


def app():
    model_type = st.sidebar.selectbox("Model:", ["BLIP", "PnP-VQA", "GIT"])

    # ===== layout =====
    st.markdown(
        "<h1 style='text-align: center;'>Visual Question Answering</h1>",
        unsafe_allow_html=True,
    )

    instructions = """Try the provided image or upload your own:"""
    file = st.file_uploader(instructions)

    col1, col2 = st.columns(2)

    col1.header("Image")
    if file:
        raw_img = Image.open(file).convert("RGB")
    else:
        raw_img = load_demo_image()

    w, h = raw_img.size
    scaling_factor = 720 / w
    resized_image = raw_img.resize((int(w * scaling_factor), int(h * scaling_factor)))

    col1.image(resized_image, use_column_width=True)
    col2.header("Question")

    user_question = col2.text_input("Input your question!", "What are objects there?")
    qa_button = st.button("Submit")

    col2.header("Answer")

    # ===== event =====

    if qa_button:
        if model_type.startswith("BLIP"):
            vis_processor = load_processor("blip_image_eval").build(image_size=480)
            text_processor = load_processor("blip_question").build()
            model = load_model_cache(
                "blip_vqa", model_type="vqav2", is_eval=True, device=device
            )

            img = vis_processor(raw_img).unsqueeze(0).to(device)
            question = text_processor(user_question)

            vqa_samples = {"image": img, "text_input": [question]}
            answers = model.predict_answers(vqa_samples, inference_method="generate")
        elif model_type.startswith("PnP-VQA"):
            vis_processor = load_processor("blip_image_eval").build(image_size=384)
            text_processor = load_processor("blip_caption").build()
            model = load_model_cache(
                "pnp_vqa", model_type="base", is_eval=True, device=device
            )

            img = vis_processor(raw_img).unsqueeze(0).to(device)
            question = text_processor(user_question)

            vqa_samples = {"image": img, "text_input": [question]}
            answers, _, _ = model.predict_answers(vqa_samples, num_captions=50, num_patches=20)
            # answers = answers[0]
        elif model_type.startswith("GIT"):
            processor = AutoProcessor.from_pretrained("microsoft/git-base-textvqa")
            model = AutoModelForCausalLM.from_pretrained("microsoft/git-base-textvqa")

            pixel_values = processor(images=raw_img, return_tensors="pt").pixel_values

            input_ids = processor(text=user_question, add_special_tokens=False).input_ids
            input_ids = [processor.tokenizer.cls_token_id] + input_ids
            input_ids = torch.tensor(input_ids).unsqueeze(0)

            generated_ids = model.generate(pixel_values=pixel_values, input_ids=input_ids, max_length=50)
            answer_ids = generated_ids[:, len(input_ids[0]):]
            answers = processor.batch_decode(generated_ids[:, len(input_ids[0]):], skip_special_tokens=True)

        col2.write("\n".join(answers))

