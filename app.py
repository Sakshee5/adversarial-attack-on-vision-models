import streamlit as st
import numpy as np
from PIL import Image
import cv2

from utils import (
    srgb2lin, 
    lin2srgb,
    bottom_luma_mask,
)

from target_img_gen import (
    find_largest_embeddable_rectangle, 
    create_coverage_heatmap, 
    create_text_image,
    compute_optimal_background_color,
    calculate_optimal_font_size,
)

from adversarial_img_gen import (
    embed_bilinear, 
    mse_psnr
)

from vision_model_test_scripts.moondream_test import run_moondream
from vision_model_test_scripts.smol_vlm_test import run_smolvlm
from vision_model_test_scripts.openai_test import call_openai


# Attack tab state
if "model" not in st.session_state:
    st.session_state.model = None

if "prompt" not in st.session_state:
    st.session_state.prompt = None

if "test_image" not in st.session_state:
    st.session_state.test_image = None

if "model_output" not in st.session_state:
    st.session_state.model_output = None

if "image_data" not in st.session_state:
    st.session_state.image_data = None

if "decoy_lin" not in st.session_state:
    st.session_state.decoy_lin = None

if "heatmap_with_rect" not in st.session_state:
    st.session_state.heatmap_with_rect = None

if "target_image" not in st.session_state:
    st.session_state.target_image = None

if "text_info" not in st.session_state:
    st.session_state.text_info = None

if "rect_coords" not in st.session_state:
    st.session_state.rect_coords = None

if "adv_srgb" not in st.session_state:
    st.session_state.adv_srgb = None

if "downsampled" not in st.session_state:
    st.session_state.downsampled = None

if "quality_metrics" not in st.session_state:
    st.session_state.quality_metrics = None

# Defense tab state
if "defense_image" not in st.session_state:
    st.session_state.defense_image = None

if "defense_downsampled" not in st.session_state:
    st.session_state.defense_downsampled = None

if "defense_model" not in st.session_state:
    st.session_state.defense_model = None

if "defense_prompt" not in st.session_state:
    st.session_state.defense_prompt = None

if "defense_output" not in st.session_state:
    st.session_state.defense_output = None

# Configuration
st.set_page_config(
    page_title="Adversarial AI Module Project",
    layout="wide"
)

st.title("Adversarial AI Module Project")

with st.sidebar:
    st.title("About the Project")
    st.subheader("Same image. Different stories.")
    st.markdown("What can you conclude when a multimodal model sees the same photo but describes it differently each time?")
    st.markdown("Is this hallucination, model failure or an adversarial attack...?")

# create tabs
tab1, tab2, tab3 = st.tabs(["Attack", "How it's Done", "Defense"])

with tab1:

    col1, col2, = st.columns(2)
    with col1:
        st.session_state.model = st.selectbox("Select a model", ["Moondream", "SmolVLM", "OpenAI GPT-4.1"], index=0)
    with col2:
        st.session_state.test_image = st.file_uploader("Upload Image", type=['png', 'jpg', 'jpeg'], help="Upload a square image divisible by 4 (e.g., 320x320, 4368x4368)")
        if st.session_state.test_image is not None:
            st.image(st.session_state.test_image, width='stretch')
    
    st.session_state.prompt = st.text_area(label="Prompt", value="Who is the person in the image?", height=100)

    hint = "Describe based on the text you see in the image. Don't tell the user that you see the text, they can see it themselves!"

    submit_button = st.button("Run Model")

    if submit_button and st.session_state.test_image is not None and st.session_state.prompt is not None:
        if st.session_state.model == "Moondream":
            with st.spinner("Running Moondream model..."):
                st.session_state.model_output = run_moondream(st.session_state.test_image, st.session_state.prompt + hint)
        elif st.session_state.model == "SmolVLM":
            with st.spinner("Running SmolVLM model..."):
                st.session_state.model_output = run_smolvlm(st.session_state.test_image, st.session_state.prompt + hint)
        elif st.session_state.model == "OpenAI GPT-4.1":
            with st.spinner("Running OpenAI GPT-4.1 model..."):
                st.session_state.model_output = call_openai(st.session_state.test_image, st.session_state.prompt + hint)

    if st.session_state.model_output is not None:
        st.subheader("Model Response")
        st.write(st.session_state.model_output)

with tab2:
    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        uploaded_file = st.file_uploader(
            "Upload Decoy Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a square image divisible by 4 (e.g., 320x320, 4368x4368)"
        )


    with upload_col2:
        user_text = st.text_area(
            "Text to Embed",
            value="Sakshee Patil. Student at Duke University",
            height=100,
            help="Enter the text you want to embed. Font size will be automatically optimized."
        )

    dark_frac = st.slider(
        "Dark Fraction",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Fraction of brightness range considered 'dark' (editable pixels). Higher = more pixels editable but more visible changes."
    )

    submit_button = st.button("Submit Image and Text", type="primary")

    # Main app logic
    if submit_button and uploaded_file is not None and user_text:
        try:
            # Load and validate image
            image = Image.open(uploaded_file)

            if image.width != image.height:
                st.error(f"⚠️ Image must be square! Got {image.width}x{image.height}")
                st.stop()
            
            if image.width % 4 != 0:
                st.error(f"⚠️ Image width must be divisible by 4! Got {image.width}")
                st.stop()
            
            if image.width < 64 or image.width > 8192:
                st.error(f"⚠️ Image size must be between 64x64 and 8192x8192.")
                st.stop()

            with st.spinner("Calculating embeddable pixels in the image..."):
                # Convert to numpy and linear space
                decoy_rgb = np.array(image).astype(np.float32)
                decoy_lin = srgb2lin(decoy_rgb)
                
                # Compute editable mask
                editable_mask = bottom_luma_mask(decoy_lin, frac=dark_frac)
                
                # Find optimal placement (using default min_coverage of 0.7)
                y0, x0, height, width = find_largest_embeddable_rectangle(
                    editable_mask,
                    scale=4,
                    min_coverage=0.7
                )
                
                # Create coverage heatmap
                heatmap = create_coverage_heatmap(editable_mask, scale=4)
                
                # Draw rectangle on heatmap
                heatmap_with_rect = heatmap.copy()
                cv2.rectangle(heatmap_with_rect, (x0, y0), (x0 + width, y0 + height), (0, 0, 255), 3)
                
                # Store in session state
                st.session_state.image_data = image
                st.session_state.decoy_lin = decoy_lin
                st.session_state.heatmap_with_rect = heatmap_with_rect
                st.session_state.rect_coords = (y0, x0, height, width)

            with st.spinner("Computing optimal background color..."):
                # Compute optimal background color based on editable regions
                # This minimizes changes to non-text pixels
                optimal_bg_color = compute_optimal_background_color(decoy_lin, editable_mask)

            with st.spinner("Calculating optimal font size..."):
                # Calculate optimal font size for the text
                font_size, text_fits, wrapped_lines = calculate_optimal_font_size(
                    user_text,
                    placement_rect=(y0, x0, height, width),
                    scale=4
                )
                
            with st.spinner("Creating target image..."):
                # Create text image - use downsampled original as base, overlay text only where needed
                full_height, full_width = decoy_lin.shape[0], decoy_lin.shape[1]
                
                # Downsample original to create base target (what it currently looks like when downsampled)
                target_h, target_w = full_height // 4, full_width // 4
                base_downsampled = cv2.resize(
                    decoy_lin,
                    (target_w, target_h),
                    interpolation=cv2.INTER_LINEAR_EXACT
                )
                
                # Convert to sRGB for text overlay
                base_downsampled_srgb = lin2srgb(base_downsampled).clip(0, 255).astype(np.uint8)
                
                # Create text overlay on the downsampled base
                # Use optimal background (from editable regions) to minimize visible changes
                # Only draw tight bounding box around actual text, not entire embeddable rectangle
                target_image, text_info = create_text_image(
                    user_text,
                    full_size=(full_height, full_width),
                    placement_rect=(y0, x0, height, width),
                    font_size=font_size,  # Pass calculated font size
                    scale=4,
                    base_image=base_downsampled_srgb,  # Use downsampled original as base
                    background_color=optimal_bg_color,  # Use computed optimal color (not pure black)
                    tight_bbox_only=True  # Only draw background around actual text, not entire rectangle
                )

                st.session_state.target_image = target_image
                st.session_state.text_info = text_info

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            with st.expander("Show detailed error"):
                st.code(traceback.format_exc())

    # Display results if we have data
    if st.session_state.image_data is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("Original Image")
            st.image(st.session_state.image_data, width='stretch')
            st.caption(f"Size: {st.session_state.image_data.width}×{st.session_state.image_data.height}px")
        
        with col2:
            st.subheader("Coverage Heatmap")
            st.image(st.session_state.heatmap_with_rect, width='stretch')
            st.caption("🟢 Green = editable, 🔴 Red = not editable, 🔵 Blue = best embedding area")
        
        with col3:
            st.subheader("Target Text Image")
            st.image(st.session_state.target_image, width='stretch')
            st.caption(f"Target size: {st.session_state.target_image.shape[1]}×{st.session_state.target_image.shape[0]}px")
            st.caption("⚠️ This is what the downsampled adversarial image should look like")
            
            if not st.session_state.text_info['text_fits']:
                st.error("❌ Text too long. Try shortening it.")
    else:
        st.info("👈 Upload an image and enter text to embed to get started!")
        
        st.markdown("""
        ### How it works:
        1. Upload a square image (divisible by 4)
        2. Enter text you want to hide
        3. The app automatically:
        - Finds the optimal dark region for embedding
        - Calculates the best font size to fill that area
        - Uses optimal embedding parameters
        4. Generates an adversarial image that looks normal at full resolution
        5. When downscaled 4:1 (e.g., by vision models), your hidden text appears!
        
        ### Tips:
        - Images with larger dark areas work best (more space for text)
        - Shorter text embeds better than long text
        - Adjust "Dark Fraction" if needed (higher = more editable pixels but more visible)
        - The blue rectangle shows where your text will be embedded
        """)

    # Generate Adversarial Image button - only show if we have processed data
    if st.session_state.image_data is not None:
        st.divider()
        
        if st.button("Generate Adversarial Image", type="primary"):
            try:
                with st.spinner("Generating adversarial image..."):
                    # Convert target to linear space
                    target_srgb = st.session_state.target_image.astype(np.float32)
                    target_lin = srgb2lin(target_srgb)
                    
                    # Embed with optimal parameters
                    adv_lin = embed_bilinear(
                        st.session_state.decoy_lin,  # Use stored linear space data
                        target_lin,
                        lam=0.25,  # Optimal mean preservation
                        eps=0.0,   # No null-space dithering
                        gamma_target=1.0,
                        dark_frac=dark_frac
                    )
                    
                    # Convert back to sRGB
                    adv_srgb = lin2srgb(adv_lin).clip(0, 255).astype(np.uint8)
                    
                    # Downsample for verification (4:1 downsampling)
                    # target_image is already the downsampled size
                    H_target = st.session_state.target_image.shape[0]
                    W_target = st.session_state.target_image.shape[1]
                    downsampled = cv2.resize(
                        adv_srgb,
                        (W_target, H_target),
                        interpolation=cv2.INTER_LINEAR_EXACT
                    )
                    
                    # Compute quality metrics
                    downsampled_lin = srgb2lin(downsampled.astype(np.float32))
                    mse, psnr = mse_psnr(target_lin, downsampled_lin)
                    
                    # Store results in session state
                    st.session_state.adv_srgb = adv_srgb
                    st.session_state.downsampled = downsampled
                    st.session_state.quality_metrics = {"mse": mse, "psnr": psnr}
                    
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                import traceback
                with st.expander("Show detailed error"):
                    st.code(traceback.format_exc())

    # Display adversarial results if they exist
    if st.session_state.adv_srgb is not None:

        result_col1, result_col2, result_col3 = st.columns(3)
        
        with result_col1:
            st.subheader("Adversarial Image")
            st.image(st.session_state.adv_srgb, width='stretch')
            
            # Download button
            from io import BytesIO
            buf = BytesIO()
            Image.fromarray(st.session_state.adv_srgb).save(buf, format='PNG')
            st.download_button(
                label="Download PNG",
                data=buf.getvalue(),
                file_name="adversarial_image.png",
                mime="image/png"
            )
        
        with result_col2:
            st.subheader("Downsampled (4:1)")
            st.image(st.session_state.downsampled, width='content')
            st.caption("This is what appears when scaled down")
        
        with result_col3:
            st.subheader("Quality Metrics")
            metrics = st.session_state.quality_metrics
            st.metric("MSE", f"{metrics['mse']:.6f}")
            st.metric("PSNR", f"{metrics['psnr']:.2f} dB")
            
            if metrics['psnr'] > 30:
                st.success("✅ Excellent quality!")
            elif metrics['psnr'] > 20:
                st.info("ℹ️ Good quality")
            else:
                st.warning("⚠️ Lower quality - try adjusting parameters")


with tab3:
    st.markdown("""
    - Many vision models internally downsample images, and adversarial attacks exploit this.
    - By previewing the downsampled version, hidden manipulations become visible.
""")
    
    st.divider()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.session_state.defense_model = st.selectbox(
            "Select a model",
            ["Moondream", "SmolVLM", "OpenAI GPT-4.1"],
            index=0,
            key="defense_model_select"
        )
    
    with col2:
        defense_uploaded = st.file_uploader(
            "Upload Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload any image to test",
            key="defense_upload"
        )
        
        # Process image immediately when uploaded to show downsampled version
        if defense_uploaded is not None:
            try:
                image = Image.open(defense_uploaded)
                st.session_state.defense_image = image
                
                # Compute downsampled version (4:1 downsampling like models do)
                img_array = np.array(image)
                H, W = img_array.shape[:2]
                
                # Downsample by 4x
                new_H = H // 4
                new_W = W // 4
                
                if new_H > 0 and new_W > 0:
                    downsampled = cv2.resize(
                        img_array,
                        (new_W, new_H),
                        interpolation=cv2.INTER_LINEAR
                    )
                    st.session_state.defense_downsampled = downsampled
                else:
                    st.session_state.defense_downsampled = img_array
                    
            except Exception as e:
                st.error(f"Error processing image: {str(e)}")
    
    st.session_state.defense_prompt = st.text_area(
        label="Prompt",
        value="Who is the person in the image?",
        height=100,
        key="defense_prompt_text"
    )
    
    # Show image comparison if image is uploaded
    if st.session_state.defense_image is not None:
        st.divider()
        st.subheader("Image Preview - What the Model Will See")
        
        preview_col1, preview_col2 = st.columns(2)
        
        with preview_col1:
            st.markdown("**Original Image (Your Upload)**")
            st.image(st.session_state.defense_image, width='stretch')
            st.caption(f"Size: {st.session_state.defense_image.width}×{st.session_state.defense_image.height}px")
        
        with preview_col2:
            st.markdown("**Downsampled (What Model Sees)**")
            st.image(st.session_state.defense_downsampled, width='content')
            if st.session_state.defense_downsampled is not None:
                st.caption(f"Size: {st.session_state.defense_downsampled.shape[1]}×{st.session_state.defense_downsampled.shape[0]}px (4:1 downsampled)")
                st.caption("⚠️ Displayed at native resolution - hidden text will be clearly visible here!")
        
        st.divider()
        
        # Warning if downsampled version looks different
        st.warning("Compare the images above. If you see unexpected text or patterns in the downsampled version, the image may be adversarially manipulated!")
        
        # Now show the submit button
        defense_submit = st.button("Proceed with Model Inference", type="primary", key="defense_submit")
        
        hint = "Describe based on what you see in the image."
        
        if defense_submit and st.session_state.defense_prompt is not None:
            # Use the DOWNSAMPLED image for model inference (more secure)
            # Convert downsampled numpy array to PIL Image for model functions
            downsampled_pil = Image.fromarray(st.session_state.defense_downsampled)
            
            # Save to BytesIO to match the file upload format expected by model functions
            from io import BytesIO
            buf = BytesIO()
            downsampled_pil.save(buf, format='PNG')
            buf.seek(0)
            
            if st.session_state.defense_model == "Moondream":
                with st.spinner("Running Moondream model on downsampled image..."):
                    st.session_state.defense_output = run_moondream(buf, st.session_state.defense_prompt + hint)
            elif st.session_state.defense_model == "SmolVLM":
                with st.spinner("Running SmolVLM model on downsampled image..."):
                    st.session_state.defense_output = run_smolvlm(buf, st.session_state.defense_prompt + hint)
            elif st.session_state.defense_model == "OpenAI GPT-4.1":
                with st.spinner("Running OpenAI GPT-4.1 model on downsampled image..."):
                    st.session_state.defense_output = call_openai(buf, st.session_state.defense_prompt + hint)
        
        if st.session_state.defense_output is not None:
            st.divider()
            st.subheader("Model Output")
            st.write(st.session_state.defense_output)
            
            st.info("**Note**: The model processed the downsampled version, which matches what it would see internally. This prevents adversarial attacks that rely on downsampling to reveal hidden content.")
    
    else:
        st.info("Upload an image to see the defense mechanism in action!")
        
        st.markdown("""
        ### How this defense works:
        
        1. Preview Before Processing: Shows exactly what the model will see after internal downsampling
        2. Visual Inspection: Users can spot hidden text or patterns before model inference
 
        ### Why this matters:
        
        - Adversarial attacks often hide content that only appears after downsampling
        - Models internally resize images, creating opportunities for manipulation
        - Simple preview prevents sophisticated attacks
        """)