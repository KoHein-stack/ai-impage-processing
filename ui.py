import streamlit as st
from PIL import Image
import numpy 
import tempfile
import cv2
import os
import ocr  # Assuming this is your OCR module
import model
from image_captioning import image_captionings  # Assuming this is your image captioning module
from bg_remove import remove_background  # Assuming this is your background removal function
import os
import streamlit as st
import time
from datetime import datetime
import numpy 
import psutil
import torch
import cpuinfo  # For detailed CPU info
import streamlit as st
import gc  # For garbage collection

def system_monitor():
    """Display comprehensive system specs in sidebar with custom font sizes"""
    # Custom CSS to make metric values smaller
    st.markdown("""
    <style>
    div[data-testid="stMetricValue"] > div {
        font-size: 14px !important;
    }
    </style>
    """, unsafe_allow_html=True)
    """Enhanced system monitor with refresh capability"""
    # with st.sidebar.expander("System Resources", expanded=True):
    #     # Add refresh button at the top
    #     refresh = st.button("Refresh", key="sys_refresh")
        
    #     # CPU Section with better measurement
    #     cpu_info = cpuinfo.get_cpu_info()
    #     cpu_percent = psutil.cpu_percent(interval=0.5)  # More responsive measurement
        
    #     st.markdown("**CPU**")
    #     col1, col2 = st.columns([1, 2])
    #     with col1:
    #         st.metric("Usage", f"{cpu_percent}%")
    #     with col2:
    #         st.caption(f"{cpu_info['brand_raw']}")
    #         st.progress(cpu_percent / 100)
   
    #     # Enhanced RAM Section
    #     ram = psutil.virtual_memory()
    #     st.markdown("**RAM**")
    #     col1, col2 = st.columns([1, 2])
    #     with col1:
    #         st.metric("Used", f"{ram.used/1e9:.1f} GB")
    #     with col2:
    #         st.caption(f"of {ram.total/1e9:.1f} GB ({ram.percent}%)")
    #         st.progress(ram.percent/100)
        
        
    with st.sidebar.expander("System Resources", expanded=True):
         # Add refresh button at the top
        refresh = st.button("Refresh", key="sys_refresh")
        # CPU Section
        cpu_info = cpuinfo.get_cpu_info()
        cpu_percent = psutil.cpu_percent(interval=1)  # Add interval to get accurate value

        st.markdown("**CPU**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Usage", f"{cpu_percent}%")
        with col2:
            st.caption(f"{cpu_info['brand_raw']}")
            st.progress(cpu_percent / 100)
   
        # RAM Section
        ram = psutil.virtual_memory()
        st.markdown("**RAM**")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.metric("Used", f"{ram.used/1e9:.1f} GB")
        with col2:
            st.caption(f"of {ram.total/1e9:.1f} GB")
            st.progress(ram.percent/100)
        
        # GPU Section
        st.markdown("**GPU**")
        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            gpu_mem = torch.cuda.get_device_properties(device).total_memory/1e9
            gpu_used = torch.cuda.memory_allocated(device)/1e9
            gpu_free = torch.cuda.memory_reserved(device)/1e9
            col1, col2 = st.columns([1, 2])
            with col1:
                st.metric("Used", f"{gpu_used:.1f} GB")
            with col2:
                percent_used = (gpu_used/gpu_mem)*100
                st.caption(f"of {gpu_mem:.1f} GB ({percent_used:.1f}%)")
                st.progress((gpu_used/gpu_mem))
            st.caption(torch.cuda.get_device_name(device))
             # Additional GPU details
            st.markdown("<details style='margin-bottom: 1rem'><summary>GPU Details</summary>"
                   f"<ul>"
                   f"<li><b>Device:</b> {torch.cuda.get_device_name(device)}</li>"
                   f"<li><b>Free:</b> {gpu_mem-gpu_used:.1f} GB</li>"
                   f"<li><b>Reserved:</b> {gpu_free:.1f} GB</li>"
                   f"<li><b>Driver:</b> {torch.version.cuda}</li>"
                   f"</ul></details>", 
                   unsafe_allow_html=True)
        else:
            st.warning("Not available")

        # Force refresh if button clicked
        if refresh:
            # st.rerun()

            gc.collect()               # Clear RAM unused objects
            if torch.cuda.is_available():
                torch.cuda.empty_cache()  # Clear GPU cache
# # Usage (call this function where you want the monitor)




def main():
    st.set_page_config(layout="wide")  # Use wider screen space
    
    # Custom CSS for better styling
    st.markdown("""
    <style>
    .main {
        max-width: 1200px;
    }
    .stRadio > div {
        flex-direction: row !important;
    }
    .stRadio [role="radiogroup"] {
        gap: 1rem;
    }
    .stButton button {
        width: 100%;
        transition: all 0.2s;
    }
    .stButton button:hover {
        transform: scale(1.02);
    }
    .result-column {
        border-radius: 10px;
        padding: 1rem;
        background: #f0f2f6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.title("AI Image Processing Suite")
    st.caption("Extract text, segment objects, or generate captions using advanced AI models")
    
    # System monitor in sidebar
    with st.sidebar:
        st.header("System Resources")
        system_monitor()
        st.markdown("---")
        st.caption("Model loading happens automatically on first run")

    # Model loading section (unchanged logic)
    if 'models' not in st.session_state:
        # [Keep all existing model loading code exactly as is]
                # First-time loading visual flow
        if 'first_load' not in st.session_state:
            loading_placeholder = st.empty()
            
            with loading_placeholder.container():
                with st.spinner("Loading AI models for the first time..."):
                    st.session_state.yolo, st.session_state.ocr, st.session_state.processor, st.session_state.caption_model  = model.load_models()
                    st.success("YOLO model loaded")
                    st.success("EasyOCR initialized")
                    st.success("BLIP model initialized")
                
                time.sleep(2)  # Show messages for 2 seconds
            
            loading_placeholder.empty()
            st.session_state.first_load = False  # Mark first load complete
            st.session_state.models_loaded = True
            
            # Small persistent indicator
            st.toast("Models ready!")
        
        # Subsequent loads will skip the visual flow
        else:
            st.session_state.yolo, st.session_state.ocr, st.session_state.processor, st.session_state.caption_model = model.load_models()
            st.session_state.models_loaded = True
    
    
    # Main content area
    st.subheader("Upload & Process Images")
    
    # File uploader with better styling
    uploaded_file = st.file_uploader(
        "Drag and drop or browse files", 
        type=["jpg", "png", "jpeg"],
        help="Supported formats: JPG, PNG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display original image with better layout
        st.markdown("### Original Image")
        original_col1, original_col2 = st.columns([2, 1])
        with original_col1:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=True)
        
        with original_col2:
            st.markdown("**Image Details**")
            st.write(f"Format: {image.format}")
            st.write(f"Size: {image.size[0]} x {image.size[1]} pixels")
            st.write(f"Mode: {image.mode}")
        
        # Processing options with cards-like UI
        st.markdown("---")
        st.markdown("### Processing Options")
        
        # Create temp file (unchanged)
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            temp_path = tmp_file.name
        
        # Processing options as tabs for better UX
        tab1, tab2, tab3 = st.tabs([
            "Text Extraction (OCR)", 
            "Object Segmentation", 
            "Image Captioning"
        ])
        
        with tab1:
            if st.button("Extract Text", key="ocr_btn", type="primary"):
                with st.spinner("Scanning for text..."):
                    # [Keep existing OCR processing code]
                    # Display results in card-like containers
                    image_rgb, extracted_text, boxes = ocr.process_image_for_ocr(temp_path, st.session_state.ocr)
                    st.markdown("---")
                    st.markdown("### OCR Results")
                    
                    result_col1, result_col2 = st.columns(2)
                    with result_col1:
                        with st.container(border=True):
                            st.markdown("**Extracted Text**")
                            if extracted_text:
                                st.code("\n".join(extracted_text), language="text")
                            else:
                                st.warning("No text detected")
                    
                    with result_col2:
                        with st.container(border=True):
                            st.markdown("**Detection Visualization**")
                            if boxes:
                                st.image(image_rgb, use_container_width=True)
                            else:
                                st.info("No text regions found")
        
        with tab2:
            if st.button("Segment Objects", key="seg_btn", type="primary"):
                with st.spinner("Identifying objects..."):
                    segmented_img, object_names, original_rgb = remove_background(temp_path, st.session_state.yolo)

                    st.markdown("---")
                    st.markdown("### Segmentation Results")

                    seg_col1, seg_col2 = st.columns(2)
                    with seg_col1:
                        with st.container(border=True):
                            st.markdown("**Detected Objects**")
                            if object_names:
                                st.json(object_names)
                            else:
                                st.warning("No objects detected")

                    with seg_col2:
                        with st.container(border=True):
                            st.markdown("**Segmented Image**")
                            if segmented_img is not None:
                                pil_image = Image.fromarray(cv2.cvtColor(segmented_img, cv2.COLOR_BGRA2RGBA))
                                st.image(pil_image, use_container_width=True)
                            else:
                                st.error("Segmentation failed")

        with tab3:
            if st.button("Generate Caption", key="caption_btn", type="primary"):
                with st.spinner("Analyzing image content..."):
                    # [Keep existing captioning code]
                    caption= image_captionings(temp_path, st.session_state.processor, st.session_state.caption_model)
                    
                    st.markdown("---")
                    st.markdown("### Captioning Results")
                    
                    caption_col1, caption_col2 = st.columns([1, 2])
                    with caption_col1:
                        with st.container(border=True):
                            st.markdown("**Generated Description**")
                            if caption:
                                st.success(caption)
                                st.download_button(
                                    "Download Caption",
                                    data=caption,
                                    file_name="image_caption.txt",
                                    mime="text/plain"
                                )
                            else:
                                st.warning("No caption generated")
                    
                    with caption_col2:
                        with st.container(border=True):
                            st.markdown("**Processed Image**")
                            try:
                                pil_image = Image.fromarray(cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGBA))
                                st.image(pil_image, use_container_width=True)
                            except:
                                pil_image = Image.fromarray(cv2.cvtColor(cv2.imread(temp_path), cv2.COLOR_BGR2RGBA))
                                st.image(pil_image, use_container_width=True)
        
        # Clean up (unchanged)
        try:
            os.unlink(temp_path)
        except:
            pass

if __name__ == "__main__":
    main()

