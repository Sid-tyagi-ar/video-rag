import streamlit as st
import os
import shutil
import atexit
from .Helper_function import Video_processor, VideoRAGChunker, VideoRAGRetriever  
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# Cache video processing
@st.cache_data
def cached_process_video(url, frame_skip, gemini_api_key):
    processor = Video_processor(output_dir="downloads", keyframe_dir="keyframes")
    return processor.process_video(url, frame_skip=frame_skip, gemini_api_key=gemini_api_key)

# Cleanup on exit
def cleanup():
    retriever = VideoRAGRetriever()
    retriever.cleanup_namespace("test_session")
    for dir in ["downloads", "keyframes"]:
        if os.path.exists(dir):
            shutil.rmtree(dir)
    logging.info("Cleaned up files and Pinecone namespace.")

atexit.register(cleanup)

# Streamlit app
st.title("Video RAG Explorer")

# Center input bar with form
st.markdown("<h3 style='text-align: center;'>Enter YouTube URL</h3>", unsafe_allow_html=True)
with st.form(key="url_form"):
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        url = st.text_input("", "https://www.youtube.com/watch?v=ftDsSB3F5kg", label_visibility="collapsed")
        submit_button = st.form_submit_button(label="Process Video")

# Advanced mode toggle (top-right)
with st.sidebar:
    advanced_mode = st.toggle("Advanced Mode", value=False)
    if advanced_mode:
        edge_depth = st.slider("Max Edge Depth", 1, 5, 3)
        bin_size = st.slider("Bin Size (seconds)", 5, 20, 10)

# Process video on submit
if submit_button:
    
    GEMINI_API_KEY = st.secrets["Gemini"]["api_key"]   # Replace with yours
    video_id = Video_processor().extract_video_id(url)
    
    # Pikachu transition
    pikachu_gif = "https://64.media.tumblr.com/tumblr_m4azcqZ9uV1qge5e6o1_500.gif"  # Running Pikachu
    with st.spinner("Processing..."):
        st.markdown(f"<div style='text-align: center;'><img src='{pikachu_gif}' width='200'></div>", unsafe_allow_html=True)
        result = cached_process_video(url, frame_skip=15, gemini_api_key=GEMINI_API_KEY)
    
    if result["status"] == "success":
        st.session_state["result"] = result
        st.session_state["video_id"] = video_id
        
        # Build RAG graph
        chunker = VideoRAGChunker(output_dir="downloads", bin_size=bin_size if advanced_mode else 10.0)
        bins = chunker.preprocess_transcript(result["transcript_file"])
        bins_with_keyframes = chunker.integrate_keyframes(bins, result["keyframe_descriptions"])
        nodes, edges = chunker.build_graph(bins_with_keyframes, video_id)
        video_metadata = {
            "title": "निर्देशक की भूमिका भाग - 1",
            "description": "For more information and related videos visit us on http://www.digitalgreen.org/",
            "uri": url
        }
        chunker.save_to_pinecone(nodes, edges, video_metadata, namespace="test_session")
        
        # Show keyframes
        st.write("### Keyframes")
        cols = st.columns(4)
        for i, (path, info) in enumerate(result["keyframe_descriptions"].items()):
            with cols[i % 4]:
                st.image(path, caption=f"{os.path.basename(path)} at {info['timestamp']:.1f}s")
        
        # Ready for query
        st.session_state["processed"] = True
        st.success("Video processed! Ready for your query.")
    else:
        st.error(f"Failed: {result['message']}")

# Query input after processing
if "processed" in st.session_state and st.session_state["processed"]:
    st.markdown("<h3 style='text-align: center;'>Ask About the Video</h3>", unsafe_allow_html=True)
    with st.form(key="query_form"):
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            query = st.text_input("", placeholder="e.g., Explain director responsibilities", label_visibility="collapsed")
            query_button = st.form_submit_button(label="Submit Query")
    
    if query_button and query:
        retriever = VideoRAGRetriever()
        max_edge_depth = edge_depth if advanced_mode else 3
        result = retriever.answer_query(query, st.session_state["video_id"], top_k=1, max_edge_depth=max_edge_depth, namespace="test_session")
        
        # Popup answer
        st.success(result["answer"])
        
        # Right-side blocks
        col1, col2 = st.columns([3, 1])
        with col2:
            with st.expander("Timestamps"):
                for chunk in result["retrieved_chunks"]:
                    ts = chunk["timestamp_coverage"][0]
                    st.write(f"Node {chunk['node_id']}: {ts['start']}–{ts['end']}s")
            with st.expander("Chunks Used"):
                for chunk in result["retrieved_chunks"]:
                    st.write(f"Node {chunk['node_id']}: {chunk['text'][:50]}... (Sim: {chunk['similarity']:.2f})")

# Cleanup reminder
st.sidebar.write("Note: Files and Pinecone data will be cleaned up when you exit the app.")
