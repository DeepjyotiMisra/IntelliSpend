"""IntelliSpend - High-End Transaction Categorization UI"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import sys
import time
import io
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from pipeline import process_transactions_file, process_single_transaction
from utils.feedback_helper import submit_feedback_from_result, load_categorized_transactions
from utils.feedback_processor import get_feedback_summary, update_merchant_seed_from_feedback
from agents.tools import get_feedback_statistics
from config.settings import settings
# Categories will be loaded from settings

# Page configuration
st.set_page_config(
    page_title="IntelliSpend - AI Transaction Categorization",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        color: #6c757d;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stButton>button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: 600;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #764ba2 0%, #667eea 100%);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2);
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        margin: 1rem 0;
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'processing_status' not in st.session_state:
    st.session_state.processing_status = None
if 'llm_provider' not in st.session_state:
    import os
    st.session_state.llm_provider = os.getenv('CLASSIFIER_MODEL_PROVIDER', 'gemini').lower()
if 'feedback_provider' not in st.session_state:
    import os
    st.session_state.feedback_provider = os.getenv('FEEDBACK_MODEL_PROVIDER', 'gemini').lower()

# Initialize session state for processing status
if 'is_processing' not in st.session_state:
    st.session_state.is_processing = False

# Initialize session state for single transaction feedback
if 'show_feedback_form' not in st.session_state:
    st.session_state.show_feedback_form = False
if 'last_classification_result' not in st.session_state:
    st.session_state.last_classification_result = None
if 'last_classification_input' not in st.session_state:
    st.session_state.last_classification_input = None


def main():
    """Main application"""
    
    # Header
    st.markdown('<h1 class="main-header">💰 IntelliSpend</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-Powered Transaction Categorization with Continuous Learning</p>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Navigation")
        
        # Store current page in session state to preserve it
        if 'current_page' not in st.session_state:
            st.session_state.current_page = "📊 Process Transactions"
        
        # If processing is active, lock to Process Transactions page and disable navigation
        if st.session_state.is_processing:
            page = "📊 Process Transactions"
            st.session_state.current_page = page
            # Show disabled radio with Process Transactions selected
            st.radio(
                "Choose a page:",
                ["🏠 Dashboard", "📊 Process Transactions", "✏️ Review & Feedback", "📈 Analytics", "⚙️ Configuration", "🔄 Apply Feedback", "🏷️ Custom Categories", "📚 Merchant Seed"],
                label_visibility="collapsed",
                disabled=True,
                index=1  # Process Transactions is index 1
            )
            st.warning("⚠️ **Processing in progress. Navigation disabled.**")
        else:
            # Normal navigation when not processing
            # Use the radio button value directly and sync with current_page
            pages = ["🏠 Dashboard", "📊 Process Transactions", "✏️ Review & Feedback", "📈 Analytics", "⚙️ Configuration", "🔄 Apply Feedback", "🏷️ Custom Categories", "📚 Merchant Seed"]
            
            # Use key-based state management - Streamlit will handle the state automatically
            # Initialize the radio state if not exists
            if "navigation_radio" not in st.session_state:
                st.session_state.navigation_radio = st.session_state.current_page
            
            # Get page from radio button (uses key-based state)
            page = st.radio(
                "Choose a page:",
                pages,
                label_visibility="collapsed",
                key="navigation_radio"
            )
            
            # Sync current_page with radio selection
            st.session_state.current_page = page
        
        st.divider()
        
        st.header("ℹ️ System Status")
        
        # Check FAISS index
        index_path = Path(settings.FAISS_INDEX_PATH)
        if index_path.exists():
            st.success("✅ FAISS Index Ready")
        else:
            st.error("❌ FAISS Index Missing")
            st.info("Run: `python utils/faiss_index_builder.py`")
        
        # Check merchant seed
        seed_path = Path(settings.MERCHANT_SEED_PATH)
        if seed_path.exists():
            try:
                seed_df = pd.read_csv(seed_path)
                st.metric("Merchant Seed Records", len(seed_df))
            except:
                st.info("Merchant seed exists")
        else:
            st.warning("⚠️ Merchant Seed Missing")
        
        # Feedback status
        try:
            summary = get_feedback_summary()
            st.metric("Pending Feedback", summary.get('pending', 0))
        except:
            st.info("No feedback data")
        
        st.divider()
        
        # Show current LLM providers
        st.header("🤖 Active LLM Providers")
        st.write(f"**Classifier:** {st.session_state.llm_provider.upper()}")
        st.write(f"**Feedback:** {st.session_state.feedback_provider.upper()}")
        st.caption("Change in Configuration page")
    
    # Force page to Process Transactions if processing is active (safety check)
    if st.session_state.is_processing and page != "📊 Process Transactions":
        page = "📊 Process Transactions"
        st.session_state.current_page = page
        st.rerun()
    
    # Route to appropriate page
    if page == "🏠 Dashboard":
        show_dashboard()
    elif page == "📊 Process Transactions":
        show_process_transactions()
    elif page == "✏️ Review & Feedback":
        show_review_feedback()
    elif page == "📈 Analytics":
        show_analytics()
    elif page == "⚙️ Configuration":
        show_configuration()
    elif page == "🔄 Apply Feedback":
        show_apply_feedback()
    elif page == "🏷️ Custom Categories":
        show_custom_categories()
    elif page == "📚 Merchant Seed":
        show_merchant_seed()


def show_dashboard():
    """Dashboard page"""
    st.header("📊 Dashboard")
    
    # Load data
    df = load_categorized_transactions()
    
    if df.empty:
        st.info("👈 Process transactions to see dashboard data")
        return
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Transactions", len(df))
    
    with col2:
        success_rate = (df['processing_status'] == 'success').sum() / len(df) * 100
        st.metric("Success Rate", f"{success_rate:.1f}%")
    
    with col3:
        avg_confidence = df['confidence_score'].mean()
        st.metric("Avg Confidence", f"{avg_confidence:.2f}")
    
    with col4:
        if 'classification_source' in df.columns:
            llm_count = (df['classification_source'] == 'llm').sum()
            rule_based_count = (df['classification_source'] == 'rule_based').sum()
            similarity_count = (df['classification_source'] == 'similarity').sum()
            st.metric("LLM Classifications", llm_count)
            st.caption(f"Rule-based: {rule_based_count} | Similarity: {similarity_count}")
        else:
            st.metric("LLM Classifications", 0)
    
    st.divider()
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Category Distribution")
        if 'category' in df.columns:
            category_counts = df['category'].value_counts()
            fig = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Transactions by Category",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("Confidence Distribution")
        if 'confidence_score' in df.columns:
            fig = px.histogram(
                df,
                x='confidence_score',
                nbins=20,
                title="Confidence Score Distribution",
                labels={'confidence_score': 'Confidence Score', 'count': 'Number of Transactions'},
                color_discrete_sequence=['#667eea']
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Recent transactions table
    st.subheader("Recent Transactions")
    # Handle different column name variations (Date vs date)
    date_col = 'date' if 'date' in df.columns else ('Date' if 'Date' in df.columns else None)
    desc_col = 'description' if 'description' in df.columns else ('Description' if 'Description' in df.columns else None)
    
    # Build display columns list, only including columns that exist
    display_cols = []
    if date_col:
        display_cols.append(date_col)
    if desc_col:
        display_cols.append(desc_col)
    if 'merchant' in df.columns:
        display_cols.append('merchant')
    if 'category' in df.columns:
        display_cols.append('category')
    if 'confidence_score' in df.columns:
        display_cols.append('confidence_score')
    if 'classification_source' in df.columns:
        display_cols.append('classification_source')
    
    if display_cols:
        display_df = df[display_cols].head(20)
        # Rename columns to lowercase for consistency
        display_df = display_df.rename(columns={
            'Date': 'date',
            'Description': 'description'
        })
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.warning("No displayable columns found in the data.")
    
    # Export button
    col1, col2, col3 = st.columns(3)
    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    with col2:
        json = df.to_json(orient='records', indent=2)
        st.download_button(
            label="📥 Download JSON",
            data=json,
            file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )
    with col3:
        excel_buffer = io.BytesIO()
        with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Transactions')
        excel_data = excel_buffer.getvalue()
        st.download_button(
            label="📥 Download Excel",
            data=excel_data,
            file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )


def show_process_transactions():
    """Process transactions page"""
    st.header("📊 Process Transactions")
    
    # Show export options if processing just completed
    if st.session_state.get('show_export_options', False) and 'export_result_df' in st.session_state:
        result_df = st.session_state.export_result_df
        st.success(f"✅ Processed {len(result_df)} transactions successfully!")
        st.divider()
        st.subheader("📥 Export Results")
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            json = result_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json,
                file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col3:
            try:
                excel_buffer = io.BytesIO()
                with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                    result_df.to_excel(writer, index=False, sheet_name='Transactions')
                excel_data = excel_buffer.getvalue()
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"categorized_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            except ImportError:
                st.info("💡 Install openpyxl for Excel export: `pip install openpyxl`")
        # Clear the flag after showing
        st.session_state.show_export_options = False
        if 'export_result_df' in st.session_state:
            del st.session_state.export_result_df
        st.divider()
    
    # Show processing status and stop button if active
    if st.session_state.is_processing:
        col1, col2 = st.columns([3, 1])
        with col1:
            st.warning("⚠️ **Processing is in progress. Please wait for it to complete.**")
        with col2:
            if st.button("🛑 Stop Processing", type="secondary", use_container_width=True):
                st.session_state.is_processing = False
                if 'processing_started' in st.session_state:
                    del st.session_state.processing_started
                # Sync radio state with current page
                if "navigation_radio" in st.session_state:
                    st.session_state.navigation_radio = st.session_state.current_page
                st.warning("⚠️ Processing stopped. You can start a new processing job.")
                # Stay on Process Transactions page - no navigation change
                st.rerun()
        st.divider()
    
    tab1, tab2 = st.tabs(["📁 File Upload", "✍️ Single Transaction"])
    
    with tab1:
        st.subheader("Upload Transaction File")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="CSV file should have columns: date, description, amount"
        )
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_path = Path("data/temp_upload.csv")
            temp_path.parent.mkdir(parents=True, exist_ok=True)
            
            df = pd.read_csv(uploaded_file)
            df.to_csv(temp_path, index=False)
            
            st.success(f"✅ File uploaded: {len(df)} transactions")
            
            # Show detected columns info
            detected_cols = []
            if any(col.lower() in ['description', 'desc', 'transaction'] for col in df.columns):
                desc_col = [col for col in df.columns if any(kw in col.lower() for kw in ['desc', 'transaction', 'detail'])][0]
                detected_cols.append(f"📝 Description: **{desc_col}**")
            if any(col.lower() in ['amount', 'amt', 'value'] for col in df.columns):
                amt_col = [col for col in df.columns if any(kw in col.lower() for kw in ['amount', 'value', 'price']) and pd.api.types.is_numeric_dtype(df[col])][0]
                detected_cols.append(f"💰 Amount: **{amt_col}**")
            if any(col.lower() in ['date', 'dt', 'timestamp'] for col in df.columns):
                date_col = [col for col in df.columns if any(kw in col.lower() for kw in ['date', 'dt', 'time'])][0]
                detected_cols.append(f"📅 Date: **{date_col}**")
            
            if detected_cols:
                st.info("🔍 Auto-detected columns: " + " | ".join(detected_cols))
            
            # Show current LLM provider
            st.info(f"🤖 Using **{st.session_state.llm_provider.upper()}** for classification (change in Configuration page)")
            
            # Use default column names - they will be auto-detected by load_transactions
            description_col = None  # Will be auto-detected
            amount_col = None  # Will be auto-detected
            date_col = None  # Will be auto-detected
            
            if st.button("🚀 Process Transactions", type="primary", use_container_width=True, disabled=st.session_state.is_processing):
                # Set processing flag IMMEDIATELY and rerun to disable navigation
                st.session_state.is_processing = True
                st.rerun()
            
            # Check if we're in processing state (after rerun)
            if st.session_state.is_processing and 'processing_started' not in st.session_state:
                # Mark that processing has started
                st.session_state.processing_started = True
                
                # Create progress container
                progress_container = st.container()
                with progress_container:
                    st.write("**Processing Progress:**")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Ensure environment is set for this session
                    import os
                    os.environ['CLASSIFIER_MODEL_PROVIDER'] = st.session_state.llm_provider
                    
                    # Get total count for progress estimation
                    total_count = len(df)
                    output_path = "output/categorized_transactions.csv"
                    
                    # Define progress callback for real-time updates
                    def update_progress(current, total, pct, llm_count, direct_count, eta_str):
                        """Update progress bar and status text"""
                        # Calculate progress percentage
                        # 0-10%: Loading file
                        # 10-15%: Initializing retriever/model
                        # 15-90%: Processing transactions
                        # 90-100%: Saving results
                        if pct < 10:
                            # Loading file phase
                            progress_bar.progress(pct / 100)
                            status_text.info("📥 Loading transaction file...")
                        elif pct < 15:
                            # Initializing phase - show specific status messages
                            progress_bar.progress(pct / 100)
                            # Use the status message from the callback if provided
                            if eta_str and "Loading" in eta_str:
                                status_text.info(eta_str)
                            elif pct < 12:
                                status_text.info("🔄 Loading embedding model (this may take 5-10 seconds on first load)...")
                            elif pct < 15:
                                status_text.info("🔄 Loading FAISS index...")
                            else:
                                status_text.info("🔧 Initializing FAISS index and embedding model...")
                        elif pct == 15.0 and current == 0:
                            # Just finished initialization, about to start processing
                            progress_bar.progress(0.15)
                            status_text.info("🚀 Starting transaction processing...")
                        else:
                            # Processing phase
                            processing_pct = 15 + ((pct - 15) * 0.75)  # 15% to 90% range
                            progress_bar.progress(processing_pct / 100)
                            
                            # Update status text with count
                            if current > 0:
                                status_msg = f"🔄 Processing: **{current}/{total}** transactions ({pct:.1f}%)"
                                if llm_count > 0 or direct_count > 0:
                                    status_msg += f" | LLM: {llm_count} | Direct: {direct_count}"
                                if eta_str:
                                    status_msg += eta_str
                                status_text.info(status_msg)
                            else:
                                # Still initializing or starting first batch
                                status_text.info("🚀 Starting transaction processing...")
                    
                    # Step 1: Load file
                    status_text.info("📥 Loading transaction file...")
                    progress_bar.progress(0.05)
                    
                    # Process transactions with real-time progress updates
                    # Ensure retriever is reloaded before batch processing to use latest index
                    # This is critical after applying feedback - ensures updated categories are used
                    try:
                        from utils.faiss_retriever import get_retriever
                        get_retriever(force_reload=False)  # Auto-reloads if index modified
                        # Retriever will auto-reload if index was modified
                    except Exception as e:
                        # Silently continue - get_retriever() will handle errors internally
                        pass
                    
                    result_df = process_transactions_file(
                        input_file=str(temp_path),
                        output_file=output_path,
                        description_col=description_col,
                        amount_col=amount_col,
                        date_col=date_col,
                        batch_size=64,
                        progress_callback=update_progress
                    )
                    
                    # Step 3: Saving results
                    progress_bar.progress(0.90)
                    status_text.info("💾 Saving results...")
                    
                    progress_bar.progress(1.0)
                    status_text.success(f"✅ Processing complete! Processed {total_count} transactions")
                    
                    # Clear processing flag
                    st.session_state.is_processing = False
                    if 'processing_started' in st.session_state:
                        del st.session_state.processing_started
                    
                    # Sync radio state with current page to ensure proper navigation
                    if "navigation_radio" in st.session_state:
                        st.session_state.navigation_radio = st.session_state.current_page
                    
                    # Clear progress indicators
                    time.sleep(0.5)
                    status_text.empty()
                    
                    st.session_state.processed_data = result_df
                    st.session_state.processing_status = "success"
                    # Store result for showing export options after rerun
                    st.session_state.show_export_options = True
                    st.session_state.export_result_df = result_df
                    
                    st.success(f"✅ Processed {len(result_df)} transactions successfully!")
                    st.balloons()
                    
                    # Rerun to update UI (remove processing messages and re-enable navigation)
                    st.rerun()
                
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.error(f"❌ Error: {str(e)}")
                    st.error(f"❌ Error: {str(e)}")
                    st.session_state.processing_status = "error"
                    # Clear processing flag on error
                    st.session_state.is_processing = False
                    if 'processing_started' in st.session_state:
                        del st.session_state.processing_started
                    # Sync radio state with current page
                    if "navigation_radio" in st.session_state:
                        st.session_state.navigation_radio = st.session_state.current_page
                    # Rerun to update UI (remove processing messages and re-enable navigation)
                    st.rerun()
    
    with tab2:
        st.subheader("Process Single Transaction")
        
        col1, col2 = st.columns(2)
        with col1:
            description = st.text_input("Transaction Description", placeholder="e.g., STARBUCKS COFFEE #1234")
        with col2:
            amount = st.number_input("Amount", value=0.0, step=0.01)
        
        date = st.date_input("Date", value=datetime.now())
        
        # Show current LLM provider
        st.info(f"🤖 Using **{st.session_state.llm_provider.upper()}** for classification (change in Configuration page)")
        
        if st.button("🔍 Classify Transaction", type="primary"):
            if description:
                with st.spinner("Classifying transaction..."):
                    try:
                        # Ensure environment is set for this session
                        import os
                        os.environ['CLASSIFIER_MODEL_PROVIDER'] = st.session_state.llm_provider
                        
                        # Ensure retriever is up-to-date (force reload to ensure latest index is used)
                        from utils.faiss_retriever import get_retriever
                        retriever = get_retriever(force_reload=False)  # Auto-reloads if index modified
                        # Double-check: if index was just rebuilt, force reload
                        # Note: Path is already imported at module level, don't re-import
                        index_path = Path(settings.FAISS_INDEX_PATH)
                        if index_path.exists() and hasattr(retriever, '_load_time'):
                            index_mtime = index_path.stat().st_mtime
                            if retriever._load_time < index_mtime:
                                # Index was modified after retriever was loaded, force reload
                                retriever = get_retriever(force_reload=True)
                        
                        result = process_single_transaction(
                            description=description,
                            amount=float(amount) if amount else None,
                            date=str(date)
                        )
                        
                        # Display results
                        classification_source = result.get('classification_source', 'unknown')
                        if classification_source == 'llm':
                            st.success("✅ Classification Complete! (Using LLM)")
                        elif classification_source == 'rule_based':
                            st.success("✅ Classification Complete! (Using Rule-Based Pattern Matching)")
                        elif classification_source == 'similarity':
                            st.success("✅ Classification Complete! (Using Similarity Search)")
                        else:
                            st.success("✅ Classification Complete!")
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Merchant", result.get('merchant', 'UNKNOWN'))
                        with col2:
                            st.metric("Category", result.get('category', 'Other'))
                        with col3:
                            st.metric("Confidence", f"{result.get('confidence_score', 0):.2f}")
                        
                        # Show explainability
                        st.divider()
                        st.subheader("🔍 Classification Details")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Processing Info:**")
                            classification_source = result.get('classification_source', 'N/A')
                            match_quality = result.get('match_quality', 'N/A')
                            confidence_score = result.get('confidence_score', 0.0)
                            
                            # Highlight classification method
                            if classification_source == 'llm':
                                st.success(f"🤖 **LLM Classification** (Confidence: {confidence_score:.3f} < threshold: 0.65)")
                            elif classification_source == 'rule_based':
                                st.success(f"📋 **Rule-Based Classification** (Fast pattern matching)")
                            elif classification_source == 'similarity':
                                st.info(f"🔍 **Similarity-Based Classification** (FAISS search, respects feedback)")
                            elif classification_source == 'direct_fallback':
                                st.warning(f"⚠️ **LLM Failed, Using Fallback** (Confidence: {confidence_score:.3f})")
                            else:
                                st.info(f"⚡ **Direct Classification** (Confidence: {confidence_score:.3f} ≥ threshold: 0.65)")
                            
                            st.write(f"- Classification Source: {classification_source}")
                            st.write(f"- Match Quality: {match_quality}")
                            st.write(f"- Payment Mode: {result.get('payment_mode', 'N/A')}")
                            st.write(f"- Matches Found: {result.get('num_matches', 0)}")
                            
                            # Show threshold info
                            if classification_source != 'llm' and confidence_score >= 0.65:
                                st.caption(f"💡 LLM is used when confidence < 0.65. Current: {confidence_score:.3f}")
                        
                        with col2:
                            st.write("**Retrieved Merchants:**")
                            top_matches = result.get('top_matches', [])
                            if top_matches:
                                for i, match in enumerate(top_matches[:3], 1):
                                    st.write(f"{i}. {match.get('merchant', 'N/A')} (score: {match.get('score', 0):.3f})")
                            else:
                                st.write("No similar merchants found")
                        
                        # Show full result in expander
                        with st.expander("📋 Full Result JSON"):
                            st.json(result)
                        
                        # Store result in session state for accept/reject actions
                        st.session_state.last_classification_result = result
                        st.session_state.last_classification_input = {
                            'description': description,
                            'amount': float(amount) if amount else None,
                            'date': str(date)
                        }
                        
                        # Accept/Reject/Feedback section
                        st.divider()
                        st.subheader("✅ Accept or Provide Feedback")
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if st.button("✅ Accept & Add to Transactions", type="primary", use_container_width=True):
                                try:
                                    # Add to categorized_transactions.csv
                                    output_path = Path("output/categorized_transactions.csv")
                                    output_path.parent.mkdir(parents=True, exist_ok=True)
                                    
                                    # Load existing transactions or create new
                                    if output_path.exists():
                                        df = pd.read_csv(output_path)
                                    else:
                                        # Create empty DataFrame with expected columns
                                        df = pd.DataFrame(columns=[
                                            'date', 'description', 'amount', 'account_type', 'location',
                                            'description_normalized', 'merchant', 'category', 'confidence_score',
                                            'payment_mode', 'num_matches', 'retrieval_source', 'processing_status',
                                            'match_quality', 'classification_source'
                                        ])
                                    
                                    # Create new row from result
                                    new_row = {
                                        'date': str(date),
                                        'description': description,
                                        'amount': float(amount) if amount else None,
                                        'account_type': 'debit' if (amount and amount < 0) else 'credit',
                                        'location': '',
                                        'description_normalized': result.get('normalized_description', description.upper()),
                                        'merchant': result.get('merchant', 'UNKNOWN'),
                                        'category': result.get('category', 'Other'),
                                        'confidence_score': result.get('confidence_score', 0.0),
                                        'payment_mode': result.get('payment_mode', 'UNKNOWN'),
                                        'num_matches': result.get('num_matches', 0),
                                        'retrieval_source': result.get('retrieval_source', 'unknown'),
                                        'processing_status': result.get('processing_status', 'success'),
                                        'match_quality': result.get('match_quality', 'unknown'),
                                        'classification_source': result.get('classification_source', 'unknown')
                                    }
                                    
                                    # Append new row
                                    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                                    df.to_csv(output_path, index=False)
                                    
                                    st.success("✅ Transaction added to categorized_transactions.csv!")
                                    st.balloons()
                                    time.sleep(1)
                                    st.rerun()
                                    
                                except Exception as e:
                                    st.error(f"❌ Error adding transaction: {str(e)}")
                        
                        with col2:
                            if st.button("✏️ Provide Feedback", type="secondary", use_container_width=True):
                                st.session_state.show_feedback_form = True
                                st.rerun()
                        
                        with col3:
                            if st.button("❌ Reject", type="secondary", use_container_width=True):
                                st.session_state.last_classification_result = None
                                st.session_state.last_classification_input = None
                                st.info("ℹ️ Transaction rejected. You can try again with different input.")
                                st.rerun()
                        
                        # Show feedback form if requested
                        if st.session_state.get('show_feedback_form', False):
                            st.divider()
                            st.subheader("✏️ Provide Feedback")
                            
                            st.write("**Original Classification:**")
                            st.write(f"- **Merchant:** {result.get('merchant', 'N/A')}")
                            category = result.get('category', 'N/A')
                            if not category or str(category).strip() == '' or str(category).lower() == 'nan':
                                category = 'N/A'
                            st.write(f"- **Category:** {category}")
                            st.write(f"- **Confidence:** {result.get('confidence_score', 0):.2f}")
                            
                            st.divider()
                            st.write("**Provide Correction:**")
                            
                            corrected_merchant = st.text_input(
                                "Corrected Merchant",
                                value=result.get('merchant', ''),
                                key="feedback_merchant_single",
                                help="Enter the correct merchant name"
                            )
                            
                            # Category selection with custom category option (same as Review & Feedback)
                            from utils.custom_categories import get_all_categories_with_custom
                            categories = get_all_categories_with_custom()
                            default_idx = categories.index(result.get('category', 'Other')) if result.get('category', 'Other') in categories else 0
                            
                            # Add "Create Custom Category..." option
                            category_options = categories + ["➕ Create Custom Category..."]
                            category_selection_idx = len(categories) if default_idx == len(categories) else default_idx
                            
                            selected_category_option = st.selectbox(
                                "Corrected Category",
                                options=category_options,
                                index=default_idx if default_idx < len(categories) else len(categories),
                                key="feedback_category_select_single",
                                help="Select the correct category or create a new one"
                            )
                            
                            # Handle custom category creation
                            corrected_category = None
                            custom_category_name = None
                            custom_category_description = None
                            
                            if selected_category_option == "➕ Create Custom Category...":
                                st.info("💡 Creating a new custom category")
                                custom_category_name = st.text_input(
                                    "Custom Category Name",
                                    key="custom_name_single",
                                    help="Enter a name for the new category",
                                    placeholder="e.g., Subscription Services"
                                )
                                custom_category_description = st.text_area(
                                    "Category Description (Optional)",
                                    key="custom_desc_single",
                                    help="Describe what transactions should belong to this category",
                                    placeholder="e.g., Monthly subscriptions like Netflix, Spotify, etc."
                                )
                                
                                if custom_category_name and custom_category_name.strip():
                                    corrected_category = custom_category_name.strip()
                                else:
                                    corrected_category = None
                            else:
                                corrected_category = selected_category_option
                            
                            # Submit feedback button
                            submit_key_single = "submit_feedback_single"
                            if submit_key_single not in st.session_state:
                                st.session_state[submit_key_single] = False
                            
                            status_container_single = st.empty()
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                if st.button("💾 Submit Feedback", type="primary", use_container_width=True, 
                                           disabled=st.session_state[submit_key_single], key="submit_feedback_btn_single"):
                                    # Handle custom category creation
                                    if custom_category_name and custom_category_name.strip():
                                        from utils.custom_categories import create_custom_category
                                        create_result = create_custom_category(
                                            name=custom_category_name.strip(),
                                            description=custom_category_description.strip() if custom_category_description else "",
                                            created_by="ui_user"
                                        )
                                        
                                        if not create_result.get('success'):
                                            status_container_single.error(f"❌ Error creating custom category: {create_result.get('error')}")
                                            st.stop()
                                        else:
                                            status_container_single.success(f"✅ Created custom category: {custom_category_name}")
                                            corrected_category = custom_category_name.strip()
                                    
                                    if not corrected_category:
                                        status_container_single.error("❌ Please select or create a category")
                                        st.stop()
                                    
                                    # Prevent double submission
                                    if st.session_state[submit_key_single]:
                                        status_container_single.warning("⚠️ Feedback already submitted. Please refresh the page.")
                                        st.stop()
                                    
                                    # Mark as submitting
                                    st.session_state[submit_key_single] = True
                                    status_container_single.info("🔄 Submitting feedback...")
                                    
                                    try:
                                        # Ensure environment is set for Feedback Agent
                                        import os
                                        os.environ['FEEDBACK_MODEL_PROVIDER'] = st.session_state.feedback_provider
                                        
                                        feedback_result = submit_feedback_from_result(
                                            result,
                                            corrected_merchant.strip().upper() if corrected_merchant else result.get('merchant', 'UNKNOWN'),
                                            corrected_category
                                        )
                                        
                                        if feedback_result.get('success'):
                                            # Increment custom category usage if applicable
                                            from utils.custom_categories import increment_category_usage, is_custom_category
                                            if is_custom_category(corrected_category):
                                                increment_category_usage(corrected_category)
                                            
                                            # Show update status
                                            if feedback_result.get('csv_updated'):
                                                status_container_single.success("✅ Feedback submitted and transaction updated successfully!")
                                                if feedback_result.get('updated_count', 0) > 0:
                                                    status_container_single.info(f"📝 Updated {feedback_result.get('updated_count')} transaction(s) in categorized_transactions.csv")
                                            else:
                                                status_container_single.warning(f"⚠️ Feedback stored, but CSV update failed: {feedback_result.get('csv_update_error', 'Unknown error')}")
                                            
                                            if feedback_result.get('duplicate'):
                                                status_container_single.info(f"ℹ️ {feedback_result.get('message', 'Duplicate feedback skipped')}")
                                            
                                            st.balloons()
                                            time.sleep(1.5)
                                            st.session_state.show_feedback_form = False
                                            st.session_state.last_classification_result = None
                                            st.session_state.last_classification_input = None
                                            st.session_state[submit_key_single] = False
                                            st.rerun()
                                        else:
                                            status_container_single.error(f"❌ Error: {feedback_result.get('error')}")
                                            # Reset submission state on error so user can try again
                                            st.session_state[submit_key_single] = False
                                    except Exception as e:
                                        status_container_single.error(f"❌ Error: {str(e)}")
                                        # Reset submission state on error
                                        st.session_state[submit_key_single] = False
                            
                            with col2:
                                if st.button("❌ Cancel", use_container_width=True, key="cancel_feedback_single"):
                                    st.session_state.show_feedback_form = False
                                    st.session_state[submit_key_single] = False
                                    st.rerun()
                        
                    except Exception as e:
                        st.error(f"❌ Error: {str(e)}")
                        import traceback
                        error_details = traceback.format_exc()
                        with st.expander("🔍 Error Details"):
                            st.code(error_details)
                        
                        # Check if it's an LLM-related error
                        error_str = str(e).lower()
                        if any(keyword in error_str for keyword in ['classifier', 'llm', 'agent', 'openai', 'gemini', 'api']):
                            st.warning("💡 LLM Classification Error - Check:")
                            st.info("""
                            **Possible Issues:**
                            - **API Key**: Ensure OPENAI_API_KEY or GOOGLE_API_KEY is set correctly
                            - **Provider**: Check Configuration page - LLM provider must match your API key
                            - **Model**: Verify model name is correct for your provider
                            - **Network**: Check if API endpoint is accessible
                            - **Rate Limits**: You may have hit API rate limits
                            
                            **Quick Fix:**
                            1. Go to Configuration page
                            2. Verify your LLM provider matches your API key
                            3. Check API key is valid
                            4. Try again
                            """)
            else:
                st.warning("Please enter a transaction description")


def show_review_feedback():
    """Review and feedback page"""
    st.header("✏️ Review & Feedback")
    
    # Load categorized transactions
    df = load_categorized_transactions()
    
    if df.empty:
        st.info("👈 Process transactions first to review and provide feedback")
        return
    
    st.subheader("Review Classifications")
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    with col1:
        filter_category = st.selectbox("Filter by Category", ["All"] + list(df['category'].unique()) if 'category' in df.columns else ["All"])
    with col2:
        min_confidence = st.slider("Min Confidence", 0.0, 1.0, 0.0, 0.01)
    with col3:
        show_low_confidence = st.checkbox("Show Low Confidence Only", value=False)
    
    # Apply filters
    filtered_df = df.copy()
    if filter_category != "All" and 'category' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['category'] == filter_category]
    if 'confidence_score' in filtered_df.columns:
        filtered_df = filtered_df[filtered_df['confidence_score'] >= min_confidence]
        if show_low_confidence:
            filtered_df = filtered_df[filtered_df['confidence_score'] < 0.76]
    
    st.info(f"Showing {len(filtered_df)} transactions")
    
    # Export filtered results
    if len(filtered_df) > 0:
        st.divider()
        col1, col2, col3 = st.columns(3)
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                label="📥 Export Filtered CSV",
                data=csv,
                file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True
            )
        with col2:
            json = filtered_df.to_json(orient='records', indent=2)
            st.download_button(
                label="📥 Export Filtered JSON",
                data=json,
                file_name=f"filtered_transactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col3:
            st.write("")  # Spacer
    
    # Display transactions with feedback form
    st.divider()
    st.subheader("Transaction Details & Feedback")
    
    for idx, row in filtered_df.iterrows():
        # Handle category display - check multiple possible column names
        category_display = row.get('category', row.get('category_y', row.get('category_x', 'N/A')))
        if pd.isna(category_display) or str(category_display).strip() == '' or str(category_display).lower() == 'nan':
            category_display = 'N/A'
        else:
            category_display = str(category_display).strip()
        
        with st.expander(f"📝 {row.get('description', 'N/A')[:60]}... | {row.get('merchant', 'N/A')} / {category_display} (Conf: {row.get('confidence_score', 0):.2f})"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Original Classification:**")
                st.write(f"- **Merchant:** {row.get('merchant', 'N/A')}")
                # Handle category display - check multiple possible column names
                category_display = row.get('category', row.get('category_y', row.get('category_x', 'N/A')))
                if pd.isna(category_display) or str(category_display).strip() == '' or str(category_display).lower() == 'nan':
                    category_display = 'N/A'
                else:
                    category_display = str(category_display).strip()
                st.write(f"- **Category:** {category_display}")
                st.write(f"- **Confidence:** {row.get('confidence_score', 0):.2f}")
                st.write(f"- **Source:** {row.get('classification_source', 'N/A')}")
                st.write(f"- **Match Quality:** {row.get('match_quality', 'N/A')}")
                st.write(f"- **Payment Mode:** {row.get('payment_mode', 'N/A')}")
                
                # Show explainability if available
                if 'top_matches' in row and pd.notna(row.get('top_matches')):
                    st.write("**Similar Merchants Found:**")
                    try:
                        import ast
                        if isinstance(row['top_matches'], str):
                            top_matches = ast.literal_eval(row['top_matches'])
                        else:
                            top_matches = row['top_matches']
                        if top_matches:
                            for i, match in enumerate(top_matches[:3], 1):
                                if isinstance(match, dict):
                                    st.write(f"  {i}. {match.get('merchant', 'N/A')} (score: {match.get('score', 0):.3f})")
                    except:
                        pass
            
            with col2:
                st.write("**Provide Correction:**")
                corrected_merchant = st.text_input(
                    "Corrected Merchant",
                    value=row.get('merchant', ''),
                    key=f"merchant_{idx}",
                    help="Enter the correct merchant name"
                )
                
                # Category selection with custom category option
                categories = get_categories()
                # Handle category - check multiple possible column names
                row_category = row.get('category', row.get('category_y', row.get('category_x', 'Other')))
                if pd.isna(row_category) or str(row_category).strip() == '' or str(row_category).lower() == 'nan':
                    row_category = 'Other'
                else:
                    row_category = str(row_category).strip()
                default_idx = categories.index(row_category) if row_category in categories else 0
                
                # Add "Create Custom Category..." option
                category_options = categories + ["➕ Create Custom Category..."]
                category_selection_idx = len(categories) if default_idx == len(categories) else default_idx
                
                selected_category_option = st.selectbox(
                    "Corrected Category",
                    options=category_options,
                    index=default_idx if default_idx < len(categories) else len(categories),
                    key=f"category_select_{idx}",
                    help="Select the correct category or create a new one"
                )
                
                # Handle custom category creation
                custom_category_name = None
                custom_category_description = None
                corrected_category = None
                
                if selected_category_option == "➕ Create Custom Category...":
                    st.info("💡 Creating a new custom category")
                    custom_category_name = st.text_input(
                        "Custom Category Name",
                        key=f"custom_name_{idx}",
                        help="Enter a name for the new category",
                        placeholder="e.g., Subscription Services"
                    )
                    custom_category_description = st.text_area(
                        "Category Description (Optional)",
                        key=f"custom_desc_{idx}",
                        help="Describe what transactions should belong to this category",
                        placeholder="e.g., Monthly subscriptions like Netflix, Spotify, etc."
                    )
                    
                    if custom_category_name and custom_category_name.strip():
                        corrected_category = custom_category_name.strip()
                    else:
                        corrected_category = None  # Will be set when button is clicked
                else:
                    corrected_category = selected_category_option
                
                # Use session state to prevent double submission
                submit_key = f"submit_feedback_{idx}"
                if submit_key not in st.session_state:
                    st.session_state[submit_key] = False
                
                # Create a container for button and status
                button_container = st.container()
                status_container = st.empty()
                
                with button_container:
                    if st.button("💾 Submit Feedback", key=f"submit_{idx}", use_container_width=True, disabled=st.session_state[submit_key]):
                        # Show immediate visual feedback
                        status_container.info("🔄 Submitting feedback...")
                        
                        # Handle custom category creation
                        if custom_category_name and custom_category_name.strip():
                            from utils.custom_categories import create_custom_category
                            create_result = create_custom_category(
                                name=custom_category_name.strip(),
                                description=custom_category_description.strip() if custom_category_description else "",
                                created_by="ui_user"
                            )
                            
                            if not create_result.get('success'):
                                status_container.error(f"❌ Error creating custom category: {create_result.get('error')}")
                                st.stop()
                            else:
                                status_container.success(f"✅ Created custom category: {custom_category_name}")
                                corrected_category = custom_category_name.strip()
                        
                        if not corrected_category:
                            status_container.error("❌ Please select or create a category")
                            st.stop()
                        
                        # Prevent double submission
                        if st.session_state[submit_key]:
                            status_container.warning("⚠️ Feedback already submitted for this transaction. Please refresh the page.")
                            st.stop()
                        
                        # Mark as submitting
                        st.session_state[submit_key] = True
                        
                        # Ensure environment is set for Feedback Agent
                        import os
                        os.environ['FEEDBACK_MODEL_PROVIDER'] = st.session_state.feedback_provider
                        
                        # Convert row to result dict format expected by submit_feedback_from_result
                        # Map column names from categorized_transactions.csv to pipeline result format
                        result_dict = {
                            'original_description': row.get('description', row.get('original_description', '')),
                            'normalized_description': row.get('normalized_description', row.get('description', '')),
                            'merchant': row.get('merchant', 'UNKNOWN'),
                            'category': row.get('category', row.get('category_y', row.get('category_x', 'Other'))),
                            'amount': row.get('amount'),
                            'date': row.get('date'),
                            'confidence_score': row.get('confidence_score', 0.0),
                            'payment_mode': row.get('payment_mode', 'UNKNOWN'),
                            'match_quality': row.get('match_quality', 'unknown'),
                            'num_matches': row.get('num_matches', 0),
                            'retrieval_source': row.get('retrieval_source', 'unknown'),
                            'classification_source': row.get('classification_source', 'unknown')
                        }
                        
                        feedback_result = submit_feedback_from_result(
                            result_dict,
                            corrected_merchant,
                            corrected_category
                        )
                        
                        if feedback_result.get('success'):
                            # Increment custom category usage if applicable
                            from utils.custom_categories import increment_category_usage, is_custom_category
                            if is_custom_category(corrected_category):
                                increment_category_usage(corrected_category)
                            
                            # Show update status
                            if feedback_result.get('csv_updated'):
                                status_container.success(f"✅ Feedback submitted and transaction updated successfully!")
                                if feedback_result.get('updated_count', 0) > 0:
                                    status_container.info(f"📝 Updated {feedback_result.get('updated_count')} transaction(s) in categorized_transactions.csv")
                            else:
                                status_container.warning(f"⚠️ Feedback stored, but CSV update failed: {feedback_result.get('csv_update_error', 'Unknown error')}")
                            
                            if feedback_result.get('duplicate'):
                                status_container.info(f"ℹ️ {feedback_result.get('message', 'Duplicate feedback skipped')}")
                            
                            st.balloons()
                            time.sleep(1.5)
                            st.rerun()
                        else:
                            status_container.error(f"❌ Error: {feedback_result.get('error')}")
                            # Reset submission state on error so user can try again
                            st.session_state[submit_key] = False


def show_analytics():
    """Analytics page"""
    st.header("📈 Analytics")
    
    # Load data
    df = load_categorized_transactions()
    
    if df.empty:
        st.info("👈 Process transactions to see analytics")
        return
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        if 'classification_source' in df.columns:
            llm_pct = (df['classification_source'] == 'llm').sum() / len(df) * 100
            st.metric("LLM Usage", f"{llm_pct:.1f}%")
    
    with col2:
        if 'confidence_score' in df.columns:
            high_conf = (df['confidence_score'] >= 0.85).sum()
            st.metric("High Confidence", high_conf)
    
    with col3:
        if 'confidence_score' in df.columns:
            low_conf = (df['confidence_score'] < 0.76).sum()
            st.metric("Low Confidence", low_conf)
    
    with col4:
        if 'match_quality' in df.columns:
            high_quality = (df['match_quality'] == 'high').sum()
            st.metric("High Quality", high_quality)
    
    with col5:
        if 'processing_status' in df.columns:
            success_rate = (df['processing_status'] == 'success').sum() / len(df) * 100
            st.metric("Success Rate", f"{success_rate:.1f}%")
    
    st.divider()
    
    # Category performance
    if 'category' in df.columns and 'confidence_score' in df.columns:
        st.subheader("Category Performance")
        
        col1, col2 = st.columns(2)
        
        with col1:
            category_perf = df.groupby('category')['confidence_score'].agg(['mean', 'count']).reset_index()
            category_perf.columns = ['Category', 'Avg Confidence', 'Count']
            
            fig = px.bar(
                category_perf,
                x='Category',
                y='Avg Confidence',
                color='Count',
                title="Average Confidence by Category",
                color_continuous_scale='Viridis',
                labels={'Avg Confidence': 'Avg Confidence', 'Count': 'Transaction Count'}
            )
            fig.update_xaxes(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Category distribution over time (if date available)
            if 'date' in df.columns:
                try:
                    df['date'] = pd.to_datetime(df['date'])
                    category_trends = df.groupby([df['date'].dt.to_period('D'), 'category']).size().reset_index(name='count')
                    category_trends['date'] = category_trends['date'].astype(str)
                    
                    fig = px.line(
                        category_trends,
                        x='date',
                        y='count',
                        color='category',
                        title="Category Trends Over Time",
                        labels={'date': 'Date', 'count': 'Transaction Count', 'category': 'Category'}
                    )
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                except:
                    st.info("Date data not available for trend analysis")
            else:
                # Alternative: Category count chart
                category_counts = df['category'].value_counts()
                fig = px.pie(
                    values=category_counts.values,
                    names=category_counts.index,
                    title="Category Distribution",
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                st.plotly_chart(fig, use_container_width=True)
    
    # Feedback statistics
    st.subheader("Feedback Statistics")
    try:
        feedback_stats = get_feedback_statistics()
        if feedback_stats.get('success'):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Feedback", feedback_stats.get('total_feedback', 0))
            with col2:
                st.metric("Processed", feedback_stats.get('processed', 0))
            with col3:
                st.metric("Pending", feedback_stats.get('pending', 0))
    except:
        st.info("No feedback data available")


def show_configuration():
    """Configuration page"""
    st.header("⚙️ System Configuration")
    
    tab1, tab2, tab3 = st.tabs(["🔧 Processing Settings", "🤖 LLM Settings", "📊 Thresholds"])
    
    with tab1:
        st.subheader("Processing Configuration")
        
        col1, col2 = st.columns(2)
        with col1:
            batch_size = st.number_input(
                "Batch Size",
                min_value=1,
                max_value=1000,
                value=settings.BATCH_SIZE,
                help="Number of transactions to process per batch"
            )
        
        with col2:
            top_k = st.number_input(
                "Top K Retrieval",
                min_value=1,
                max_value=20,
                value=settings.TOP_K_RETRIEVAL,
                help="Number of similar merchants to retrieve"
            )
        
        st.info("💡 Changes take effect on next processing run")
    
    with tab2:
        st.subheader("LLM Configuration")
        
        import os
        
        # Classifier Agent Provider
        st.write("**Classifier Agent (for transaction classification)**")
        current_classifier_provider = st.session_state.llm_provider
        
        classifier_provider = st.radio(
            "Classifier LLM Provider",
            ["Gemini", "OpenAI"],
            index=0 if current_classifier_provider == 'gemini' else 1,
            help="Select the LLM provider for Classifier Agent",
            key="classifier_provider_radio"
        )
        
        # Update session state and environment
        provider_lower = classifier_provider.lower()
        if provider_lower != st.session_state.llm_provider:
            st.session_state.llm_provider = provider_lower
            os.environ['CLASSIFIER_MODEL_PROVIDER'] = provider_lower
            st.success(f"✅ Switched to {classifier_provider} for Classifier Agent")
            st.info("💡 This will be used for the next classification")
        
        # Feedback Agent Provider
        st.divider()
        st.write("**Feedback Agent (for feedback processing)**")
        current_feedback_provider = st.session_state.feedback_provider
        
        feedback_provider = st.radio(
            "Feedback LLM Provider",
            ["Gemini", "OpenAI"],
            index=0 if current_feedback_provider == 'gemini' else 1,
            help="Select the LLM provider for Feedback Agent",
            key="feedback_provider_radio"
        )
        
        # Update session state and environment
        feedback_provider_lower = feedback_provider.lower()
        if feedback_provider_lower != st.session_state.feedback_provider:
            st.session_state.feedback_provider = feedback_provider_lower
            os.environ['FEEDBACK_MODEL_PROVIDER'] = feedback_provider_lower
            st.success(f"✅ Switched to {feedback_provider} for Feedback Agent")
            st.info("💡 This will be used for the next feedback processing")
        
        st.divider()
        
        # Show current configuration
        st.subheader("Current Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Classifier Agent:**")
            st.write(f"- Provider: {st.session_state.llm_provider.upper()}")
            if st.session_state.llm_provider == 'gemini':
                gemini_key = os.getenv('GOOGLE_API_KEY', '')
                if gemini_key:
                    st.success("✅ GOOGLE_API_KEY configured")
                else:
                    st.error("❌ GOOGLE_API_KEY missing")
                st.write(f"- Model: {os.getenv('GEMINI_MODEL_NAME', 'gemini-2.0-flash')}")
            else:
                openai_key = os.getenv('OPENAI_API_KEY', '')
                if openai_key:
                    st.success("✅ OPENAI_API_KEY configured")
                else:
                    st.error("❌ OPENAI_API_KEY missing")
                st.write(f"- Model: {os.getenv('MODEL_NAME', 'gpt-4o-mini')}")
        
        with col2:
            st.write("**Feedback Agent:**")
            st.write(f"- Provider: {st.session_state.feedback_provider.upper()}")
            if st.session_state.feedback_provider == 'gemini':
                gemini_key = os.getenv('GOOGLE_API_KEY', '')
                if gemini_key:
                    st.success("✅ GOOGLE_API_KEY configured")
                else:
                    st.error("❌ GOOGLE_API_KEY missing")
            else:
                openai_key = os.getenv('OPENAI_API_KEY', '')
                if openai_key:
                    st.success("✅ OPENAI_API_KEY configured")
                else:
                    st.error("❌ OPENAI_API_KEY missing")
        
        st.divider()
        
        # Option to persist to .env file
        st.subheader("💾 Persist Settings")
        st.info("""
        **Note:** Changes above are active for this session only.
        To persist settings across app restarts, update your `.env` file:
        - `CLASSIFIER_MODEL_PROVIDER=gemini` or `openai`
        - `FEEDBACK_MODEL_PROVIDER=gemini` or `openai`
        """)
        
        if st.button("📝 Show .env File Instructions", use_container_width=True):
            env_content = f"""# LLM Provider Configuration
CLASSIFIER_MODEL_PROVIDER={st.session_state.llm_provider}
FEEDBACK_MODEL_PROVIDER={st.session_state.feedback_provider}

# For Gemini
GOOGLE_API_KEY=your_google_api_key_here
GEMINI_MODEL_NAME=gemini-2.0-flash

# For OpenAI
OPENAI_API_KEY=your_openai_api_key_here
MODEL_NAME=gpt-4o-mini
AZURE_OPENAI_ENDPOINT=  # Optional, for Azure OpenAI
MODEL_API_VERSION=  # Optional, for Azure OpenAI
"""
            st.code(env_content, language="bash")
            st.info("💡 Copy these settings to your `.env` file to persist them")
    
    with tab3:
        st.subheader("Classification Thresholds")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            local_threshold = st.slider(
                "Local Match Threshold",
                min_value=0.0,
                max_value=1.0,
                value=settings.LOCAL_MATCH_THRESHOLD,
                step=0.01,
                help="Confidence threshold for direct classification"
            )
        
        with col2:
            agent_threshold = st.slider(
                "LLM Agent Threshold",
                min_value=0.0,
                max_value=1.0,
                value=settings.CLASSIFIER_AGENT_THRESHOLD,
                step=0.01,
                help="Confidence below which LLM is used"
            )
        
        with col3:
            high_threshold = st.slider(
                "High Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=settings.CONFIDENCE_HIGH_THRESHOLD,
                step=0.01,
                help="Threshold for high confidence classification"
            )
        
        st.info(f"""
        **Current Settings:**
        - Direct classification: ≥ {local_threshold:.2f}
        - LLM classification: < {agent_threshold:.2f}
        - High confidence: ≥ {high_threshold:.2f}
        """)
        
        st.warning("⚠️ Threshold changes require updating `.env` file and restarting")


def show_apply_feedback():
    """Apply feedback page"""
    st.header("🔄 Apply Feedback to System")
    
    st.info("""
    This will update the merchant seed with pending feedback and rebuild the FAISS index.
    This improves future classifications based on your corrections.
    """)
    
    # Show pending feedback summary
    try:
        from utils.feedback_processor import get_pending_feedback
        summary = get_feedback_summary()
        pending_count = summary.get('pending', 0)
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Pending Feedback", pending_count)
        with col2:
            st.metric("Total Feedback", summary.get('total', 0))
        
        # Explanation
        st.caption("💡 **Pending Feedback** = User corrections that haven't been applied to the merchant seed yet. Applying feedback will improve future transaction categorizations.")
        
        # Check for duplicates
        from utils.feedback_processor import find_duplicate_feedback, remove_duplicate_feedback
        duplicate_info = find_duplicate_feedback()
        
        if duplicate_info.get('success') and duplicate_info.get('duplicate_count', 0) > 0:
            st.warning(f"⚠️ Found {duplicate_info.get('duplicate_count')} duplicate feedback record(s) in {duplicate_info.get('duplicate_groups', 0)} group(s)")
            
            col1, col2 = st.columns([0.7, 0.3])
            with col1:
                st.caption("💡 Duplicates are feedback entries with the same transaction, amount, date, and corrections. You can remove them to clean up the feedback list.")
            with col2:
                if st.button("🧹 Remove Duplicates", type="secondary", help="Remove all duplicate feedback records"):
                    remove_result = remove_duplicate_feedback()
                    if remove_result.get('success'):
                        st.success(f"✅ {remove_result.get('message')}")
                        st.info(f"Remaining feedback records: {remove_result.get('remaining_count', 0)}")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.error(f"❌ Error: {remove_result.get('error')}")
        
        # Show details of pending feedback
        if pending_count > 0:
            st.divider()
            st.subheader("📋 Pending Feedback Details")
            
            pending_df = get_pending_feedback()
            
            if not pending_df.empty:
                # Prepare display columns
                display_cols = []
                if 'transaction_description' in pending_df.columns:
                    display_cols.append('transaction_description')
                if 'original_merchant' in pending_df.columns and 'corrected_merchant' in pending_df.columns:
                    display_cols.extend(['original_merchant', 'corrected_merchant'])
                if 'original_category' in pending_df.columns and 'corrected_category' in pending_df.columns:
                    display_cols.extend(['original_category', 'corrected_category'])
                if 'amount' in pending_df.columns:
                    display_cols.append('amount')
                if 'date' in pending_df.columns:
                    display_cols.append('date')
                if 'timestamp' in pending_df.columns:
                    display_cols.append('timestamp')
                
                # Filter to available columns
                available_cols = [col for col in display_cols if col in pending_df.columns]
                
                if available_cols:
                    # Create a formatted display with delete buttons
                    display_df = pending_df[available_cols].copy()
                    
                    # Rename columns for better readability
                    rename_map = {
                        'transaction_description': 'Transaction',
                        'original_merchant': 'Original Merchant',
                        'corrected_merchant': 'Corrected Merchant',
                        'original_category': 'Original Category',
                        'corrected_category': 'Corrected Category',
                        'amount': 'Amount',
                        'date': 'Date',
                        'timestamp': 'Feedback Date'
                    }
                    display_df = display_df.rename(columns=rename_map)
                    
                    # Display with delete buttons
                    st.write("**Pending Feedback Records:**")
                    
                    # Create a container for each feedback item with delete button
                    for idx, row in pending_df.iterrows():
                        with st.container():
                            col1, col2 = st.columns([0.9, 0.1])
                            
                            with col1:
                                # Display key information
                                import pandas as pd
                                transaction = row.get('transaction_description', 'N/A')
                                # Handle NaN values in display
                                if pd.isna(transaction) or str(transaction).lower() == 'nan':
                                    transaction = 'N/A'
                                
                                original_cat = row.get('original_category', 'N/A')
                                if pd.isna(original_cat) or str(original_cat).strip() == '' or str(original_cat).lower() == 'nan':
                                    original_cat = 'N/A'
                                else:
                                    original_cat = str(original_cat).strip()
                                
                                corrected_cat = row.get('corrected_category', 'N/A')
                                if pd.isna(corrected_cat) or str(corrected_cat).strip() == '' or str(corrected_cat).lower() == 'nan':
                                    corrected_cat = 'N/A'
                                else:
                                    corrected_cat = str(corrected_cat).strip()
                                
                                original_merchant = row.get('original_merchant', 'N/A')
                                if pd.isna(original_merchant):
                                    original_merchant = 'N/A'
                                
                                corrected_merchant = row.get('corrected_merchant', 'N/A')
                                if pd.isna(corrected_merchant):
                                    corrected_merchant = 'N/A'
                                
                                amount = row.get('amount', 'N/A')
                                if pd.isna(amount):
                                    amount = 'N/A'
                                
                                date = row.get('date', 'N/A')
                                if pd.isna(date):
                                    date = 'N/A'
                                
                                # Format display
                                st.markdown(f"""
                                **Transaction:** {transaction}  
                                **Category:** {original_cat} → {corrected_cat}  
                                **Merchant:** {original_merchant} → {corrected_merchant}  
                                **Amount:** ${amount} | **Date:** {date}
                                """)
                            
                            with col2:
                                # Delete button
                                delete_key = f"delete_feedback_{idx}"
                                if st.button("🗑️", key=delete_key, help="Delete this feedback"):
                                    # Get the original index in the full feedback DataFrame
                                    from utils.feedback_processor import load_feedback, delete_feedback
                                    import pandas as pd
                                    full_df = load_feedback()
                                    
                                    # Find matching row in full DataFrame
                                    # Match by multiple fields to ensure we find the right record
                                    # Handle NaN values properly
                                    mask = pd.Series([True] * len(full_df), index=full_df.index)
                                    
                                    # Match transaction_description (handle NaN)
                                    if 'transaction_description' in full_df.columns:
                                        row_desc = row.get('transaction_description', '')
                                        if pd.isna(row_desc) or str(row_desc).lower() == 'nan':
                                            mask = mask & (full_df['transaction_description'].isna() | (full_df['transaction_description'].astype(str).str.lower() == 'nan'))
                                        else:
                                            mask = mask & (full_df['transaction_description'] == row_desc)
                                    
                                    # Match amount (handle NaN)
                                    if 'amount' in full_df.columns:
                                        row_amount = row.get('amount')
                                        if pd.isna(row_amount):
                                            mask = mask & full_df['amount'].isna()
                                        else:
                                            mask = mask & (full_df['amount'] == row_amount)
                                    
                                    # Match date
                                    if 'date' in full_df.columns:
                                        row_date = row.get('date', '')
                                        if pd.isna(row_date) or str(row_date).lower() == 'nan':
                                            mask = mask & (full_df['date'].isna() | (full_df['date'].astype(str).str.lower() == 'nan'))
                                        else:
                                            mask = mask & (full_df['date'] == row_date)
                                    
                                    # Match corrected_merchant
                                    if 'corrected_merchant' in full_df.columns:
                                        row_merchant = row.get('corrected_merchant', '')
                                        if pd.notna(row_merchant):
                                            mask = mask & (full_df['corrected_merchant'] == row_merchant)
                                    
                                    # Match corrected_category
                                    if 'corrected_category' in full_df.columns:
                                        row_category = row.get('corrected_category', '')
                                        if pd.notna(row_category):
                                            mask = mask & (full_df['corrected_category'] == row_category)
                                    
                                    # Match timestamp if available (for more precise matching)
                                    if 'timestamp' in full_df.columns and 'timestamp' in row:
                                        row_timestamp = row.get('timestamp')
                                        if pd.notna(row_timestamp):
                                            mask = mask & (full_df['timestamp'] == row_timestamp)
                                    
                                    matching_indices = full_df[mask].index.tolist()
                                    
                                    if matching_indices:
                                        # Delete the first matching record
                                        result = delete_feedback(matching_indices[:1])
                                        
                                        if result.get('success'):
                                            st.success(f"✅ {result.get('message')}")
                                            time.sleep(0.5)  # Brief pause for user to see success message
                                            st.rerun()  # Refresh the page
                                        else:
                                            st.error(f"❌ Error: {result.get('error')}")
                                    else:
                                        st.error("❌ Could not find feedback record to delete")
                                        # Debug info
                                        st.caption(f"Debug: Looking for record with transaction='{row.get('transaction_description')}', amount={row.get('amount')}, date={row.get('date')}")
                            
                            st.divider()
                    
                    # Show summary statistics
                    st.caption(f"Showing {len(display_df)} pending feedback record(s)")
                    
                    # Show category changes summary
                    if 'corrected_category' in pending_df.columns:
                        category_changes = pending_df.groupby(['original_category', 'corrected_category']).size().reset_index(name='count')
                        if len(category_changes) > 0:
                            with st.expander("📊 Category Changes Summary"):
                                st.dataframe(
                                    category_changes.rename(columns={
                                        'original_category': 'From Category',
                                        'corrected_category': 'To Category',
                                        'count': 'Count'
                                    }),
                                    use_container_width=True,
                                    hide_index=True
                                )
                else:
                    st.warning("⚠️ Pending feedback data structure is unexpected")
            else:
                st.info("No pending feedback records found")
        
        if pending_count > 0:
            if st.button("🔄 Apply Feedback & Rebuild Index", type="primary"):
                # Create progress container
                progress_container = st.container()
                with progress_container:
                    st.write("**Progress:**")
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                
                try:
                    # Step 1: Load feedback
                    status_text.info("📥 Loading pending feedback...")
                    progress_bar.progress(10)
                    
                    # Step 2: Process feedback
                    status_text.info("🔄 Processing feedback and updating merchant seed...")
                    progress_bar.progress(30)
                    
                    result = update_merchant_seed_from_feedback(rebuild_index=False)  # We'll rebuild manually with progress
                    
                    progress_bar.progress(60)
                    
                    if result.get('success'):
                        # Step 3: Rebuild FAISS index if new patterns added OR categories updated
                        # Categories are stored in FAISS metadata, so index must be rebuilt when they change
                        new_patterns = result.get('new_patterns', 0)
                        categories_updated = result.get('categories_updated', False)
                        
                        if new_patterns > 0 or categories_updated:
                            if categories_updated and new_patterns == 0:
                                status_text.info("🔨 Rebuilding FAISS index due to category updates...")
                            elif new_patterns > 0 and categories_updated:
                                status_text.info("🔨 Rebuilding FAISS index with new patterns and category updates...")
                            else:
                                status_text.info("🔨 Rebuilding FAISS index with new patterns...")
                            progress_bar.progress(70)
                            
                            import subprocess
                            import sys
                            rebuild_result = subprocess.run(
                                [sys.executable, "utils/faiss_index_builder.py"],
                                capture_output=True,
                                text=True
                            )
                            
                            progress_bar.progress(90)
                            
                            if rebuild_result.returncode == 0:
                                status_text.success("✅ FAISS index rebuilt successfully!")
                                # Reload the retriever singleton to use the updated index
                                try:
                                    from utils.faiss_retriever import reload_retriever
                                    reload_retriever()
                                    status_text.success("✅ FAISS retriever reloaded with updated index!")
                                except Exception as e:
                                    status_text.warning(f"⚠️ Could not reload FAISS retriever: {str(e)[:100]}")
                            else:
                                status_text.warning(f"⚠️ FAISS index rebuild had issues: {rebuild_result.stderr[:200]}")
                        else:
                            # Even if no rebuild, ensure retriever is up-to-date
                            try:
                                from utils.faiss_retriever import get_retriever
                                get_retriever()  # This will auto-reload if index was modified
                                status_text.info("ℹ️ No index rebuild needed, but verified retriever is up-to-date")
                            except Exception as e:
                                status_text.warning(f"⚠️ Could not verify retriever: {str(e)[:100]}")
                        
                        progress_bar.progress(100)
                        status_text.success("✅ Feedback applied and index rebuilt successfully!")
                        
                        st.success(f"✅ {result.get('message')}")
                        
                        # Show metrics with before/after counts
                        new_patterns = result.get('new_patterns', 0)
                        updated_merchants = result.get('updated_merchants', 0)
                        
                        # Get current merchant seed count for display
                        try:
                            seed_path = Path(settings.MERCHANT_SEED_PATH)
                            if seed_path.exists():
                                seed_df = pd.read_csv(seed_path)
                                current_count = len(seed_df)
                                previous_count = current_count - new_patterns
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Merchant Seed Records", 
                                        current_count,
                                        delta=new_patterns if new_patterns > 0 else None,
                                        help=f"Previous: {previous_count}, Added: {new_patterns}"
                                    )
                                with col2:
                                    st.metric("New Patterns Added", new_patterns)
                                with col3:
                                    st.metric("Updated Merchants", updated_merchants)
                            else:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("New Patterns Added", new_patterns)
                                with col2:
                                    st.metric("Updated Merchants", updated_merchants)
                        except Exception as e:
                            st.metric("New Patterns Added", new_patterns)
                            st.metric("Updated Merchants", updated_merchants)
                            st.warning(f"Could not load merchant seed count: {e}")
                        
                        if new_patterns > 0:
                            st.info(f"💡 Added {new_patterns} new pattern(s) to merchant seed. FAISS index has been rebuilt.")
                        elif categories_updated:
                            st.info(f"💡 Updated categories for {updated_merchants} merchant(s). FAISS index has been rebuilt with updated categories.")
                        else:
                            st.info("💡 No new patterns added (may be duplicates or already exist)")
                        
                        # Important: Force reload retriever after page refresh
                        st.info("🔄 **Please refresh the page or process a new transaction to see the updated category.**")
                        
                        st.balloons()
                        
                        # IMPORTANT: Force reload retriever BEFORE page refresh to ensure updated index is used
                        # The retriever singleton needs to be reloaded so subsequent queries use the updated category
                        try:
                            from utils.faiss_retriever import reload_retriever
                            reload_retriever()
                            st.success("🔄 Retriever reloaded - updated categories will be used in next classification!")
                            
                            # Also clear any cached results in session state that might interfere
                            if 'last_classification_result' in st.session_state:
                                del st.session_state['last_classification_result']
                        except Exception as e:
                            st.warning(f"⚠️ Could not reload retriever: {str(e)[:100]}")
                            st.info("💡 Please restart the Streamlit app to ensure updated categories are used")
                        
                        # Refresh the page after a brief delay to show updated data
                        time.sleep(2)
                        st.rerun()
                    else:
                        progress_bar.progress(100)
                        status_text.empty()
                        st.error(f"❌ Error: {result.get('error')}")
                except Exception as e:
                    progress_bar.progress(100)
                    status_text.empty()
                    st.error(f"❌ Error: {str(e)}")
        else:
            st.info("No pending feedback to apply")
    except Exception as e:
        st.error(f"Error: {str(e)}")


def show_custom_categories():
    """Custom Categories management page"""
    st.header("🏷️ Custom Categories Management")
    
    from utils.custom_categories import (
        get_all_custom_categories,
        get_custom_categories_summary,
        delete_custom_category,
        create_custom_category,
        is_custom_category
    )
    from pipeline import process_single_transaction
    import pandas as pd
    
    # Summary
    summary = get_custom_categories_summary()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Custom Categories", summary['total_custom_categories'])
    with col2:
        st.metric("Transactions in Custom Categories", summary['total_transactions_in_custom'])
    
    st.divider()
    
    # Create new custom category
    with st.expander("➕ Create New Custom Category", expanded=False):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_category_name = st.text_input(
                "Category Name",
                placeholder="e.g., Subscription Services",
                help="Enter a name for the new category"
            )
            new_category_description = st.text_area(
                "Description (Optional)",
                placeholder="Describe what transactions should belong to this category",
                help="Optional description to help identify transactions for this category"
            )
        
        if st.button("✨ Create Category", type="primary"):
            if new_category_name and new_category_name.strip():
                result = create_custom_category(
                    name=new_category_name.strip(),
                    description=new_category_description.strip() if new_category_description else "",
                    created_by="ui_user"
                )
                if result.get('success'):
                    st.success(f"✅ {result.get('message')}")
                    st.balloons()
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error(f"❌ {result.get('error')}")
            else:
                st.error("❌ Please enter a category name")
    
    st.divider()
    
    # List existing custom categories
    custom_cats = get_all_custom_categories()
    
    if not custom_cats:
        st.info("📝 No custom categories created yet. Create one above!")
    else:
        st.subheader("Existing Custom Categories")
        
        for cat in custom_cats:
            with st.expander(f"🏷️ {cat['name']} ({cat.get('transaction_count', 0)} transactions)"):
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    st.write(f"**Description:** {cat.get('description', 'No description')}")
                    st.write(f"**Created:** {cat.get('created_at', 'Unknown')}")
                    st.write(f"**Transactions:** {cat.get('transaction_count', 0)}")
                    if cat.get('keywords'):
                        st.write(f"**Keywords:** {', '.join(cat.get('keywords', []))}")
                
                with col2:
                    if st.button("🗑️ Delete", key=f"delete_{cat['name']}", type="secondary", use_container_width=True):
                        # Confirm deletion
                        delete_result = delete_custom_category(cat['name'])
                        
                        if delete_result.get('success'):
                            affected_count = delete_result.get('affected_count', 0)
                            st.warning(f"⚠️ Deleted category '{cat['name']}'. {affected_count} transactions need reclassification.")
                            
                            # Option to reclassify affected transactions
                            if affected_count > 0:
                                st.info("💡 Reclassifying affected transactions...")
                                
                                # Load categorized transactions
                                from utils.feedback_helper import load_categorized_transactions
                                df = load_categorized_transactions()
                                
                                if not df.empty and 'category' in df.columns:
                                    # Find transactions with deleted category
                                    affected_df = df[df['category'] == cat['name']].copy()
                                    
                                    if not affected_df.empty:
                                        st.write(f"Found {len(affected_df)} transactions to reclassify")
                                        
                                        # Option 1: Use LLM to reclassify
                                        if st.button("🤖 Reclassify with LLM", key=f"reclassify_{cat['name']}"):
                                            st.info("Reclassifying with LLM...")
                                            reclassified = 0
                                            
                                            # Ensure retriever is up-to-date before reclassifying
                                            from utils.faiss_retriever import get_retriever
                                            get_retriever()  # Auto-reloads if index modified
                                            
                                            progress_bar = st.progress(0)
                                            for i, (_, row) in enumerate(affected_df.iterrows()):
                                                try:
                                                    result = process_single_transaction(
                                                        description=row.get('description', ''),
                                                        amount=row.get('amount'),
                                                        date=row.get('date')
                                                    )
                                                    reclassified += 1
                                                    progress_bar.progress((i + 1) / len(affected_df))
                                                except Exception as e:
                                                    st.error(f"Error: {e}")
                                            
                                            st.success(f"✅ Reclassified {reclassified} transactions")
                                            st.info("💡 Please reprocess your transactions file to see updated categories")
                                        
                                        # Show affected transactions
                                        st.write("**Affected Transactions:**")
                                        st.dataframe(affected_df[['description', 'merchant', 'category', 'amount']].head(10))
                                        
                                        if len(affected_df) > 10:
                                            st.caption(f"... and {len(affected_df) - 10} more")
                            
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ {delete_result.get('error')}")


def show_merchant_seed():
    """Merchant Seed management page"""
    st.header("📚 Merchant Seed Management")
    
    st.info("""
    **Merchant Seed** is the knowledge base for the RAG system. It contains merchant patterns used to:
    - Build the FAISS vector index for similarity search
    - Classify transactions with high confidence
    - Learn from your feedback
    
    **Note:** The merchant seed is updated automatically when you apply feedback, but you can also generate or expand it manually.
    """)
    
    seed_path = Path(settings.MERCHANT_SEED_PATH)
    
    # Show current status
    col1, col2, col3 = st.columns(3)
    with col1:
        if seed_path.exists():
            try:
                seed_df = pd.read_csv(seed_path)
                st.metric("Current Records", len(seed_df))
            except:
                st.metric("Current Records", "Error")
        else:
            st.metric("Current Records", "Missing")
    
    with col2:
        index_path = Path(settings.FAISS_INDEX_PATH)
        if index_path.exists():
            st.metric("FAISS Index", "✅ Ready")
        else:
            st.metric("FAISS Index", "❌ Missing")
    
    with col3:
        try:
            summary = get_feedback_summary()
            st.metric("Pending Feedback", summary.get('pending', 0))
        except:
            st.metric("Pending Feedback", 0)
    
    st.divider()
    
    # View current merchant seed
    if seed_path.exists():
        st.subheader("📋 Current Merchant Seed")
        try:
            seed_df = pd.read_csv(seed_path)
            
            if seed_df.empty:
                st.warning("⚠️ Merchant seed file is empty!")
            else:
                # Show statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Records", len(seed_df))
                with col2:
                    st.metric("Unique Merchants", seed_df['merchant'].nunique() if 'merchant' in seed_df.columns else 0)
                with col3:
                    if 'category' in seed_df.columns:
                        st.metric("Categories", seed_df['category'].nunique())
                
                # Show sample data
                with st.expander("📊 View Merchant Seed Data", expanded=False):
                    display_cols = ['merchant', 'category']
                    if 'description' in seed_df.columns:
                        display_cols.append('description')
                    if 'transaction_pattern' in seed_df.columns:
                        display_cols.append('transaction_pattern')
                    
                    st.dataframe(seed_df[display_cols], use_container_width=True, hide_index=True)
                
                # Download button
                csv = seed_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Merchant Seed CSV",
                    data=csv,
                    file_name=f"merchants_seed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"❌ Error reading merchant seed: {e}")
    else:
        st.warning("⚠️ Merchant seed file not found. Generate it below.")
    
    st.divider()
    
    # Generate merchant seed
    st.subheader("🔧 Generate Merchant Seed")
    st.write("Generate a comprehensive merchant seed with multiple transaction patterns per merchant from your transactions file.")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload Transactions CSV File",
        type=['csv'],
        help="Upload a CSV file with transaction data. The file should have a description/transaction column and optionally an amount column."
    )
    
    # Or use existing file path
    use_existing = st.checkbox("Or use existing file path", value=False)
    transactions_file = None
    
    if use_existing:
        transactions_file = st.text_input(
            "Transactions File Path",
            value="data/raw_transactions_bkp.csv",
            help="Path to raw transactions CSV file. Tip: Use raw_transactions_bkp.csv (2000 transactions) for comprehensive seed generation."
        )
    
    # Column selection (shown after file is selected)
    description_col = None
    amount_col = None
    df_preview = None
    
    if uploaded_file is not None:
        try:
            # Reset file pointer to beginning
            uploaded_file.seek(0)
            df_preview = pd.read_csv(uploaded_file)
            # Reset again for later use
            uploaded_file.seek(0)
            
            if df_preview.empty:
                st.error("❌ The uploaded file is empty. Please upload a file with transaction data.")
            elif len(df_preview.columns) == 0:
                st.error("❌ The uploaded file has no columns. Please check the file format.")
            else:
                st.info(f"✅ File loaded: {len(df_preview)} transactions, {len(df_preview.columns)} columns")
                
                # Column selection
                st.write("**Column Selection:**")
                col1, col2 = st.columns(2)
                with col1:
                    # Auto-detect description column
                    desc_candidates = [col for col in df_preview.columns if any(kw in col.lower() for kw in ['desc', 'transaction', 'detail', 'memo', 'note'])]
                    default_desc = desc_candidates[0] if desc_candidates else df_preview.columns[0]
                    description_col = st.selectbox(
                        "Description Column",
                        options=list(df_preview.columns),
                        index=list(df_preview.columns).index(default_desc) if default_desc in df_preview.columns else 0,
                        help="Column containing transaction descriptions"
                    )
                with col2:
                    # Auto-detect amount column
                    amount_candidates = [col for col in df_preview.columns if any(kw in col.lower() for kw in ['amount', 'value', 'price', 'cost']) and pd.api.types.is_numeric_dtype(df_preview[col])]
                    default_amount = amount_candidates[0] if amount_candidates else None
                    if default_amount:
                        amount_col = st.selectbox(
                            "Amount Column (Optional)",
                            options=["None"] + list(df_preview.columns),
                            index=list(df_preview.columns).index(default_amount) + 1 if default_amount in df_preview.columns else 0,
                            help="Column containing transaction amounts (optional)"
                        )
                        if amount_col == "None":
                            amount_col = None
                    else:
                        amount_col = st.selectbox(
                            "Amount Column (Optional)",
                            options=["None"] + list(df_preview.columns),
                            index=0,
                            help="Column containing transaction amounts (optional)"
                        )
                        if amount_col == "None":
                            amount_col = None
                
                if len(df_preview) < 100:
                    st.warning(f"⚠️ Your file has only {len(df_preview)} transactions. For a comprehensive seed, consider using a file with 500+ transactions.")
                else:
                    st.info(f"✅ Good! Your file has {len(df_preview)} transactions. This should generate a comprehensive seed.")
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded file is empty or has no valid data. Please upload a CSV file with transaction data.")
        except Exception as e:
            st.warning(f"⚠️ Could not preview file: {str(e)}")
    elif use_existing and transactions_file:
        if Path(transactions_file).exists():
            try:
                df_preview = pd.read_csv(transactions_file)
                
                st.info(f"✅ File loaded: {len(df_preview)} transactions, {len(df_preview.columns)} columns")
                
                # Column selection
                st.write("**Column Selection:**")
                col1, col2 = st.columns(2)
                with col1:
                    # Auto-detect description column
                    desc_candidates = [col for col in df_preview.columns if any(kw in col.lower() for kw in ['desc', 'transaction', 'detail', 'memo', 'note'])]
                    default_desc = desc_candidates[0] if desc_candidates else df_preview.columns[0]
                    description_col = st.selectbox(
                        "Description Column",
                        options=list(df_preview.columns),
                        index=list(df_preview.columns).index(default_desc) if default_desc in df_preview.columns else 0,
                        help="Column containing transaction descriptions",
                        key="desc_col_existing"
                    )
                with col2:
                    # Auto-detect amount column
                    amount_candidates = [col for col in df_preview.columns if any(kw in col.lower() for kw in ['amount', 'value', 'price', 'cost']) and pd.api.types.is_numeric_dtype(df_preview[col])]
                    default_amount = amount_candidates[0] if amount_candidates else None
                    if default_amount:
                        amount_col = st.selectbox(
                            "Amount Column (Optional)",
                            options=["None"] + list(df_preview.columns),
                            index=list(df_preview.columns).index(default_amount) + 1 if default_amount in df_preview.columns else 0,
                            help="Column containing transaction amounts (optional)",
                            key="amount_col_existing"
                        )
                        if amount_col == "None":
                            amount_col = None
                    else:
                        amount_col = st.selectbox(
                            "Amount Column (Optional)",
                            options=["None"] + list(df_preview.columns),
                            index=0,
                            help="Column containing transaction amounts (optional)",
                            key="amount_col_existing2"
                        )
                        if amount_col == "None":
                            amount_col = None
                
                if len(df_preview) < 100:
                    st.warning(f"⚠️ Selected file has only {len(df_preview)} transactions. For a comprehensive seed, consider using 'data/raw_transactions_bkp.csv' (2000 transactions).")
                else:
                    st.info(f"✅ Good! Selected file has {len(df_preview)} transactions. This should generate a comprehensive seed.")
            except Exception as e:
                st.warning(f"⚠️ Could not preview file: {str(e)}")
    
    col1, col2 = st.columns(2)
    with col1:
        min_occurrences = st.number_input(
            "Min Occurrences",
            value=1,
            min_value=1,
            help="Minimum occurrences to include a merchant. Use 1 for maximum coverage (includes all merchants), 2+ to filter rare merchants."
        )
    with col2:
        max_patterns = st.number_input(
            "Max Patterns per Merchant",
            value=15,
            min_value=1,
            max_value=50,
            help="Maximum transaction patterns per merchant. Higher values = more comprehensive seed (recommended: 10-15)"
        )
    
    if st.button("🔨 Generate Merchant Seed", type="primary", use_container_width=True):
        # Determine which file to use
        file_to_use = None
        temp_file_path = None
        
        if uploaded_file is not None:
            # Save uploaded file temporarily
            temp_file_path = Path("data/temp_transactions_for_seed.csv")
            temp_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            try:
                # Reset file pointer to beginning
                uploaded_file.seek(0)
                df_upload = pd.read_csv(uploaded_file)
                
                if df_upload.empty:
                    st.error("❌ The uploaded file is empty. Please upload a file with transaction data.")
                    st.stop()
                
                if len(df_upload.columns) == 0:
                    st.error("❌ The uploaded file has no columns. Please check the file format.")
                    st.stop()
                
                # Auto-detect and rename columns to standard names
                # First, try to detect description column
                if description_col is None or description_col not in df_upload.columns:
                    desc_candidates = [col for col in df_upload.columns if any(kw in col.lower() for kw in ['desc', 'transaction', 'detail', 'memo', 'note'])]
                    description_col = desc_candidates[0] if desc_candidates else df_upload.columns[0]
                
                # Rename description column
                if description_col and description_col in df_upload.columns and description_col != 'description':
                    df_upload = df_upload.rename(columns={description_col: 'description'})
                    st.info(f"📝 Using '{description_col}' as description column")
                
                # Auto-detect and rename amount column
                if amount_col is None or (amount_col not in df_upload.columns and amount_col != "None"):
                    amount_candidates = [col for col in df_upload.columns if any(kw in col.lower() for kw in ['amount', 'value', 'price', 'cost']) and pd.api.types.is_numeric_dtype(df_upload[col])]
                    amount_col = amount_candidates[0] if amount_candidates else None
                
                if amount_col and amount_col in df_upload.columns and amount_col != 'amount':
                    df_upload = df_upload.rename(columns={amount_col: 'amount'})
                    st.info(f"💰 Using '{amount_col}' as amount column")
                elif 'amount' not in df_upload.columns:
                    # Add dummy amount column if not provided
                    df_upload['amount'] = 0.0
                    st.info("💰 Amount column not found, using 0.0 for all transactions")
                
                # Verify required columns exist
                if 'description' not in df_upload.columns:
                    st.error(f"❌ Could not find description column. Available columns: {list(df_upload.columns)}")
                    st.stop()
                
                # Save file with correct columns
                df_upload.to_csv(temp_file_path, index=False)
                
                # Verify the saved file has correct columns
                df_verify = pd.read_csv(temp_file_path)
                if 'description' not in df_verify.columns:
                    st.error(f"❌ Error: Saved file does not have 'description' column. Columns: {list(df_verify.columns)}")
                    st.stop()
                
                file_to_use = str(temp_file_path)
                st.info(f"📁 Using uploaded file: {uploaded_file.name} ({len(df_upload)} transactions)")
                st.info(f"📋 Columns in processed file: {list(df_upload.columns)}")
            except pd.errors.EmptyDataError:
                st.error("❌ The uploaded file is empty or has no valid data. Please upload a CSV file with transaction data.")
                st.stop()
            except Exception as e:
                st.error(f"❌ Error reading uploaded file: {str(e)}")
                st.stop()
        elif use_existing and transactions_file:
            if not Path(transactions_file).exists():
                st.error(f"❌ File not found: {transactions_file}")
                st.stop()
            
            # Always process the file to ensure correct column names
            try:
                df_existing = pd.read_csv(transactions_file)
                
                # Auto-detect description column if not provided
                if description_col is None or description_col not in df_existing.columns:
                    desc_candidates = [col for col in df_existing.columns if any(kw in col.lower() for kw in ['desc', 'transaction', 'detail', 'memo', 'note'])]
                    description_col = desc_candidates[0] if desc_candidates else df_existing.columns[0]
                
                # Rename description column
                if description_col and description_col in df_existing.columns and description_col != 'description':
                    df_existing = df_existing.rename(columns={description_col: 'description'})
                    st.info(f"📝 Using '{description_col}' as description column")
                
                # Auto-detect amount column if not provided
                if amount_col is None or (amount_col not in df_existing.columns and amount_col != "None"):
                    amount_candidates = [col for col in df_existing.columns if any(kw in col.lower() for kw in ['amount', 'value', 'price', 'cost']) and pd.api.types.is_numeric_dtype(df_existing[col])]
                    amount_col = amount_candidates[0] if amount_candidates else None
                
                if amount_col and amount_col in df_existing.columns and amount_col != 'amount':
                    df_existing = df_existing.rename(columns={amount_col: 'amount'})
                    st.info(f"💰 Using '{amount_col}' as amount column")
                elif 'amount' not in df_existing.columns:
                    df_existing['amount'] = 0.0
                    st.info("💰 Amount column not found, using 0.0 for all transactions")
                
                # Verify required columns exist
                if 'description' not in df_existing.columns:
                    st.error(f"❌ Could not find description column. Available columns: {list(df_existing.columns)}")
                    st.stop()
                
                # Always create temp file to ensure correct column names
                temp_file_path = Path("data/temp_transactions_for_seed_existing.csv")
                temp_file_path.parent.mkdir(parents=True, exist_ok=True)
                df_existing.to_csv(temp_file_path, index=False)
                
                # Verify the saved file has correct columns
                df_verify = pd.read_csv(temp_file_path)
                if 'description' not in df_verify.columns:
                    st.error(f"❌ Error: Saved file does not have 'description' column. Columns: {list(df_verify.columns)}")
                    st.stop()
                
                file_to_use = str(temp_file_path)
                st.info(f"📋 Columns in processed file: {list(df_existing.columns)}")
            except Exception as e:
                st.error(f"❌ Error processing file: {str(e)}")
                import traceback
                st.code(traceback.format_exc())
                st.stop()
        else:
            st.error("❌ Please upload a file or provide a file path")
            st.stop()
        
        if file_to_use:
            # Verify the file exists and has correct columns before calling expand_merchant_seed
            try:
                df_check = pd.read_csv(file_to_use)
                if 'description' not in df_check.columns:
                    st.error(f"❌ Error: File '{file_to_use}' does not have 'description' column.")
                    st.error(f"Available columns: {list(df_check.columns)}")
                    st.error("This should not happen - please try uploading the file again.")
                    st.stop()
            except Exception as e:
                st.error(f"❌ Error reading file before processing: {e}")
                st.stop()
            
            with st.spinner("Generating comprehensive merchant seed (this may take a moment)..."):
                try:
                    from utils.expand_merchant_seed import expand_merchant_seed
                    
                    st.info(f"🔍 Processing file: {file_to_use}")
                    seed_df = expand_merchant_seed(
                        transactions_file=file_to_use,
                        output_file=str(seed_path),
                        min_occurrences=int(min_occurrences),
                        max_patterns_per_merchant=int(max_patterns)
                    )
                    
                    # Clean up temp files if created
                    if temp_file_path and temp_file_path.exists():
                        temp_file_path.unlink()
                    temp_existing_path = Path("data/temp_transactions_for_seed_existing.csv")
                    if temp_existing_path.exists():
                        temp_existing_path.unlink()
                    
                    st.success(f"✅ Generated {len(seed_df)} merchant patterns!")
                    st.info(f"💡 Found {seed_df['merchant'].nunique() if 'merchant' in seed_df.columns else 0} unique merchants")
                    st.info("💡 Next step: Rebuild the FAISS index to use the new seed")
                    time.sleep(1)
                    st.rerun()
                except Exception as e:
                    # Clean up temp files on error
                    if temp_file_path and temp_file_path.exists():
                        temp_file_path.unlink()
                    temp_existing_path = Path("data/temp_transactions_for_seed_existing.csv")
                    if temp_existing_path.exists():
                        temp_existing_path.unlink()
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())
    
    st.divider()
    
    # Rebuild FAISS index
    st.subheader("🔨 Rebuild FAISS Index")
    st.write("Rebuild the FAISS index from the current merchant seed. This is required after generating or expanding the seed.")
    
    if st.button("🔨 Rebuild FAISS Index", type="primary"):
        if not seed_path.exists():
            st.error("❌ Merchant seed file not found. Please generate it first.")
        else:
            with st.spinner("Rebuilding FAISS index (this may take a moment)..."):
                try:
                    import subprocess
                    import sys
                    
                    result = subprocess.run(
                        [sys.executable, "utils/faiss_index_builder.py"],
                        capture_output=True,
                        text=True,
                        cwd=Path(__file__).parent.parent
                    )
                    
                    if result.returncode == 0:
                        st.success("✅ FAISS index rebuilt successfully!")
                        st.info("💡 The index is now ready for transaction processing")
                    else:
                        st.error(f"❌ Error rebuilding index: {result.stderr}")
                        st.code(result.stdout)
                except Exception as e:
                    st.error(f"❌ Error: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def get_categories():
    """Get list of categories including custom categories"""
    try:
        from utils.custom_categories import get_all_categories_with_custom
        return get_all_categories_with_custom()
    except Exception as e:
        # Fallback to standard categories
        try:
            from config.settings import settings
            return settings.get_category_names()
        except:
            return ['Shopping', 'Food & Dining', 'Transport', 'Entertainment', 'Bills & Payments', 'Banking', 'Other']


if __name__ == "__main__":
    main()

