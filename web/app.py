"""
IntelliSpend v2.0 - Professional AI Transaction Categorization Platform
Enterprise-grade financial intelligence with beautiful, intuitive interface
"""

# Force CPU usage for PyTorch to avoid MPS tensor issues
import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

import streamlit as st
import pandas as pd
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import time
import os
import sys
from typing import Dict, Any

# Add project root to path for agent imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our enhanced agent system
try:
    from agents.agent_team import IntelliSpendAgentTeam
    from agents.feedback_agent import FeedbackAgent
    from models.transaction import TransactionData
    AGENT_SYSTEM_AVAILABLE = True
    st.session_state['agent_system_status'] = "✅ Enhanced Agent System Loaded"
except ImportError as e:
    AGENT_SYSTEM_AVAILABLE = False
    st.session_state['agent_system_status'] = f"❌ Agent System Error: {str(e)}"

# Set page config
st.set_page_config(
    page_title="IntelliSpend - AI Financial Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    .main {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 3.2rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f093fb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -0.02em;
        text-shadow: 0 4px 8px rgba(0,0,0,0.1);
        display: block;
        line-height: 1.2;
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: #4f46e5;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    
    .tagline {
        font-size: 1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 3rem;
        font-weight: 400;
        background: linear-gradient(135deg, rgba(102, 126, 234, 0.05) 0%, rgba(118, 75, 162, 0.05) 100%);
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid rgba(102, 126, 234, 0.1);
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 1.5rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
    }
    
    .metric-number {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .success-metric {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 10px 25px rgba(16, 185, 129, 0.2);
    }
    
    .warning-metric {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        box-shadow: 0 10px 25px rgba(245, 158, 11, 0.2);
    }
    
    .error-metric {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        box-shadow: 0 10px 25px rgba(239, 68, 68, 0.2);
    }
    
    .stTab {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1rem;
    }
    
    .result-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1) 0%, rgba(5, 150, 105, 0.05) 100%);
        border-radius: 16px;
        padding: 2rem;
        border: 1px solid rgba(16, 185, 129, 0.2);
        margin: 1.5rem 0;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }
    
    .result-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 35px rgba(16, 185, 129, 0.15);
    }
    
    .confidence-high {
        color: #10b981;
        font-weight: 600;
    }
    
    .confidence-medium {
        color: #f59e0b;
        font-weight: 600;
    }
    
    .confidence-low {
        color: #ef4444;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Category definitions for real agent system
CATEGORIES = {
    "🍽️ Food & Dining": ["restaurants", "fast_food", "cafes", "food_delivery", "groceries"],
    "🚗 Transportation": ["gas_stations", "uber_lyft", "public_transport", "car_services"],
    "🛍️ Shopping": ["retail", "online_shopping", "clothing", "electronics", "department_stores"],
    "🎬 Entertainment": ["streaming", "movies", "gaming", "music", "events"],
    "⚡ Utilities & Bills": ["electricity", "water", "internet", "phone", "insurance"],
    "🏥 Health & Medical": ["hospitals", "pharmacies", "doctors", "medical_services"],
    "🏠 Home & Garden": ["home_improvement", "furniture", "gardening", "maintenance"],
    "✈️ Travel": ["flights", "hotels", "car_rental", "travel_expenses"],
    "💵 Income": ["salary", "freelance", "investments", "refunds", "other_income"],
    "📋 Other": ["miscellaneous", "uncategorized"]
}

# Sample transaction data for demo
SAMPLE_TRANSACTIONS = [
    {"merchant": "Starbucks", "amount": 5.67, "description": "Coffee purchase"},
    {"merchant": "Apollo Pharmacy", "amount": 23.45, "description": "Medicine purchase"},
    {"merchant": "Amazon", "amount": 89.99, "description": "Online shopping"},
    {"merchant": "Shell Gas Station", "amount": 45.00, "description": "Fuel"},
    {"merchant": "Netflix", "amount": 15.99, "description": "Streaming subscription"},
    {"merchant": "Paradise Biryani", "amount": 34.50, "description": "Restaurant meal"},
    {"merchant": "Uber", "amount": 12.30, "description": "Ride sharing"},
    {"merchant": "CVS Pharmacy", "amount": 18.75, "description": "Pharmacy purchase"},
    {"merchant": "Target", "amount": 156.78, "description": "Retail shopping"},
    {"merchant": "McDonald's", "amount": 9.45, "description": "Fast food"},
    {"merchant": "Old Monk", "amount": 25.99, "description": "Beverage store"},
    {"merchant": "SBI Mutual Fund", "amount": 1000.00, "description": "Investment"},
    {"merchant": "Spotify", "amount": 9.99, "description": "Music streaming"},
    {"merchant": "Whole Foods", "amount": 87.34, "description": "Grocery shopping"},
    {"merchant": "Tesla Supercharger", "amount": 28.67, "description": "Electric vehicle charging"}
]

def generate_sample_data():
    """Generate sample transaction data for demo purposes"""
    return pd.DataFrame(SAMPLE_TRANSACTIONS)

def create_classification_method_stats(processed_results):
    """Create detailed statistics about classification methods used"""
    method_counts = {}
    for result in processed_results:
        method = result['method']  # Use clean method name
        if method not in method_counts:
            method_counts[method] = 0
        method_counts[method] += 1
    
    # Calculate percentages
    total = len(processed_results)
    method_stats = {}
    for method, count in method_counts.items():
        method_stats[method] = {
            'count': count,
            'percentage': (count / total) * 100
        }
    
    return method_stats

def process_user_feedback(merchant_name: str, original_category: str, correct_category: str, 
                          transaction_amount: float, user_comments: str = "") -> Dict[str, Any]:
    """Process user feedback through the enhanced knowledge-based system"""
    if not AGENT_SYSTEM_AVAILABLE:
        return {'success': False, 'error': 'Enhanced feedback system not available'}
    
    try:
        # Initialize enhanced feedback agent
        feedback_agent = FeedbackAgent()
        
        # Create mock transaction and prediction for feedback processing
        transaction = TransactionData(
            id=f"feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            merchant_name=merchant_name,
            amount=transaction_amount,
            description=f"Feedback correction for {merchant_name}",
            transaction_date=datetime.now()
        )
        
        # Create mock prediction
        from models.transaction import CategoryPrediction
        prediction = CategoryPrediction(
            category=original_category,
            confidence_score=0.5,  # Assume low confidence since it was corrected
            agent_name="llm_classification",
            reasoning=f"Original prediction for {merchant_name}"
        )
        
        # Process feedback
        user_feedback = {
            'correct_category': correct_category,
            'feedback_type': 'correction',
            'comments': user_comments
        }
        
        result = feedback_agent.process_feedback(transaction, prediction, user_feedback)
        
        if result['success']:
            # Get learning insights
            insights = feedback_agent.get_learning_insights(merchant_name)
            result['insights'] = insights
        
        return result
        
    except Exception as e:
        return {
            'success': False,
            'error': f"Feedback processing error: {str(e)}"
        }
    """Create detailed statistics about classification methods used"""
    method_counts = {}
    for result in processed_results:
        method = result['method']  # Use clean method name
        if method not in method_counts:
            method_counts[method] = 0
        method_counts[method] += 1
    
    # Calculate percentages
    total = len(processed_results)
    method_stats = {}
    for method, count in method_counts.items():
        method_stats[method] = {
            'count': count,
            'percentage': (count / total) * 100
        }
    
    return method_stats

def process_transactions_with_agents(uploaded_data):
    """Process transactions using our enhanced multi-agent system"""
    if not AGENT_SYSTEM_AVAILABLE:
        st.error("❌ Enhanced agent system not available. Please check system configuration.")
        return [], {}
    
    try:
        # Initialize the enhanced agent team
        with st.spinner("🤖 Initializing Enhanced AI Agent Team..."):
            agent_team = IntelliSpendAgentTeam()
        
        # Convert uploaded data to TransactionData objects
        transactions = []
        for _, row in uploaded_data.iterrows():
            transaction = TransactionData(
                id=f"web_{len(transactions)}",
                merchant_name=row.get('merchant', row.get('description', 'Unknown')),
                amount=float(row.get('amount', 0)),
                description=row.get('description', ''),
                transaction_date=datetime.now()
            )
            transactions.append(transaction)
        
        # Process through enhanced agent system
        with st.spinner("🧠 Processing transactions through AI agents..."):
            results = agent_team.process_transactions(transactions)
        
        if results['status'] == 'success':
            # Convert agent results to web format
            processed_results = []
            predictions = results['results']['predictions']
            
            for i, prediction in enumerate(predictions):
                # Check if this used learned feedback
                reasoning_lower = prediction.reasoning.lower()
                learned_from_feedback = any(indicator in reasoning_lower for indicator in [
                    'exact match', 'known merchant', 'learned', 'feedback'
                ])
                
                # Clean up method name for display
                clean_method = prediction.agent_name.replace('_', ' ').title()
                if 'llm' in clean_method.lower():
                    clean_method = '🧠 LLM Classification'
                elif 'rule' in clean_method.lower():
                    clean_method = '📋 Rule-based'
                elif 'embedding' in clean_method.lower():
                    clean_method = '🔗 Embedding Similarity'
                elif 'exact' in clean_method.lower():
                    clean_method = '🎯 Exact Match'
                elif 'similarity' in clean_method.lower():
                    clean_method = '🔗 Similarity Match'
                
                processed_results.append({
                    'id': transactions[i].id,
                    'merchant': transactions[i].merchant_name,
                    'amount': transactions[i].amount,
                    'category': prediction.category,
                    'confidence': prediction.confidence_score,
                    'method': clean_method,
                    'original_method': prediction.agent_name,
                    'reasoning': prediction.reasoning,
                    'learned_from_feedback': learned_from_feedback
                })
            
            # Extract statistics
            stats = results['results']['pipeline_stats']
            
            return processed_results, stats
        else:
            st.error(f"❌ Agent processing failed: {results.get('error', 'Unknown error')}")
            return [], {}
            
    except Exception as e:
        st.error(f"❌ Error in agent processing: {str(e)}")
        st.error("Please check your system configuration and try again.")
        return [], {}

def process_single_transaction(description, amount):
    """Process a single transaction through the agent system"""
    if not AGENT_SYSTEM_AVAILABLE:
        st.error("❌ Enhanced agent system not available. Please check system configuration.")
        return None
    
    try:
        # Initialize agent team
        agent_team = IntelliSpendAgentTeam()
        
        # Create transaction object
        transaction = TransactionData(
            id="web_single",
            merchant_name=description,
            amount=float(amount),
            description=f"Transaction at {description}",
            transaction_date=datetime.now(),
            account_id="web_account"
        )
        
        # Process through real agent system
        results = agent_team.process_transactions([transaction])
        
        if results['status'] == 'success' and results['results']['predictions']:
            prediction = results['results']['predictions'][0]
            
            # Check if this used learned feedback
            reasoning_lower = prediction.reasoning.lower()
            learned_from_feedback = any(indicator in reasoning_lower for indicator in [
                'exact match', 'known merchant', 'learned', 'feedback'
            ])
            
            # Clean up method name for display
            clean_method = prediction.agent_name.replace('_', ' ').title()
            if 'llm' in clean_method.lower():
                clean_method = '🧠 LLM Classification'
            elif 'rule' in clean_method.lower():
                clean_method = '📋 Rule-based'
            elif 'embedding' in clean_method.lower():
                clean_method = '🔗 Embedding Similarity'
            elif 'exact' in clean_method.lower():
                clean_method = '🎯 Exact Match'
            elif 'similarity' in clean_method.lower():
                clean_method = '🔗 Similarity Match'
            
            return {
                'category': prediction.category,
                'confidence': prediction.confidence_score,
                'method': clean_method,
                'reasoning': prediction.reasoning,
                'learned_from_feedback': learned_from_feedback
            }
        else:
            st.error("❌ Failed to process transaction. Please try again.")
            return None
    except Exception as e:
        st.error(f"❌ Agent system error: {str(e)}")
        st.error("Please check your system configuration and try again.")
        return None

def create_results_visualization(results_df, stats, processed_results):
    """Create enhanced visualizations for transaction results"""
    if results_df.empty:
        st.warning("No data to visualize")
        return
    
    # Create tabs for different visualizations
    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["📊 Categories", "🤖 Methods", "📈 Confidence"])
    
    with viz_tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Category distribution
            category_counts = results_df['category'].value_counts()
            st.markdown("#### 📊 Category Distribution")
            fig_pie = px.pie(
                values=category_counts.values,
                names=category_counts.index,
                title="Transaction Categories",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            fig_pie.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Amount by category
            category_amounts = results_df.groupby('category')['amount'].sum().sort_values(ascending=True)
            st.markdown("#### 💰 Spending by Category")
            fig_bar = px.bar(
                x=category_amounts.values,
                y=category_amounts.index,
                orientation='h',
                title="Total Amount by Category",
                color_discrete_sequence=['#667eea']
            )
            fig_bar.update_layout(
                xaxis_title="Amount ($)",
                yaxis_title="Category"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with viz_tab2:
        # Classification method statistics
        method_stats = create_classification_method_stats(processed_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 🤖 Classification Methods Used")
            
            # Create method breakdown chart
            method_names = list(method_stats.keys())
            method_counts = [stats['count'] for stats in method_stats.values()]
            method_percentages = [stats['percentage'] for stats in method_stats.values()]
            
            # Clean up method names for display
            clean_names = []
            for name in method_names:
                if 'llm' in name.lower():
                    clean_names.append('🧠 LLM Classification')
                elif 'rule' in name.lower():
                    clean_names.append('📋 Rule-based')
                elif 'web' in name.lower():
                    clean_names.append('� Web Search Enhanced')
                elif 'embedding' in name.lower():
                    clean_names.append('🔗 Embedding Similarity')
                else:
                    clean_names.append(f'⚙️ {name.replace("_", " ").title()}')
            
            fig_methods = px.pie(
                values=method_counts,
                names=clean_names,
                title="AI Classification Methods",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            fig_methods.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig_methods, use_container_width=True)
        
        with col2:
            st.markdown("#### 📊 Method Performance Details")
            
            # Method performance table
            method_df = pd.DataFrame({
                'Method': clean_names,
                'Count': method_counts,
                'Percentage': [f"{p:.1f}%" for p in method_percentages]
            })
            
            st.dataframe(method_df, use_container_width=True)
            
            # LLM vs Others breakdown
            llm_count = sum([stats['count'] for method, stats in method_stats.items() if 'llm' in method.lower()])
            other_count = len(processed_results) - llm_count
            
            st.markdown("#### 🧠 LLM vs Other Methods")
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric(
                    "LLM Classified",
                    llm_count,
                    delta=f"{(llm_count/len(processed_results)*100):.1f}%"
                )
            with col_b:
                st.metric(
                    "Other Methods",
                    other_count,
                    delta=f"{(other_count/len(processed_results)*100):.1f}%"
                )
    
    with viz_tab3:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📈 Confidence Distribution")
            fig_hist = px.histogram(
                results_df, 
                x='confidence',
                title="Classification Confidence Scores",
                nbins=20,
                color_discrete_sequence=['#667eea']
            )
            fig_hist.update_layout(
                xaxis_title="Confidence Score",
                yaxis_title="Number of Transactions"
            )
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            # Confidence by method
            st.markdown("#### 🎯 Confidence by Method")
            conf_by_method = results_df.groupby('method')['confidence'].mean().sort_values(ascending=True)
            fig_conf = px.bar(
                x=conf_by_method.values,
                y=[name.replace('_', ' ').title() for name in conf_by_method.index],
                orientation='h',
                title="Average Confidence by Method",
                color_discrete_sequence=['#10b981']
            )
            fig_conf.update_layout(
                xaxis_title="Average Confidence",
                yaxis_title="Classification Method"
            )
            st.plotly_chart(fig_conf, use_container_width=True)

def main():
    """Main application function"""
    
    # Initialize session state for feedback forms and prevent resets
    if 'feedback_state' not in st.session_state:
        st.session_state.feedback_state = {}
    if 'form_selections' not in st.session_state:
        st.session_state.form_selections = {}
    
    # System status check
    if not AGENT_SYSTEM_AVAILABLE:
        st.error("❌ IntelliSpend Agent System Not Available")
        st.error("Please ensure the agent system is properly configured and try again.")
        st.code(st.session_state['agent_system_status'])
        st.stop()
    
    # Header with enhanced styling
    st.markdown('<div class="main-header">💰 IntelliSpend</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">🤖 AI-Powered Financial Transaction Intelligence Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="tagline">✨ Smart categorization • 🧠 Machine learning • 📚 Continuous learning • 📊 Real-time analytics</div>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Processing Options")
        
        # Mode selection
        processing_mode = st.radio(
            "Mode:",
            ["🔄 Real-time", "📦 Batch", "🔀 Hybrid"],
            help="Select processing mode"
        )
        
        # Confidence threshold
        confidence_threshold = st.slider(
            "Min Confidence:",
            min_value=0.0,
            max_value=1.0,
            value=0.70,
            step=0.05,
            help="Minimum confidence threshold for classifications"
        )
        
        # Enable learning
        enable_learning = st.checkbox(
            "🧠 Enable Learning",
            value=True,
            help="Allow system to learn from feedback"
        )
        
        st.markdown("---")
        st.markdown(f"**System Status:** {st.session_state['agent_system_status']}")
        
        # Add feedback analytics if available
        if st.button("📊 View Learning Analytics", help="See how the AI is learning from feedback"):
            try:
                feedback_agent = FeedbackAgent()
                analytics = feedback_agent.get_feedback_analytics()
                
                if not analytics.get('error'):
                    st.markdown("### 🧠 AI Learning Progress")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Total Feedback Entries",
                            analytics.get('total_feedback_entries', 0)
                        )
                    
                    with col2:
                        corrections = analytics.get('most_corrected_categories', {})
                        if corrections:
                            top_correction = list(corrections.keys())[0]
                            st.metric(
                                "Top Correction Pattern",
                                top_correction.split(' → ')[0],
                                delta=f"→ {top_correction.split(' → ')[1]}"
                            )
                    
                    if corrections:
                        st.markdown("**Most Common Corrections:**")
                        for pattern, count in list(corrections.items())[:3]:
                            st.write(f"• {pattern} ({count} times)")
                else:
                    st.info("📊 No learning analytics available yet. Submit some feedback to see insights!")
                    
            except Exception as e:
                st.error(f"Analytics unavailable: {str(e)}")
        
        # Simple reset feedback option without confirmation
        st.markdown("---")
        if st.button("🗑️ Reset Feedback Data", help="Clear all stored feedback and reset the AI learning"):
            try:
                feedback_agent = FeedbackAgent()
                result = feedback_agent.reset_feedback_storage()
                
                if result.get('success'):
                    st.success(f"✅ {result.get('message')}")
                    st.info("🔄 The AI will now classify transactions without any learned patterns from feedback.")
                    # Force a rerun to update everything immediately
                    st.rerun()
                else:
                    st.error(f"❌ Failed to reset feedback: {result.get('error')}")
            except Exception as e:
                st.error(f"❌ Error resetting feedback: {str(e)}")
    
    # Main interface
    tab1, tab2, tab3 = st.tabs(["📁 File Upload", "🎯 Demo Data", "✏️ Single Transaction"])
    
    with tab1:
        st.markdown("#### 📁 Upload Transaction Data")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your transaction data in CSV format"
        )
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.success(f"✅ Loaded {len(df)} transactions")
            
            with st.expander("📋 Data Preview"):
                st.dataframe(df.head())
            
            if st.button("🚀 Process Transactions", type="primary"):
                with st.spinner("🔄 Processing transactions through AI agent system..."):
                    processed_results, stats = process_transactions_with_agents(df)
                
                if processed_results:
                    st.success(f"✅ Successfully processed {len(processed_results)} transactions!")
                    
                    # Convert to DataFrame for display
                    results_df = pd.DataFrame(processed_results)
                    
                    # Display results
                    st.markdown("### 📊 Classification Results")
                    
                    # Metrics
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric(
                            "Total Processed", 
                            len(processed_results)
                        )
                    
                    with col2:
                        high_confidence = len([r for r in processed_results if r['confidence'] >= confidence_threshold])
                        st.metric(
                            "High Confidence",
                            high_confidence,
                            delta=f"{high_confidence/len(processed_results)*100:.1f}%"
                        )
                    
                    with col3:
                        avg_confidence = sum(r['confidence'] for r in processed_results) / len(processed_results)
                        st.metric(
                            "Avg Confidence",
                            f"{avg_confidence:.2%}"
                        )
                    
                    with col4:
                        learned_count = len([r for r in processed_results if r.get('learned_from_feedback', False)])
                        st.metric(
                            "📚 Learned",
                            learned_count,
                            delta=f"{(learned_count/len(processed_results)*100):.0f}%"
                        )
                    
                    # Data table
                    st.markdown("#### 📋 Detailed Results")
                    display_df = pd.DataFrame({
                        'Merchant': [r['merchant'][:30] + "..." if len(r['merchant']) > 30 else r['merchant'] for r in processed_results],
                        'Amount': [f"${abs(r['amount']):,.2f}" for r in processed_results],
                        'Category': [r['category'] for r in processed_results],
                        'Confidence': [f"{r['confidence']:.1%}" for r in processed_results],
                        'Method': [r['method'] for r in processed_results],
                        'Learned': ['📚 Yes' if r.get('learned_from_feedback', False) else '🆕 New' for r in processed_results]
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Add feedback option for batch results
                    st.markdown("#### 📝 Provide Feedback for Better Learning")
                    with st.expander("🧠 Help AI Learn - Correct Any Misclassifications", expanded=False):
                        # Create options for selection - OUTSIDE form for dynamic updates
                        options = [f"{r['merchant']} (${abs(r['amount']):.2f}) → {r['category']}" for r in processed_results]
                        
                        selected_merchant = st.selectbox(
                            "Select a transaction to provide feedback:",
                            options=options,
                            key=f"batch_feedback_select_{len(processed_results)}"
                        )
                        
                        if selected_merchant:
                            # Extract merchant name from selection
                            merchant_name = selected_merchant.split(' (')[0]
                            selected_result = next((r for r in processed_results if r['merchant'] == merchant_name), None)
                            
                            if selected_result:
                                feedback_type = st.radio(
                                    "Is this classification correct?",
                                    ["✅ Correct", "❌ Incorrect"],
                                    key=f"batch_feedback_type_{len(processed_results)}"
                                )
                                
                                all_categories = [
                                    "Food & Dining", "Transportation", "Shopping", "Entertainment",
                                    "Bills & Utilities", "Health & Medical", "Home & Garden", 
                                    "Travel", "Income", "Other"
                                ]
                                
                                # Show dropdown and text area immediately when Incorrect is selected
                                if feedback_type == "❌ Incorrect":
                                    correct_category = st.selectbox(
                                        "What's the correct category?",
                                        all_categories,
                                        key=f"batch_correct_category_{len(processed_results)}"
                                    )
                                    
                                    comments = st.text_area(
                                        "Additional feedback (optional):",
                                        key=f"batch_feedback_comments_{len(processed_results)}",
                                        placeholder="Why should this be categorized differently?"
                                    )
                                    
                                    # Submit button for incorrect feedback
                                    if st.button("📤 Submit Feedback", key=f"batch_submit_incorrect_{len(processed_results)}", type="primary"):
                                        feedback_result = process_user_feedback(
                                            merchant_name=selected_result['merchant'],
                                            original_category=selected_result['category'],
                                            correct_category=correct_category,
                                            transaction_amount=selected_result['amount'],
                                            user_comments=comments
                                        )
                                        
                                        if feedback_result.get('success'):
                                            st.success("🎉 Thank you! Your feedback helps improve the AI.")
                                        else:
                                            st.error(f"❌ Feedback failed: {feedback_result.get('error')}")
                                else:
                                    # Submit button for correct confirmation
                                    if st.button("👍 Confirm Correct", key=f"batch_submit_correct_{len(processed_results)}", type="secondary"):
                                        st.success("✅ Thank you for confirming the correct classification!")
                    
                    # Visualizations
                    st.markdown("#### 📈 Analytics")
                    create_results_visualization(results_df, stats, processed_results)
    
    with tab2:
        st.markdown("#### 🎯 Demo Data - Test the AI System")
        st.info("🚀 **Demo Mode**: Use our sample data to test the enhanced AI classification system with web search capabilities!")
        
        col_demo1, col_demo2 = st.columns(2)
        
        with col_demo1:
            if st.button("📦 Load Sample Data", type="secondary"):
                st.session_state['demo_data'] = generate_sample_data()
                st.success("✅ Sample data loaded! Click 'Process Demo Data' to see AI in action.")
        
        with col_demo2:
            if st.button("🎲 Generate Random Data", type="secondary"):
                # Generate random variations of sample data
                demo_data = generate_sample_data()
                # Add some randomization to amounts
                demo_data['amount'] = demo_data['amount'] * (0.8 + 0.4 * np.random.random(len(demo_data)))
                st.session_state['demo_data'] = demo_data
                st.success("✅ Random demo data generated! Click 'Process Demo Data' to test.")
        
        if 'demo_data' in st.session_state:
            st.markdown("#### 📋 Demo Data Preview")
            st.dataframe(st.session_state['demo_data'], use_container_width=True)
            
            if st.button("🚀 Process Demo Data", type="primary"):
                with st.spinner("🧠 Processing demo transactions through AI agent system..."):
                    processed_results, stats = process_transactions_with_agents(st.session_state['demo_data'])
                
                if processed_results:
                    st.success(f"✅ Successfully processed {len(processed_results)} demo transactions!")
                    
                    # Convert to DataFrame for display
                    results_df = pd.DataFrame(processed_results)
                    
                    # Enhanced metrics for demo
                    st.markdown("### 📊 Demo Classification Results")
                    
                    col1, col2, col3, col4, col5 = st.columns(5)
                    
                    # Calculate method statistics
                    method_stats = create_classification_method_stats(processed_results)
                    llm_count = sum([stats['count'] for method, stats in method_stats.items() if 'llm' in method.lower()])
                    web_search_count = sum([stats['count'] for method, stats in method_stats.items() if 'web' in method.lower()])
                    rule_count = sum([stats['count'] for method, stats in method_stats.items() if 'rule' in method.lower()])
                    
                    with col1:
                        st.metric("Total Processed", len(processed_results))
                    
                    with col2:
                        st.metric(
                            "🧠 LLM Classified",
                            llm_count,
                            delta=f"{(llm_count/len(processed_results)*100):.0f}%"
                        )
                    
                    with col3:
                        st.metric(
                            "� Web Enhanced",
                            web_search_count,
                            delta=f"{(web_search_count/len(processed_results)*100):.0f}%"
                        )
                    
                    with col4:
                        st.metric(
                            "📋 Rule-based",
                            rule_count,
                            delta=f"{(rule_count/len(processed_results)*100):.0f}%"
                        )
                    
                    with col5:
                        avg_confidence = sum(r['confidence'] for r in processed_results) / len(processed_results)
                        st.metric(
                            "Avg Confidence",
                            f"{avg_confidence:.1%}"
                        )
                    
                    # Detailed results table
                    st.markdown("#### 📋 Detailed Demo Results")
                    display_df = pd.DataFrame({
                        'Merchant': [r['merchant'] for r in processed_results],
                        'Amount': [f"${abs(r['amount']):,.2f}" for r in processed_results],
                        'Category': [r['category'] for r in processed_results],
                        'Confidence': [f"{r['confidence']:.1%}" for r in processed_results],
                        'Method': [r['method'] for r in processed_results],
                        'Web Search': ['🔍 Yes' if r.get('web_search_used', False) else '❌ No' for r in processed_results],
                        'AI Reasoning': [r['reasoning'][:50] + "..." if len(r['reasoning']) > 50 else r['reasoning'] for r in processed_results]
                    })
                    
                    st.dataframe(display_df, use_container_width=True)
                    
                    # Enhanced visualizations
                    st.markdown("#### 📈 Demo Analytics & AI Method Breakdown")
                    create_results_visualization(results_df, stats, processed_results)
                    
                    # Add feedback option for demo results
                    st.markdown("#### 📝 Demo Feedback - Help AI Learn")
                    with st.expander("🧠 Provide Feedback on Demo Classifications", expanded=False):
                        # Create unique key for demo selection with persistent session state
                        demo_select_key = f"demo_feedback_select_{len(processed_results)}"
                        
                        options = [f"{r['merchant']} → {r['category']}" for r in processed_results[:10]]  # Show first 10
                        
                        # Initialize and maintain selection in session state
                        if demo_select_key not in st.session_state:
                            st.session_state[demo_select_key] = options[0] if options else ""
                        
                        # Find current index based on session state value
                        try:
                            current_index = options.index(st.session_state[demo_select_key]) if st.session_state[demo_select_key] in options else 0
                        except (ValueError, IndexError):
                            current_index = 0
                            st.session_state[demo_select_key] = options[0] if options else ""
                        
                        demo_merchant = st.selectbox(
                            "Select a demo transaction to provide feedback:",
                            options=options,
                            index=current_index,
                            key=demo_select_key
                        )
                        
                        # Update session state when selection changes
                        if demo_merchant != st.session_state.get(demo_select_key):
                            st.session_state[demo_select_key] = demo_merchant
                        
                        if demo_merchant:
                            merchant_name = demo_merchant.split(' → ')[0]
                            demo_result = next((r for r in processed_results if r['merchant'] == merchant_name), None)
                            
                            if demo_result:
                                # Create unique key for demo feedback
                                demo_feedback_key = f"demo_{merchant_name}_{demo_result['amount']}".replace(" ", "_").replace(".", "_")
                                
                                is_correct = st.radio(
                                    f"Is '{demo_result['category']}' correct for '{merchant_name}'?",
                                    ["✅ Correct", "❌ Incorrect"],
                                    key=f"demo_feedback_radio_{demo_feedback_key}"
                                )
                                
                                demo_categories = ["Food & Dining", "Transportation", "Shopping", "Entertainment",
                                     "Bills & Utilities", "Health & Medical", "Home & Garden", "Travel", "Income", "Other"]
                                
                                # Show dropdown immediately when Incorrect is selected
                                if is_correct == "❌ Incorrect":
                                    correct_cat = st.selectbox(
                                        "What's the correct category?",
                                        demo_categories,
                                        key=f"demo_correct_category_{demo_feedback_key}",
                                        help="Select the correct category for this demo transaction"
                                    )
                                    
                                    # Submit button for incorrect feedback
                                    if st.button("📤 Submit Demo Feedback", key=f"demo_submit_incorrect_{demo_feedback_key}", type="primary"):
                                        feedback_result = process_user_feedback(
                                            merchant_name=demo_result['merchant'],
                                            original_category=demo_result['category'],
                                            correct_category=correct_cat,
                                            transaction_amount=demo_result['amount']
                                        )
                                        
                                        if feedback_result.get('success'):
                                            st.success("🎉 Demo feedback recorded! This will help improve future classifications.")
                                        else:
                                            st.error(f"❌ Feedback failed: {feedback_result.get('error')}")
                                else:
                                    # Submit button for correct confirmation
                                    if st.button("👍 Confirm Correct", key=f"demo_submit_correct_{demo_feedback_key}", type="secondary"):
                                        st.success("✅ Thank you for confirming the correct classification!")
                    
                    # Highlight interesting findings
                    st.markdown("#### 🔍 Key Findings from Demo")
                    
                    findings_col1, findings_col2 = st.columns(2)
                    
                    with findings_col1:
                        st.info("🧠 **LLM Performance**: Advanced language model handles complex merchant names and provides detailed reasoning for classifications.")
                        
                        web_enhanced = [r for r in processed_results if 'web' in r.get('method', '').lower()]
                        if web_enhanced:
                            st.success(f"🔍 **Web Search Enhancement**: {len(web_enhanced)} transactions were enhanced using real-time web search for unknown merchants.")
                    
                    with findings_col2:
                        high_conf = [r for r in processed_results if r['confidence'] > 0.8]
                        st.metric(
                            "🎯 High Confidence Rate",
                            f"{len(high_conf)/len(processed_results)*100:.0f}%",
                            delta="Quality Assurance"
                        )
                        
                        if avg_confidence > 0.75:
                            st.success("✅ **Excellent Performance**: AI system demonstrates high accuracy and confidence across diverse transaction types.")
    
    with tab3:
        st.markdown("#### ✏️ Single Transaction Entry")
        
        with st.form("single_transaction"):
            col_i, col_ii = st.columns(2)
            with col_i:
                description = st.text_input(
                    "Description*", 
                    "",
                    help="Enter the merchant or transaction description"
                )
                amount = st.number_input(
                    "Amount*", 
                    value=0.0,
                    help="Transaction amount"
                )
            with col_ii:
                account_type = st.selectbox(
                    "Account Type", 
                    ["debit", "credit", "autopay", "transfer", "cash"]
                )
                location = st.text_input(
                    "Location", 
                    "",
                    help="Transaction location (optional)"
                )
            
            submitted = st.form_submit_button("🔍 Categorize Transaction", type="primary")
        
        # Process results outside the form
        if submitted and description:
            with st.spinner("🤖 Processing with AI agents..."):
                result = process_single_transaction(description, amount)
                
                if result:
                    # Store result in session state to maintain it across interactions
                    st.session_state['transaction_result'] = result
                    st.session_state['transaction_description'] = description
                    st.session_state['transaction_amount'] = amount
        
        # Display results if available in session state
        if 'transaction_result' in st.session_state:
            result = st.session_state['transaction_result']
            description = st.session_state['transaction_description']
            amount = st.session_state['transaction_amount']
            
            # Enhanced result display
            st.markdown("#### 📋 Classification Results")
            
            col_x, col_y, col_z, col_w = st.columns(4)
            with col_x:
                st.metric(
                    "Category", 
                    result['category'],
                    help="Primary classification category"
                )
            with col_y:
                st.metric(
                    "Method", 
                    result['method'].replace('_', ' ').title(),
                    help="Classification method used"
                )
            with col_z:
                confidence_color = "🟢" if result['confidence'] > 0.8 else "🟡" if result['confidence'] > 0.6 else "🔴"
                st.metric(
                    "Confidence", 
                    f"{confidence_color} {result['confidence']:.1%}",
                    help="AI confidence score"
                )
            with col_w:
                learned_indicator = "📚 Yes" if result.get('learned_from_feedback', False) else "🆕 New"
                st.metric(
                    "From Learning",
                    learned_indicator,
                    help="Whether this was learned from previous feedback"
                )
            
            # Reasoning
            st.info(f"💡 **AI Reasoning**: {result['reasoning']}")
            
            # Enhanced feedback section
            st.markdown("#### 📝 Help the AI Learn - Knowledge-Based Feedback")
            st.info("🧠 **Smart Learning**: Your feedback is stored and helps the AI learn patterns for better future classifications!")
            
            # Use form to prevent immediate rerun on selections
            feedback_col1, feedback_col2 = st.columns(2)
            
            # Create unique key for this transaction's feedback
            feedback_key = f"single_{description}_{amount}".replace(" ", "_").replace(".", "_")
            
            with feedback_col1:
                feedback_type = st.radio(
                    "Is this classification correct?",
                    ["✅ Correct", "❌ Incorrect", "🤔 Partially Correct"],
                    horizontal=True,
                    key=f"feedback_radio_{feedback_key}"
                )
            
            with feedback_col2:
                if feedback_type != "✅ Correct":
                    # Category selector for corrections
                    all_categories = [
                        "Food & Dining", "Transportation", "Shopping", "Entertainment",
                        "Utilities & Bills", "Health & Medical", "Home & Garden", 
                        "Travel", "Income", "Other"
                    ]
                    
                    correct_category = st.selectbox(
                        "What's the correct category?",
                        all_categories,
                        key=f"correct_category_{feedback_key}",
                        help="Select the correct category for this transaction",
                        index=0  # Always start with first option to prevent index errors
                    )
            
            user_comments = ""
            if feedback_type != "✅ Correct":
                user_comments = st.text_area(
                    "Additional feedback (optional):",
                    placeholder="Help us understand why this should be categorized differently...",
                    key=f"feedback_comments_{feedback_key}",
                    help="Your comments help improve the AI's understanding"
                )
            
            # Always show submit button
            if feedback_type != "✅ Correct":
                submitted_feedback = st.button("📤 Submit Smart Feedback", type="primary", key=f"submit_{feedback_key}")
            else:
                submitted_feedback = st.button("👍 Confirm Correct Classification", type="secondary", key=f"confirm_{feedback_key}")
            
            # Process feedback immediately when button clicked
            if submitted_feedback:
                if feedback_type == "✅ Correct":
                    st.success("✅ Thank you for confirming the correct classification!")
                else:
                    with st.spinner("🧠 Processing feedback through knowledge base..."):
                        feedback_result = process_user_feedback(
                            merchant_name=description,
                            original_category=result['category'],
                            correct_category=correct_category,
                            transaction_amount=amount,
                            user_comments=user_comments if user_comments else ""
                        )
                    
                    if feedback_result.get('success'):
                        st.success("🎉 Thank you! Your feedback has been stored in our knowledge base.")
                        st.info(f"💡 {feedback_result.get('message')}")
                        
                        # Show learning insights if available
                        insights = feedback_result.get('insights', {})
                        if insights and insights.get('similar_corrections'):
                            with st.expander("🔍 Learning Insights", expanded=False):
                                st.write("**Similar corrections found:**")
                                for correction in insights['similar_corrections'][:3]:
                                    st.write(f"• {correction.get('merchant', 'Unknown')} → {correction.get('correct_category', 'Unknown')}")
                                
                                st.write("The AI is learning from these patterns to improve future classifications!")
                    else:
                        st.error(f"❌ Feedback submission failed: {feedback_result.get('error')}")
            
            # Add a button to clear results and try a new transaction
            if st.button("🔄 Try New Transaction", key="clear_results"):
                # Clear session state
                for key in ['transaction_result', 'transaction_description', 'transaction_amount']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()

if __name__ == "__main__":
    main()