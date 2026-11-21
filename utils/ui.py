"""Small shared UI helpers (theme colours, emoji, safe plotting) so the pages stay short."""
import streamlit as st

EMOJI = {"neutral": "😐", "calm": "😌", "happy": "😄", "sad": "😢",
         "angry": "😠", "fearful": "😨", "disgust": "🤢", "surprised": "😲"}
COLORS = {"neutral": "#9AA0A6", "calm": "#4DB6AC", "happy": "#FFD54F", "sad": "#64B5F6",
          "angry": "#E57373", "fearful": "#BA68C8", "disgust": "#81C784", "surprised": "#FFB74D"}
MODEL_COLORS = {"CNN": "#4C9BE8", "SVM (tuned)": "#F28E2B", "Random Forest (tuned)": "#59A14F"}


def show_plot(fig):
    """st.plotly_chart that works across Streamlit versions."""
    fig.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=10, r=10, t=50, b=10))
    try:
        st.plotly_chart(fig, width="stretch")
    except TypeError:
        st.plotly_chart(fig, use_container_width=True)


def sidebar_title():
    st.sidebar.markdown("### ⚙️ Control Panel")
