"""Local music studio presentation, scoped to the generation-only interface."""

CSS = """
.gradio-container {width: 96% !important; max-width: 1800px !important;
    margin: auto !important; padding: 20px 24px !important; box-sizing: border-box;}
.main-header {text-align: left !important; padding: 28px 8px 12px; margin-bottom: 8px !important;}
.main-header h1 {font-size: clamp(26px, 4vw, 38px); letter-spacing: -0.03em;}
.main-header p {opacity: .7; font-size: 15px; line-height: 1.8;}
#music-prompt textarea {font-size: 17px; line-height: 1.8; min-height: 130px;}
#music-create {min-height: 56px; border-radius: 16px; font-size: 18px;}
#music-status {padding: 12px 16px; border-radius: 12px; border: 1px solid var(--border-color-primary);}
#music-workspace {gap: 28px; align-items: flex-start;}
#music-input-panel, #music-output-panel {min-width: 0;}
@media (min-width: 1000px) {
    #music-workspace {flex-wrap: nowrap !important;}
    #music-output-panel {position: sticky; top: 20px;}
}
@media (max-width: 999px) {
    #music-workspace {flex-direction: column;}
    #music-input-panel, #music-output-panel {width: 100% !important;}
}
.section-header {background: var(--background-fill-secondary) !important;
    color: var(--body-text-color) !important; border-radius: 12px !important;}
button, textarea, input {transition: border-color .15s, box-shadow .15s;}
@media (max-width: 640px) {.gradio-container {width: 100% !important; padding: 12px !important;}}
"""

HEADER = """
<div class="main-header">
  <p>個人化音樂工作室 · ACE-Step 1.5</p>
  <h1>讓聲音，回到你的節奏。</h1>
  <p>一段描述，創作屬於你的純樂器音樂。支援鋼琴、頌缽與自由配器。</p>
</div>
"""
