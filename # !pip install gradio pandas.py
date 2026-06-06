# !pip install gradio pandas

import gradio as gr
import pandas as pd
import random
import os

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 1. 블랙 & 네온 그린 (Dark Cyberpunk / Terminal Style) 커스텀 CSS
# ━━━━━━━━━━━━━━━━━━━━━━━━━
black_green_css = """
/* 전체 배경을 깊은 블랙으로 설정 */
body, .gradio-container {
    background-color: #0A0A0C !important;
    color: #E0E0E0 !important;
    font-family: 'Pretendard', -apple-system, sans-serif !important;
}

/* 메인 컨테이너 테두리에 은은한 그린 글로우 효과 */
.gradio-container {
    border: 2px solid #00FF66 !important;
    border-radius: 16px !important;
    box-shadow: 0 0 20px rgba(0, 255, 102, 0.15) !important;
    padding: 20px !important;
}

/* 입력창, 결과창, 탭 내부 블록들을 어두운 그레이와 그린 테두리로 통일 */
.block, .form, fieldset, .gradio-textbox, .gradio-label, .gradio-dataframe, .tabs {
    background-color: #141417 !important;
    border: 1px solid #222226 !important;
    color: #E0E0E0 !important;
    border-radius: 12px !important;
}

/* 텍스트 입력창 내부 스타일 */
textarea, input[type="text"] {
    background-color: #0F0F12 !important;
    color: #FFFFFF !important;
    border: 1px solid #2C2C35 !important;
}
textarea:focus, input[type="text"]:focus {
    border-color: #00FF66 !important;
    box-shadow: 0 0 8px rgba(0, 255, 102, 0.3) !important;
}

/* 네온 그린 시스템 버튼 */
button.primary {
    background-color: #00FF66 !important;
    color: #0A0A0C !important;
    font-weight: bold !important;
    border-radius: 10px !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(0, 255, 102, 0.3) !important;
    transition: all 0.2s ease;
}
button.primary:hover {
    background-color: #00CC52 !important;
    box-shadow: 0 4px 25px rgba(0, 255, 102, 0.5) !important;
    cursor: pointer;
}

/* 데이터프레임(표) 디자인 수정: 헤더는 그린, 셀은 화이트 */
.gradio-dataframe table th {
    background-color: #1A1A22 !important;
    color: #00FF66 !important;
    font-weight: bold !important;
}
.gradio-dataframe table td {
    background-color: #141417 !important;
    color: #FFFFFF !important;
}

/* 탭 버튼 스타일 */
.tabs .tab-nav button {
    color: #8E8E93 !important;
    font-weight: bold !important;
}
.tabs .tab-nav button.selected {
    color: #00FF66 !important;
    border-bottom: 2px solid #00FF66 !important;
    background-color: #141417 !important;
}

/* 링크 및 강조 텍스트 */
a, h1, h2, h3 {
    color: #00FF66 !important;
}
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 2. AI 모델 예측 모의 (Mock)
# ━━━━━━━━━━━━━━━━━━━━━━━━━
def _predict(text):
    danger_keywords = ["원금 보장", "급등", "상한가", "미공개", "무조건", "작전주", "리딩방", "수익률", "폭등"]
    if any(kw in text for kw in danger_keywords):
        prob_danger = random.uniform(0.7, 0.99)
    else:
        prob_danger = random.uniform(0.01, 0.3)
    prob_normal = 1.0 - prob_danger
    pred = 1 if prob_danger > 0.5 else 0
    return pred, prob_normal, prob_danger

MODEL_CARD = """
# 📄 Model Card: 금융 사기 및 가짜뉴스 판별기 (KoELECTRA + LoRA)

## 모델 개요
* **Task:** 금융/주식 관련 텍스트의 불법 리딩방·허위 정보 이진 분류 (정상 0 / 위험 1)
* **Training Data:** 소규모 금융 사기 홍보물 및 정상 금융 기사 데이터

## ⚠️ 한계점 및 주의사항
1. **편향성:** 특정 사기성 키워드(예: 급등, 원금 보장)에만 민감하게 반응할 수 있습니다.
2. **취약성:** 교묘한 재무제표 왜곡, 합법을 위장한 투자 권유 등은 판별이 어렵습니다.
3. **책임 제한:** 본 AI의 결과는 절대적인 투자 지표로 사용될 수 없습니다.
"""

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 3. 업로드된 CSV 파일 로드 (데이터 전처리 및 변수 자동 매핑)
# ━━━━━━━━━━━━━━━━━━━━━━━━━
stress_file = 'stress_test_results.csv'
failures_file = 'stress_test_failures.csv'

df_stress_display = pd.DataFrame()
df_failures_display = pd.DataFrame()
overall_acc = 0.0
HAS_STRESS = False

try:
    if os.path.exists(stress_file):
        df_stress = pd.read_csv(stress_file)
        if 'text' in df_stress.columns:
            # 원본 CSV 데이터 컬럼 그대로 매핑하여 한글화 표시
            df_stress_display = df_stress[['text', 'label', 'pred', 'correct', 'test_type']].rename(
                columns={'text': '문장', 'label': '실제', 'pred': '예측', 'correct': '정답', 'test_type': '유형'}
            )

            # 정확도 계산 (correct 컬럼 기준 계산)
            if 'correct' in df_stress.columns:
                correct_series = df_stress['correct'].map({'True': True, 'False': False, True: True, False: False})
                overall_acc = correct_series.mean()
            HAS_STRESS = True

    if os.path.exists(failures_file):
        df_failures = pd.read_csv(failures_file)
        if 'text' in df_failures.columns:
            df_failures_display = df_failures[['text', 'label', 'pred', 'test_type']].rename(
                columns={'text': '문장', 'label': '실제', 'pred': '예측', 'test_type': '유형'}
            )
except Exception as e:
    print(f"⚠️ CSV 파일 연동 실패 (기본 화면으로 대체됩니다): {e}")
    HAS_STRESS = False

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# Tab 1: 실시간 판별기 함수
# ━━━━━━━━━━━━━━━━━━━━━━━━━
def classify(text, is_agreed):
    if not is_agreed:
        return {}, '🚫 접근 거부', '⚠️ 아래 "Model Card 한계 고지" 동의 체크박스를 클릭해주세요.'

    if not text.strip():
        return {}, '⚠️ 텍스트 입력 없음', '판별할 문장을 입력하세요.'

    pred, prob_normal, prob_danger = _predict(text)
    label_probs = {'🟢 정상 (안전)': prob_normal, '🔴 위험 (사기/허위)': prob_danger}

    if pred == 1:
        verdict = f'🔴 위험 판정 (확률: {prob_danger:.1%})'
        detail = '⚠️ 불법 리딩방, 주식 사기 또는 과장 광고의 특징이 감지되었습니다. 금융감독원 등록 업체인지 교차 확인하세요.'
    else:
        verdict = f'🟢 정상 판정 (확률: {prob_normal:.1%})'
        detail = '명백한 사기 패턴은 감지되지 않았으나, 투자 결정 전 항상 비판적 사고를 유지하세요.'

    return label_probs, verdict, detail

EXAMPLES = [
    ['[단독] 삼성전자 핵심 기술 인수 확정... 내일 상한가 직행합니다. 미리 매집하세요.'],
    ['금융감독원은 최근 기승을 부리는 불법 주식 리딩방에 대해 소비자 경보를 발령했다.'],
    ['미공개 정보 입수! 한 달 내 300% 수익 원금 보장합니다. 지금 입장하세요.']
]

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 4. Gradio UI 조립 (기존 구조 완전 유지)
# ━━━━━━━━━━━━━━━━━━━━━━━━━
with gr.Blocks(title='금융 사기 판별 AI', css=black_green_css, theme=gr.themes.Default()) as demo:

    # 헤더 섹션
    gr.HTML("""
    <div style="text-align:center; padding:15px 0;">
      <h1 style="font-weight: 800; font-size: 2.2rem; letter-spacing: -0.5px;">1조 루나테라 편향도 측정 프로그램</h1>
      <h3 style="color: #A0A0A5; font-weight: 500;">금융 사기/가짜뉴스 판별 AI — KoELECTRA + LoRA</h3>
      <p style="color:#FF3B30; font-weight:bold; margin-top:8px;">헤헿</p>
    </div>
    """)

    with gr.Tabs():

        # ── Tab 1: 판별기 ──
        with gr.TabItem('🔍 실시간 판별기'):
            gr.Markdown('### 뉴스나 문자 문장을 입력하면 AI가 정상/위험을 판별합니다.')

            with gr.Row():
                with gr.Column(scale=3):
                    input_text = gr.Textbox(
                        label='📝 판별할 문장을 입력하세요',
                        placeholder='예: 미공개 호재 입수! 선착순 50명 무조건 수익 보장 리딩방 주소 클릭...',
                        lines=4
                    )

                    # 규제/검증용 필수 체크박스
                    agree_checkbox = gr.Checkbox(
                        label="위 Model Card 탭의 한계점(편향성 및 취약성)을 모두 숙지하였으며, AI 결과를 투자에 직접 활용하지 않을 것에 동의합니다.",
                        value=False
                    )
                    btn = gr.Button('🔍 판별 시작', variant='primary', size='lg')

                with gr.Column(scale=2):
                    output_label = gr.Label(label='📊 판별 결과', num_top_classes=2)
                    output_verdict = gr.Textbox(label='📋 판정', lines=1)
                    output_detail = gr.Textbox(label='💡 안내', lines=3)

            btn.click(
                fn=classify,
                inputs=[input_text, agree_checkbox],
                outputs=[output_label, output_verdict, output_detail]
            )

            gr.Markdown('### 🧪 예시 문장으로 테스트해 보세요')
            gr.Examples(examples=EXAMPLES, inputs=input_text)

            gr.Markdown(
                '> ⚠️ **한계 고지:** 이 AI는 소규모 데이터로 학습되었으며, '
                '교묘한 금융 사기 및 위장형 광고 공격에 취약합니다. '
                '결과를 맹신하지 말고 반드시 금융감독원 등 원문 출처를 확인하세요.'
            )

        # ── Tab 2: Model Card ──
        with gr.TabItem('📄 Model Card'):
            gr.Markdown(MODEL_CARD)

        # ── Tab 3: 스트레스 테스트 ──
        with gr.TabItem('📊 스트레스 테스트'):
            if HAS_STRESS and not df_stress_display.empty:
                gr.Markdown(f'### 📊 첨부파일 스트레스 테스트 결과 — 종합 방어율: {overall_acc:.0%}')
                gr.Dataframe(
                    value=df_stress_display,
                    label='전체 결과 (stress_test_results.csv)',
                    wrap=True,
                    interactive=False
                )

                if not df_failures_display.empty:
                    gr.Markdown(f'### ❌ 실패 케이스 ({len(df_failures_display)}건)')
                    gr.Dataframe(
                        value=df_failures_display,
                        label='실패 케이스 (stress_test_failures.csv)',
                        wrap=True,
                        interactive=False
                    )
            else:
                gr.Markdown(
                    '### ℹ️ 스트레스 테스트 결과 파일이 연동되지 않았습니다.\n'
                    '코랩 왼쪽 폴더 메뉴에 `stress_test_results.csv`와 `stress_test_failures.csv` 파일을 올바르게 업로드했는지 확인해 주세요.'
                )

# ━━━━━━━━━━━━━━━━━━━━━━━━━
# 5. 앱 실행
# ━━━━━━━━━━━━━━━━━━━━━━━━━
if __name__ == "__main__":
    try:
        demo.close()
    except:
        pass
    print('🚀 블랙&그린 텍 테마 금융 사기 판별 대시보드 구동 중…')
    demo.launch(share=True, debug=True)