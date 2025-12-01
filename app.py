"""
ギターIR（Impulse Response）解析 Streamlitアプリケーション
デジタル庁デザインシステム準拠UI
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav
from scipy import signal
from io import BytesIO
import warnings

warnings.filterwarnings('ignore')

# ページ設定
st.set_page_config(
    page_title="Guitar IR Analyzer",
    page_icon="🎸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# デジタル庁デザインシステム準拠のカスタムCSS
st.markdown("""
<style>
    /* デジタル庁デザインシステムのカラーパレット */
    :root {
        --primary-color: #0F4C81;  /* 藍色 */
        --text-color: #1A1A1C;     /* ダークグレー */
        --bg-color: #FFFFFF;        /* 白背景 */
        --secondary-bg: #F7F7F9;    /* ライトグレー背景 */
        --border-color: #D8D8DD;    /* ボーダー */
    }

    /* メインヘッダー */
    h1 {
        color: var(--primary-color);
        font-weight: 700;
        padding-bottom: 1rem;
        border-bottom: 3px solid var(--primary-color);
        margin-bottom: 2rem;
    }

    /* サブヘッダー */
    h2, h3 {
        color: var(--text-color);
        font-weight: 600;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }

    /* サイドバー */
    .css-1d391kg, [data-testid="stSidebar"] {
        background-color: var(--secondary-bg);
        padding: 2rem 1rem;
    }

    /* ファイルアップローダー */
    [data-testid="stFileUploader"] {
        background-color: var(--bg-color);
        border: 2px dashed var(--border-color);
        border-radius: 8px;
        padding: 2rem;
        margin: 1rem 0;
    }

    /* ボタン */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 4px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        transition: opacity 0.2s;
    }

    .stButton > button:hover {
        opacity: 0.85;
    }

    /* スライダー */
    .stSlider {
        padding: 1rem 0;
    }

    /* 情報ボックス */
    .stAlert {
        border-radius: 4px;
        padding: 1rem;
        margin: 1rem 0;
    }

    /* 余白の調整 */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* セレクトボックス */
    .stSelectbox {
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


class ImpulseResponsePlotter:
    """
    インパルス応答データの解析・可視化クラス
    Streamlit対応版（figオブジェクトを返す）
    """

    def __init__(self, original_data, adjusted_data, sample_rate, time_axis, file_names):
        """
        Parameters:
        -----------
        original_data : list of np.ndarray
            オリジナル波形データ（正規化済み）
        adjusted_data : list of np.ndarray
            ピーク調整済み波形データ
        sample_rate : int
            サンプリングレート（Hz）
        time_axis : np.ndarray
            時間軸（ms）
        file_names : list of str
            ファイル名リスト
        """
        self.original_data = original_data
        self.adjusted_data = adjusted_data
        self.sample_rate = sample_rate
        self.time_axis = time_axis
        self.file_names = file_names

    def plot_waveform(self, mode='adjusted'):
        """
        波形をプロット（Streamlit用にfigを返す）

        Parameters:
        -----------
        mode : str
            'original' または 'adjusted'
        """
        data_to_plot = self.adjusted_data if mode == 'adjusted' else self.original_data

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, data in enumerate(data_to_plot):
            ax.plot(self.time_axis, data, label=self.file_names[i], alpha=0.8, linewidth=1.5)

        ax.set_xlabel('時間 (ms)', fontsize=12, fontweight='bold', color='#1A1A1C')
        ax.set_ylabel('振幅（正規化）', fontsize=12, fontweight='bold', color='#1A1A1C')
        ax.set_title(
            f'インパルス応答 波形 - {mode.capitalize()}',
            fontsize=14,
            fontweight='bold',
            color='#0F4C81',
            pad=20
        )
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim([self.time_axis[0], self.time_axis[-1]])

        plt.tight_layout()
        return fig

    def plot_fft(self, mode='adjusted', fft_size=262144, smoothing=None):
        """
        FFT解析結果をプロット（Streamlit用にfigを返す）

        Parameters:
        -----------
        mode : str
            'original' または 'adjusted'
        fft_size : int
            FFTサイズ（2のべき乗）
        smoothing : str or None
            スムージング設定（'1/3', '1/6', '1/12', '1/24', '1/48', None）
        """
        data_to_plot = self.adjusted_data if mode == 'adjusted' else self.original_data

        fig, ax = plt.subplots(figsize=(14, 6))

        for i, data in enumerate(data_to_plot):
            # FFT実行
            fft_result = np.fft.fft(data, n=fft_size)
            frequencies = np.fft.fftfreq(fft_size, d=1/self.sample_rate)

            # 正の周波数のみ抽出
            positive_freq_idx = frequencies > 0
            frequencies = frequencies[positive_freq_idx]
            magnitude = np.abs(fft_result[positive_freq_idx])

            # 20Hz～20kHzの範囲でフィルタリング
            freq_range = (frequencies >= 20) & (frequencies <= 20000)
            frequencies_plot = frequencies[freq_range]
            magnitude_plot = magnitude[freq_range]

            # 正規化（範囲内の最大値を基準）
            if len(magnitude_plot) > 0:
                magnitude_plot = magnitude_plot / np.max(magnitude_plot)

            # スムージング処理
            if smoothing and smoothing != 'なし':
                magnitude_plot = self._apply_smoothing(
                    frequencies_plot,
                    magnitude_plot,
                    smoothing
                )

            # デシベル変換
            magnitude_db = 20 * np.log10(magnitude_plot + 1e-10)

            ax.semilogx(
                frequencies_plot,
                magnitude_db,
                label=self.file_names[i],
                alpha=0.8,
                linewidth=2
            )

        ax.set_xlabel('周波数 (Hz)', fontsize=12, fontweight='bold', color='#1A1A1C')
        ax.set_ylabel('振幅 (dB)', fontsize=12, fontweight='bold', color='#1A1A1C')
        smoothing_text = f' - スムージング: {smoothing}' if smoothing and smoothing != 'なし' else ''
        ax.set_title(
            f'FFT解析結果 - {mode.capitalize()}{smoothing_text}',
            fontsize=14,
            fontweight='bold',
            color='#0F4C81',
            pad=20
        )
        ax.legend(loc='best', fontsize=10, framealpha=0.9)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')
        ax.set_xlim([20, 20000])
        ax.set_ylim([-60, 5])

        plt.tight_layout()
        return fig

    def _apply_smoothing(self, frequencies, magnitude, smoothing_type):
        """
        オクターブバンドスムージング適用

        Parameters:
        -----------
        frequencies : np.ndarray
            周波数配列
        magnitude : np.ndarray
            振幅配列
        smoothing_type : str
            '1/3', '1/6', '1/12', '1/24', '1/48'
        """
        # スムージング倍率のマッピング
        octave_fraction_map = {
            '1/3': 3,
            '1/6': 6,
            '1/12': 12,
            '1/24': 24,
            '1/48': 48
        }

        fraction = octave_fraction_map.get(smoothing_type, 12)

        # オクターブバンドの移動平均
        smoothed = np.zeros_like(magnitude)
        for i, freq in enumerate(frequencies):
            if freq <= 0:
                smoothed[i] = magnitude[i]
                continue

            # 周波数範囲の計算
            f_lower = freq / (2 ** (1 / (2 * fraction)))
            f_upper = freq * (2 ** (1 / (2 * fraction)))

            # 範囲内のインデックス取得
            mask = (frequencies >= f_lower) & (frequencies <= f_upper)
            if np.sum(mask) > 0:
                smoothed[i] = np.mean(magnitude[mask])
            else:
                smoothed[i] = magnitude[i]

        return smoothed


@st.cache_data
def load_wav_file(uploaded_file):
    """
    WAVファイルを読み込み、モノラル化・正規化して返す

    Parameters:
    -----------
    uploaded_file : UploadedFile
        Streamlitのアップロードファイルオブジェクト

    Returns:
    --------
    tuple : (sample_rate, normalized_data)
    """
    try:
        # BytesIOに変換して読み込み
        bytes_data = BytesIO(uploaded_file.read())
        sample_rate, data = wav.read(bytes_data)

        # ステレオの場合はモノラルに変換（平均）
        if len(data.shape) == 2:
            data = np.mean(data, axis=1)

        # 正規化（最大振幅を1.0に）
        data = data.astype(np.float64)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        return sample_rate, data
    except Exception as e:
        st.error(f"ファイル '{uploaded_file.name}' の読み込みに失敗しました: {str(e)}")
        return None, None


def align_peaks(data_list, sample_rate):
    """
    複数のIRデータのピーク位置を揃える

    Parameters:
    -----------
    data_list : list of np.ndarray
        波形データのリスト
    sample_rate : int
        サンプリングレート

    Returns:
    --------
    list of np.ndarray : ピーク調整済みデータ
    """
    # 各データのピーク位置を検出
    peak_positions = [np.argmax(np.abs(data)) for data in data_list]

    # 最も早いピーク位置を基準にする
    earliest_peak = min(peak_positions)

    # 各データをシフトしてピークを揃える
    adjusted_data = []
    for data, peak_pos in zip(data_list, peak_positions):
        shift = peak_pos - earliest_peak
        if shift != 0:
            adjusted = np.roll(data, -shift)
        else:
            adjusted = data.copy()
        adjusted_data.append(adjusted)

    return adjusted_data


def main():
    """メインアプリケーション"""

    # ヘッダー
    st.markdown("# 🎸 Guitar IR Analyzer")
    st.markdown("""
    ### ギターキャビネット インパルス応答 解析ツール
    複数のWAVファイルをアップロードし、波形とFFT解析結果を可視化します。
    """)

    # サイドバー: 設定パネル
    with st.sidebar:
        st.markdown("## ⚙️ 解析設定")
        st.markdown("---")

        # FFTサイズ設定
        st.markdown("### FFT設定")
        fft_size_power = st.select_slider(
            "FFTサイズ",
            options=list(range(14, 20)),
            value=18,
            format_func=lambda x: f"2^{x} ({2**x:,})",
            help="FFTのサイズを選択します。大きいほど周波数分解能が高くなりますが、計算時間が増加します。"
        )
        fft_size = 2 ** fft_size_power

        st.markdown(f"**選択中:** {fft_size:,} サンプル")

        st.markdown("---")

        # スムージング設定
        st.markdown("### スムージング")
        smoothing_option = st.selectbox(
            "オクターブバンド",
            options=['なし', '1/3', '1/6', '1/12', '1/24', '1/48'],
            index=0,
            help="FFT結果に適用するオクターブバンドスムージングを選択します。"
        )
        smoothing = None if smoothing_option == 'なし' else smoothing_option

        st.markdown("---")

        # プロットモード設定
        st.markdown("### 表示モード")
        plot_mode = st.radio(
            "波形調整",
            options=['Original（未調整）', 'Adjusted（ピーク合わせ済み）'],
            index=1,
            help="Originalは元の波形、Adjustedは複数ファイルのピーク位置を揃えた波形を表示します。"
        )
        mode = 'original' if 'Original' in plot_mode else 'adjusted'

        st.markdown("---")
        st.markdown("#### 📘 使い方")
        st.markdown("""
        1. 右のエリアにWAVファイルをドラッグ&ドロップ
        2. 左の設定を調整
        3. グラフが自動更新されます
        """)

    # メインエリア: ファイルアップロード
    st.markdown("## 📁 ファイルアップロード")
    uploaded_files = st.file_uploader(
        "WAVファイルを選択してください（複数選択可）",
        type=['wav'],
        accept_multiple_files=True,
        help="IR（Impulse Response）のWAVファイルをアップロードしてください。"
    )

    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} 個のファイルがアップロードされました")

        # ファイル読み込み
        all_data = []
        all_sample_rates = []
        file_names = []

        with st.spinner('ファイルを読み込んでいます...'):
            for uploaded_file in uploaded_files:
                sample_rate, data = load_wav_file(uploaded_file)
                if sample_rate is not None and data is not None:
                    all_data.append(data)
                    all_sample_rates.append(sample_rate)
                    file_names.append(uploaded_file.name)

        if len(all_data) == 0:
            st.error("有効なWAVファイルが見つかりませんでした。")
            return

        # サンプリングレートの確認
        unique_sample_rates = set(all_sample_rates)
        if len(unique_sample_rates) > 1:
            st.warning(f"⚠️ 異なるサンプリングレートが検出されました: {unique_sample_rates}")
            st.info("最初のファイルのサンプリングレートを基準にします。")

        sample_rate = all_sample_rates[0]

        # データ長を揃える（最短に合わせる）
        min_length = min(len(data) for data in all_data)
        all_data = [data[:min_length] for data in all_data]

        # ピークアライメント
        with st.spinner('ピーク位置を調整しています...'):
            adjusted_data = align_peaks(all_data, sample_rate)

        # 時間軸作成
        time_axis = np.arange(min_length) / sample_rate * 1000  # ms単位

        # プロッタークラスのインスタンス作成
        plotter = ImpulseResponsePlotter(
            original_data=all_data,
            adjusted_data=adjusted_data,
            sample_rate=sample_rate,
            time_axis=time_axis,
            file_names=file_names
        )

        # ファイル情報表示
        with st.expander("📊 ファイル情報", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("サンプリングレート", f"{sample_rate:,} Hz")
            with col2:
                st.metric("データ長", f"{min_length:,} サンプル")
            with col3:
                duration_ms = (min_length / sample_rate) * 1000
                st.metric("長さ", f"{duration_ms:.2f} ms")

            st.markdown("**アップロードファイル:**")
            for i, name in enumerate(file_names, 1):
                st.markdown(f"{i}. `{name}`")

        st.markdown("---")

        # グラフ描画
        st.markdown("## 📈 解析結果")

        # 波形プロット
        st.markdown("### 波形")
        with st.spinner('波形をプロットしています...'):
            fig_waveform = plotter.plot_waveform(mode=mode)
            st.pyplot(fig_waveform)
            plt.close(fig_waveform)

        st.markdown("---")

        # FFTプロット
        st.markdown("### FFT解析（周波数特性）")
        with st.spinner('FFT解析を実行しています...'):
            fig_fft = plotter.plot_fft(mode=mode, fft_size=fft_size, smoothing=smoothing)
            st.pyplot(fig_fft)
            plt.close(fig_fft)

        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666; padding: 2rem;'>
            <p>Powered by Streamlit | デジタル庁デザインシステム準拠</p>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ファイル未アップロード時のプレースホルダー
        st.info("👆 上記のエリアにWAVファイルをアップロードしてください。")

        st.markdown("---")
        st.markdown("### 💡 このツールについて")
        st.markdown("""
        このアプリケーションは、ギターキャビネットのインパルス応答（IR）ファイルを解析し、
        以下の情報を可視化します:

        - **波形表示**: 時間軸での振幅変化
        - **FFT解析**: 周波数特性（20Hz～20kHz）
        - **ピークアライメント**: 複数ファイルの位相を揃えた比較
        - **スムージング**: オクターブバンドでの平滑化

        #### 📝 推奨設定
        - **FFTサイズ**: 2^18（262,144）が標準的
        - **スムージング**: 1/12 または 1/24 が見やすい
        - **ファイル形式**: 44.1kHz または 48kHz のモノラル/ステレオWAV
        """)


if __name__ == "__main__":
    main()
